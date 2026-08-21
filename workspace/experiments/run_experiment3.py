from argparse import ArgumentParser
import os

import numpy as np

from run_experiment1 import load_data, build_prompt_mapping
from run_experiment2 import run_experiment, _resolve_k_values, _resolve_plot_k

from metrics3 import (
    compute_topk_text_retrieval,
    print_topk_results,
    save_topk_csv,
    plot_topk_histogram,
    print_skip_summary,
    save_skip_log_csv,
    compute_confusion_matrix,
    save_confusion_matrix_csv,
    plot_confusion_matrix,
    DEFAULT_K_VALUES,
)

# ------------------------------------------------------------------------------------------
# --------------------------------------- Overview -------------------------------------------
# ------------------------------------------------------------------------------------------
#
# Experiment 3: image mask -> text. For each GT instance mask (the query),
# rank every class in a fixed *word bank* (one text embedding per class) by
# cosine similarity to the mask's visual embedding. Correct if the query's
# true class is among the top-k word-bank entries. Scored the same way as
# Experiment 2 (top-k vs. chance, per-class breakdown, top-1 confusion
# matrix, nearest-neighbor similarity histogram) -- see metrics2's
# compute_topk_text_retrieval docstring for exactly how it reuses Experiment
# 2's machinery.
#
# The GT-mask visual embedding pass itself is IDENTICAL to Experiment 2's --
# this script reuses run_experiment2.run_experiment() directly rather than
# re-implementing the batch/embed loop, and asks it to also embed the word
# bank via the same loaded model before cleanup (see that function's
# text_class_ids/text_class_names docstring).
#
# Word bank membership: this reuses Experiment 1's `custom_prompts` target
# classes (build_prompt_mapping, run_experiment1.py) -- the same class
# definitions Experiment 1 evaluates text-to-mask prediction against --
# rather than a separate `instance_classes` YAML key. Every custom_prompts
# target is a word-bank candidate; only targets with a real GT category id
# can ever actually be a query (see build_prompt_mapping's docstring for
# the synthetic-negative-id case). A class's multiple synonym phrases
# (e.g. `boat: boat, ship`) get collapsed into exactly one word-bank
# embedding per class before scoring -- see _collapse_word_bank_by_class
# below for why, and the per-env branch above it for how each env's
# synonyms get combined differently.

# Only clip/siglip have a text embedding compatible with embed_gt_masks's
# visual embedding space for env="gsam" -- see
# gsam_masks_helper.embed_text_classes's docstring for why gsam/sam3 are
# excluded on architectural grounds, not just "not implemented yet".
#
# clipdino and radio are now wired up too, each via a different route --
# see each helper's embed_text_classes docstring for the details:
#   - "radio": the siglip2 adaptor already has a real text tower
#     (_predict_radio computes text_tokens @ spatial_feats.T directly), so
#     this isn't the text-blind case the old version of this comment
#     assumed -- that assumption only holds for RADIO's native backbone,
#     not the siglip2 adaptor path this codebase actually uses.
#   - "clipdino": has no raw text encoder exposed, so its word bank is a
#     one-hot vector per class in the same background-offset vocab space
#     embed_gt_masks pools into -- similarity-ranking against it reduces to
#     the model's own per-pixel classification. This ONLY works if
#     run_experiment() is called with global_vocab set to the exact same
#     list (same classes, same order) as text_class_names -- see below,
#     where this file does exactly that.
TEXT_CAPABLE_MODELS_BY_ENV = {
    "gsam": ("clip", "siglip"),
    "clipdino": ("clipdino",),
    "radio": ("radio",),
}


def _resolve_min_class_count(yaml_min_class_count):
    # Kept for symmetry with run_experiment2's YAML-driven config, but
    # Experiment 3 doesn't have an analogous knob: "scoreable" here means
    # "class is in the word bank", not "class has >= N instances in the
    # pool" (see compute_topk_text_retrieval's docstring). Present only so
    # a shared exp2/exp3 YAML doesn't need to omit the key.
    return int(yaml_min_class_count) if yaml_min_class_count is not None else 2


def _collapse_word_bank_by_class(raw_word_bank):
    """
    Collapse a word bank that may have multiple rows for the same class
    (one row per custom_prompts synonym phrase, for gsam/radio -- see the
    per-env branch in __main__ below) down to exactly one L2-normalized row
    per unique class id, by mean-pooling that class's rows -- standard CLIP
    prompt-ensembling practice.

    This matters because metrics3.compute_topk_text_retrieval assumes
    exactly one embedding per candidate class: it computes chance level as
    k / num_word_bank_classes, using len(word_bank["class_ids"]) as the
    class count -- if a multi-synonym class occupied 2+ rows, it would
    silently both inflate that class's odds of landing in the top-k AND
    throw off the reported chance baseline for every class. Collapsing
    first keeps that one-row-per-class invariant regardless of how many
    synonyms a class had in the YAML.

    A class with only one row already (a single-synonym class, or any
    class under env="clipdino", which builds one row per class from the
    start -- see __main__ below) passes through unchanged, since
    mean-pooling a single row is a no-op.
    """
    embeddings = np.asarray(raw_word_bank["embeddings"], dtype=np.float64)
    class_ids = np.asarray(raw_word_bank["class_ids"])

    unique_ids = list(dict.fromkeys(class_ids.tolist()))  # preserve first-seen order
    pooled_rows = []
    for cid in unique_ids:
        rows = embeddings[class_ids == cid]
        mean_row = rows.mean(axis=0)
        norm = np.linalg.norm(mean_row)
        if norm > 0:
            mean_row = mean_row / norm
        pooled_rows.append(mean_row)

    return {"embeddings": np.stack(pooled_rows, axis=0), "class_ids": np.asarray(unique_ids)}


def arg_parser():
    parser = ArgumentParser(description="Running Experiment 3: Image Mask to Text Instance Consistency")
    parser.add_argument('conda_env', type=str, default='gsam', help="Enter conda env framework from [gsam, clipdino, radio]")
    parser.add_argument('--image_dataset', type=str, required=True, help="Folderpath to dataset being evaluated on")
    parser.add_argument('--output', type=str, default='', help="Output folder to save all results to")
    parser.add_argument('--image_subset', type=int, default=-1, help="Only evaluate on <image_subset> number of images")
    parser.add_argument('--model', type=str, default=None,
                         help=f"Embedding model to evaluate. Valid per env: {TEXT_CAPABLE_MODELS_BY_ENV}. "
                              f"Defaults to the first valid option for the chosen --conda_env.")
    parser.add_argument('--batch_size', type=int, default=8, help="Image batch size")
    parser.add_argument('--device', type=str, default="cuda:0", help="Device to run models on")
    return parser.parse_args()


if __name__ == "__main__":
    args = arg_parser()

    if args.conda_env not in ["gsam", "clipdino", "radio"]:
        print("Incorrect conda environment name. Should be either [gsam, clipdino, radio]")
        exit()

    text_capable_models = TEXT_CAPABLE_MODELS_BY_ENV[args.conda_env]
    if len(text_capable_models) == 0:
        print(
            f"No text-capable embedding model is wired up for env '{args.conda_env}' yet "
            f"(see the TEXT_CAPABLE_MODELS_BY_ENV note at the top of this file)."
        )
        exit()

    if args.model is None:
        args.model = text_capable_models[0]
        print(f"--model not specified, defaulting to '{args.model}' for env '{args.conda_env}'.")
    elif args.model not in text_capable_models:
        print(f"--model must be one of {text_capable_models} for env '{args.conda_env}' (Experiment 3).")
        exit()

    output_folder = args.output if len(args.output) > 0 else None
    if output_folder:
        os.makedirs(output_folder, exist_ok=True)

    (
        image_paths, ann_ids_list, coco, category_names, ignore_ids,
        gt_class_dict, custom_prompts, void_class_ids, confusion_pairs_yaml,
        instance_class_ids, coco_id_remap, yaml_top_k, yaml_min_class_count, yaml_plot_k
    ) = load_data(
        args.image_dataset,
        image_subset=args.image_subset,
    )

    k_values = _resolve_k_values(yaml_top_k)
    plot_k = _resolve_plot_k(yaml_plot_k, k_values)

    # Word bank membership now comes from the YAML's custom_prompts target
    # classes (same class definitions Experiment 1 evaluates against), not
    # instance_classes. Every custom_prompts target is a word-bank
    # candidate; a target only becomes a QUERY class if it has a real GT
    # category id -- synthetic negative ids (see build_prompt_mapping's
    # docstring) can never have a GT mask, so they can only ever act as
    # distractors in the word bank, never as something to be scored against.
    name_to_id = {v: k for k, v in gt_class_dict.items()}
    prompt_category_dict, prompt_groups = build_prompt_mapping(custom_prompts, name_to_id)

    word_bank_class_ids = list(prompt_category_dict.keys())
    query_class_ids = {cid for cid in word_bank_class_ids if cid in gt_class_dict}

    if not query_class_ids:
        print("No custom_prompts target class has a matching real GT category -- nothing to query.")
        exit()

    # Display names for printing/CSVs: prompt_category_dict's names are
    # authoritative (the custom_prompts target class names as written in
    # the YAML), falling back to gt_class_dict for anything it doesn't
    # cover (shouldn't normally happen, since every query class comes from
    # prompt_category_dict too).
    display_class_dict = {**gt_class_dict, **prompt_category_dict}

    # Per-env vocab construction -- see _collapse_word_bank_by_class's
    # docstring for why gsam/radio and clipdino need different treatment
    # of a class's multiple synonym phrases.
    if args.conda_env == "clipdino":
        # CLIP-DINOiser's vocab is a flat list of per-pixel classification
        # channels competing in ONE joint softmax (see
        # clipdino_masks_helper.embed_text_classes's docstring) -- giving a
        # class two channels (e.g. "boat" and "ship") would just split
        # that class's probability mass between them under softmax
        # competition, unlike gsam/radio's independent per-phrase text
        # embeddings, where extra synonyms are pure signal. So for
        # clipdino only, each class contributes just its FIRST prompt
        # phrase as its one vocab word -- any additional synonyms in the
        # YAML are ignored for this env.
        flat_class_ids = word_bank_class_ids
        flat_class_names = [prompt_groups[cid][0] for cid in word_bank_class_ids]
    else:
        # gsam (clip/siglip) and radio embed each phrase independently
        # (real text-tower forward passes), so every synonym is used --
        # one row per phrase, with duplicated class ids for multi-synonym
        # classes -- and mean-pooled into one embedding per class by
        # _collapse_word_bank_by_class after run_experiment() returns.
        flat_class_ids, flat_class_names = [], []
        for cid in word_bank_class_ids:
            for p in prompt_groups[cid]:
                flat_class_ids.append(cid)
                flat_class_names.append(p)

    dataset_name = os.path.basename(args.image_dataset)

    pool, skip_log, raw_word_bank = run_experiment(
        image_paths,
        ann_ids_list,
        coco,
        category_names,
        ignore_ids,
        query_class_ids,
        env=args.conda_env,
        model_name=args.model,
        batch_size=args.batch_size,
        device=args.device,
        coco_id_remap=coco_id_remap,
        # clipdino's embed_gt_masks needs global_vocab to build the pool's
        # embeddings (it raises if it's None -- gsam/radio just ignore it
        # via **kwargs). It MUST be exactly flat_class_names -- same
        # classes, same order -- because clipdino's embed_text_classes
        # builds a one-hot word bank indexed into that same vocab; passing
        # anything else here would silently misalign the two. See
        # clipdino_masks_helper.embed_text_classes's docstring. Harmless
        # for gsam/radio, which ignore it.
        global_vocab=flat_class_names,
        text_class_ids=flat_class_ids,
        text_class_names=flat_class_names,
    )

    word_bank = _collapse_word_bank_by_class(raw_word_bank)

    results = compute_topk_text_retrieval(pool, word_bank, k_values=k_values)
    confusion = compute_confusion_matrix(results, normalize="row")

    print_topk_results(results, display_class_dict, args.model, dataset_name, experiment_label="EXPERIMENT 3")
    print_skip_summary(skip_log, display_class_dict, num_instances_attempted=len(pool["embeddings"]) + len(skip_log))

    if output_folder:
        csv_path = os.path.join(output_folder, f"{args.model}_{dataset_name}_exp3_results.csv")
        save_topk_csv(results, display_class_dict, csv_path)
        print(f"Saved results to: {csv_path}")

        if skip_log:
            skip_csv_path = os.path.join(output_folder, f"{args.model}_{dataset_name}_exp3_skipped.csv")
            save_skip_log_csv(skip_log, display_class_dict, skip_csv_path)
            print(f"Saved skipped-instance log to: {skip_csv_path}")

        plot_path = os.path.join(output_folder, f"{args.model}_{dataset_name}_exp3_histogram.png")
        plot_topk_histogram(results, display_class_dict, args.model, dataset_name, k=plot_k, save_path=plot_path)

        confusion_csv_path = os.path.join(output_folder, f"{args.model}_{dataset_name}_exp3_confusion.csv")
        save_confusion_matrix_csv(confusion, display_class_dict, confusion_csv_path)
        print(f"Saved confusion matrix to: {confusion_csv_path}")

        confusion_plot_path = os.path.join(output_folder, f"{args.model}_{dataset_name}_exp3_confusion.png")
        plot_confusion_matrix(confusion, display_class_dict, args.model, dataset_name, save_path=confusion_plot_path)
    else:
        plot_topk_histogram(results, display_class_dict, args.model, dataset_name, k=plot_k)
        plot_confusion_matrix(confusion, display_class_dict, args.model, dataset_name)
from argparse import ArgumentParser
import os
import time

import numpy as np

# Same-directory import now that everything's flat -- no path bootstrap needed.
from run_experiment1 import load_data, load_image_and_gt

from metrics2 import (
    new_embedding_pool,
    add_batch_to_pool,
    compute_topk_retrieval,
    print_topk_results,
    save_topk_csv,
    plot_topk_histogram,
    print_skip_summary,
    save_skip_log_csv,
    DEFAULT_K_VALUES,
)

EMBEDDING_MODELS_BY_ENV = {
    "gsam": ("clip", "siglip", "gsam", "sam3"),
    "clipdino": ("clipdino",),
    "radio": ("radio",),
}

# ------------------------------------------------------------------------------------------
# --------------------------------------- Experiment ----------------------------------------
# ------------------------------------------------------------------------------------------

def run_experiment(
    image_paths, ann_ids_list, coco, category_names, ignore_ids,
    query_class_ids, env="gsam", model_name="clip", batch_size=8, device="cuda:0",
    coco_id_remap=None, global_vocab=None,
):
    """
    Pass 1 (this function): decode every image's GT masks, embed each one
    with the model's image encoder, and pool every (embedding, class_id)
    across the whole dataset. Unlike Experiment 1's per-batch accumulator,
    top-k retrieval needs the *entire* pool at once (a query's neighbors can
    live in any image), so nothing is scored until every batch is embedded.

    Returns (pool, skip_log): skip_log is every GT instance embed_gt_masks
    could not embed (e.g. gsam/sam3's masked-pooling collapsing to 0
    feature cells for a small object), each tagged with image_id, class_id,
    and why -- see metrics2.print_skip_summary/save_skip_log_csv.
    """
    if env == "gsam":
        import gsam_masks_helper as helper
    elif env == "clipdino":
        import clipdino_masks_helper as helper
    elif env == "radio":
        import radio_masks_helper as helper
    else:
        raise ValueError("Invalid environment specified.")

    valid_models = EMBEDDING_MODELS_BY_ENV[env]
    if model_name not in valid_models:
        raise ValueError(
            f"Model '{model_name}' is not valid for env '{env}'. Choose from {valid_models}."
        )

    print(f"Running {model_name}...")
    # prompt_class_names is unused by embed_gt_masks, but init_model's clip/siglip
    # loaders take it positionally for parity with Experiment 1's init_model calls.
    model = helper.init_model(model_name, [], device)

    pool = new_embedding_pool()
    skip_log = []  # every GT instance embed_gt_masks couldn't embed, with why -- see metrics2.print_skip_summary
    num_batches = (len(image_paths) + batch_size - 1) // batch_size
    total_runtime = 0.0

    for batch_num in range(num_batches):
        start_idx = batch_num * batch_size
        end_idx = min(start_idx + batch_size, len(image_paths))

        try:
            batch_image_paths = image_paths[start_idx:end_idx]
            batch_ann_ids = ann_ids_list[start_idx:end_idx]

            batch_images, batch_gt_masks, batch_gt_class_ids = [], [], []
            for img_path, ann_ids in zip(batch_image_paths, batch_ann_ids):
                image, masks, class_ids, _ = load_image_and_gt(
                    img_path, ann_ids, coco, category_names, ignore_ids, coco_id_remap
                )

                # Restrict queries to the instance classes named in the YAML
                # (spec item "b"). GT masks for other classes are dropped
                # here entirely -- they can't be a query *or* a neighbor,
                # since "instance consistency" is only meaningful for classes
                # the YAML has flagged as individually-countable instances.
                if query_class_ids is not None:
                    keep = [i for i, cid in enumerate(class_ids) if cid in query_class_ids]
                    masks = [masks[i] for i in keep]
                    class_ids = [class_ids[i] for i in keep]

                batch_images.append(image)
                batch_gt_masks.append(masks)
                batch_gt_class_ids.append(class_ids)

            start_time = time.perf_counter()

            # global_vocab is only used by clipdino's embed_gt_masks; the
            # other helpers accept and ignore it via **kwargs.
            batch_embed_results = helper.embed_gt_masks(
                model_name=model_name,
                model=model,
                images=batch_images,
                batch_gt_masks=batch_gt_masks,
                batch_gt_class_ids=batch_gt_class_ids,
                device=device,
                batch_image_ids=batch_image_paths,
                skip_log=skip_log,
                global_vocab=global_vocab,
            )

            total_runtime += time.perf_counter() - start_time

            add_batch_to_pool(pool, batch_embed_results, batch_image_ids=batch_image_paths)

            del batch_images, batch_gt_masks, batch_gt_class_ids, batch_embed_results

        except Exception as e:
            print(f"WARNING: batch {batch_num + 1}/{num_batches} failed with {type(e).__name__}: {e}")
            continue

    helper.cleanup_model(model)
    print(f"Embedded {len(pool['embeddings'])} GT instances in {total_runtime:.2f} sec.")
    if skip_log:
        print(f"  ({len(skip_log)} GT instances could not be embedded -- see skip summary below)")
    return pool, skip_log


# ------------------------------------------------------------------------------------------
# ------------------------------ YAML-driven retrieval config -------------------------------
# ------------------------------------------------------------------------------------------
#
# top_k / min_class_count / plot_k now live in the dataset's YAML (alongside
# ignore_classes/void_classes/instance_classes/confusion_pairs) instead of
# being CLI flags, since they're properties of how a dataset should be
# scored, not something you'd want to override per invocation. load_data()
# (run_experiment1.py) returns the raw values -- None if the key is absent
# -- and the functions below apply the same defaults the old CLI flags used.

def _resolve_k_values(yaml_top_k):
    """
    Accepts whatever YAML gives us for `top_k`: absent -> DEFAULT_K_VALUES;
    a single int (e.g. `top_k: 3`) -> a one-element tuple; a list (e.g.
    `top_k: [1, 5, 10]`) -> a tuple of ints.
    """
    if yaml_top_k is None:
        return DEFAULT_K_VALUES
    if isinstance(yaml_top_k, (list, tuple)):
        return tuple(int(k) for k in yaml_top_k)
    return (int(yaml_top_k),)


def _resolve_min_class_count(yaml_min_class_count):
    return int(yaml_min_class_count) if yaml_min_class_count is not None else 2


def _resolve_plot_k(yaml_plot_k, k_values):
    return int(yaml_plot_k) if yaml_plot_k is not None else min(k_values)


def arg_parser():
    parser = ArgumentParser(description="Running Experiment 2: Image Mask to Image Mask Instance Consistency")
    parser.add_argument('conda_env', type=str, default='gsam', help="Enter conda env framework from [gsam, clipdino, radio]")
    parser.add_argument('--image_dataset', type=str, required=True, help="Folderpath to dataset being evaluated on")
    parser.add_argument('--output', type=str, default='', help="Output folder to save all results to")
    parser.add_argument('--image_subset', type=int, default=-1, help="Only evaluate on <image_subset> number of images")
    parser.add_argument('--model', type=str, default=None,
                         help=f"Embedding model to evaluate. Valid per env: {EMBEDDING_MODELS_BY_ENV}. "
                              f"Defaults to the only/first valid option for the chosen --conda_env.")
    parser.add_argument('--batch_size', type=int, default=8, help="Image batch size")
    parser.add_argument('--device', type=str, default="cuda:0", help="Device to run models on")
    return parser.parse_args()


if __name__ == "__main__":
    args = arg_parser()

    if args.conda_env not in ["gsam", "clipdino", "radio"]:
        print("Incorrect conda environment name. Should be either [gsam, clipdino, radio]")
        exit()

    valid_models = EMBEDDING_MODELS_BY_ENV[args.conda_env]
    if args.model is None:
        args.model = valid_models[0]
        print(f"--model not specified, defaulting to '{args.model}' for env '{args.conda_env}'.")
    elif args.model not in valid_models:
        print(f"--model must be one of {valid_models} for env '{args.conda_env}'.")
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
    min_class_count = _resolve_min_class_count(yaml_min_class_count)
    plot_k = _resolve_plot_k(yaml_plot_k, k_values)

    # Spec item "b": run over instance masks listed in the YAML. If the
    # dataset's YAML doesn't define instance_classes, fall back to every
    # (non-ignored) GT class -- there's nothing more specific to restrict to.
    if instance_class_ids is not None:
        query_class_ids = set(instance_class_ids)
    else:
        print("No `instance_classes` found in YAML -- evaluating over all GT classes.")
        query_class_ids = set(gt_class_dict.keys())

    dataset_name = os.path.basename(args.image_dataset)

    # Only used by clipdino's embed_gt_masks (see clipdino_masks_helper.py),
    # but harmless to always build: the full dataset vocabulary, not just the
    # query classes, so clipdino's similarity-vector embeddings are as
    # discriminative as possible.
    global_vocab = list(gt_class_dict.values())

    pool, skip_log = run_experiment(
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
        global_vocab=global_vocab,
    )

    results = compute_topk_retrieval(pool, k_values=k_values, min_class_count=min_class_count)

    print_topk_results(results, gt_class_dict, args.model, dataset_name)
    print_skip_summary(skip_log, gt_class_dict, num_instances_attempted=len(pool["embeddings"]) + len(skip_log))

    if output_folder:
        csv_path = os.path.join(output_folder, f"{args.model}_{dataset_name}_exp2_results.csv")
        save_topk_csv(results, gt_class_dict, csv_path)
        print(f"Saved results to: {csv_path}")

        if skip_log:
            skip_csv_path = os.path.join(output_folder, f"{args.model}_{dataset_name}_exp2_skipped.csv")
            save_skip_log_csv(skip_log, gt_class_dict, skip_csv_path)
            print(f"Saved skipped-instance log to: {skip_csv_path}")

        plot_path = os.path.join(output_folder, f"{args.model}_{dataset_name}_exp2_histogram.png")
        plot_topk_histogram(results, gt_class_dict, args.model, dataset_name, k=plot_k, save_path=plot_path)
    else:
        plot_topk_histogram(results, gt_class_dict, args.model, dataset_name, k=plot_k)
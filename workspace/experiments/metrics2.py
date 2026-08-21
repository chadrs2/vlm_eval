import csv
import os
from collections import Counter
from math import comb
import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------------------------------------
# ----------------------------------------- Overview ----------------------------------------
# ------------------------------------------------------------------------------------------
#
# Experiment 2 asks a different question than Experiment 1's AP: given a
# *perfect* GT mask crop and nothing else (no text prompt), does the visual
# embedding space put same-class objects near each other?
#
#   For each GT instance mask (the "query"), rank every *other* GT instance
#   mask in the dataset by cosine similarity of their embeddings. If any of
#   the top-k neighbors share the query's class, the query counts as
#   correct. Accuracy is the fraction of correct queries -- overall and
#   per-class.
#
# This is leave-one-out retrieval over the whole embedding pool, not a
# per-image metric, so unlike metrics.py there's no "accumulate per batch,
# stream to disk" story: embeddings from every image must be collected
# first, then similarity is computed once at the end.

DEFAULT_K_VALUES = (1, 5, 10)


# ------------------------------------------------------------------------------------------
# ------------------------------------- Accumulation ----------------------------------------
# ------------------------------------------------------------------------------------------

def new_embedding_pool():
    """A fresh accumulator to collect embeddings across batches/images."""
    return {
        "embeddings": [],
        "class_ids": [],
        "image_ids": [],
    }


def add_batch_to_pool(pool, batch_embed_results, batch_image_ids=None):
    """
    batch_embed_results: output of helper.embed_gt_masks -- a list (per image)
    of lists of {"embedding": np.ndarray, "class_id": int}.
    batch_image_ids: optional list of identifiers (e.g. image path or COCO id),
    parallel to batch_embed_results, purely for traceability in the CSV dump.
    """
    if batch_image_ids is None:
        batch_image_ids = [None] * len(batch_embed_results)

    for image_id, instances in zip(batch_image_ids, batch_embed_results):
        for inst in instances:
            pool["embeddings"].append(np.asarray(inst["embedding"]))
            pool["class_ids"].append(inst["class_id"])
            pool["image_ids"].append(image_id)


# ------------------------------------------------------------------------------------------
# --------------------------------------- Chance Level ----------------------------------------
# ------------------------------------------------------------------------------------------

def _chance_topk_for_class(n_class, pool_size, k):
    """
    Analytic (not simulated) chance-level top-k accuracy for a class with
    `n_class` total instances in a pool of `pool_size`, under the null
    hypothesis that the embedding carries no class information -- i.e. the
    top-k "neighbors" are a uniform random sample of the remaining pool
    rather than the actual nearest ones.

    Setup mirrors the real leave-one-out draw exactly: remove the query
    itself, leaving `pool_size - 1` candidates, of which `n_class - 1` are
    still the query's class. Chance accuracy is the probability that a
    random k-sample from those candidates contains at least one same-class
    hit -- i.e. 1 minus the probability every sampled candidate is a
    *different* class (hypergeometric, sampling without replacement):

        P(hit) = 1 - C(other, k) / C(remaining, k)

    where `remaining = pool_size - 1` and `other = remaining - (n_class - 1)`
    is how many different-class candidates exist to draw from.

    This is exactly the baseline a class earns purely from its share of the
    dataset, independent of embedding quality -- so `observed - chance`
    isolates what the embedding space is actually contributing, which is
    what's comparable across datasets with different class distributions
    (e.g. coastal vs. terrestrial).
    """
    remaining = pool_size - 1
    if remaining <= 0:
        return float("nan")

    k = min(k, remaining)
    other = remaining - (n_class - 1)
    other = max(other, 0)

    if k > other:
        # Fewer different-class candidates than k -- a same-class hit is
        # guaranteed even under random sampling.
        return 1.0

    return 1.0 - comb(other, k) / comb(remaining, k)


# ------------------------------------------------------------------------------------------
# ----------------------------------- Top-K Retrieval ----------------------------------------
# ------------------------------------------------------------------------------------------

def compute_topk_retrieval(pool, k_values=DEFAULT_K_VALUES, min_class_count=2):
    """
    Leave-one-out top-k class-retrieval accuracy over every embedding in `pool`.

    min_class_count: classes with fewer than this many total instances in the
    dataset are excluded from scoring. A class with only 1 total instance can
    NEVER be correctly retrieved (there is no other same-class mask to find,
    so accuracy is floored at 0 no matter how good the embeddings are). This
    is a dataset cardinality issue, not a model failure -- those classes are
    reported separately as "unscoreable" rather than dragging the mean down.

    Returns a results dict; see `_build_results_dict` for the exact shape.
    """
    embeddings = np.asarray(pool["embeddings"], dtype=np.float64)
    class_ids = np.asarray(pool["class_ids"])
    n = len(class_ids)

    if n == 0:
        return _build_results_dict({}, {}, {}, [], 0, 0, k_values)

    # Cosine similarity -- normalize defensively even though embeddings
    # coming out of the helper (CLIP/SigLIP/GSAM/SAM3) are already L2-normalized.
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1e-12
    normed = embeddings / norms
    sim_matrix = normed @ normed.T
    np.fill_diagonal(sim_matrix, -np.inf)  # a mask can't retrieve itself

    class_counts = {}
    for cid in class_ids:
        class_counts[cid] = class_counts.get(cid, 0) + 1

    scoreable = np.array([class_counts[cid] >= min_class_count for cid in class_ids])
    max_k = max(k_values)

    per_instance_correct = {k: np.zeros(n, dtype=bool) for k in k_values}
    # Also track, for every scoreable query, the similarity of its single
    # nearest neighbor and whether that nearest neighbor was correct --
    # this feeds the "correct vs incorrect nearest-neighbor similarity"
    # histogram, which is a more informative diagnostic than accuracy alone.
    nn_similarity = np.full(n, np.nan)
    nn_correct = np.zeros(n, dtype=bool)

    for i in range(n):
        if not scoreable[i]:
            continue
        order = np.argsort(-sim_matrix[i])[:max_k]
        retrieved_classes = class_ids[order]
        query_class = class_ids[i]

        for k in k_values:
            per_instance_correct[k][i] = bool(np.any(retrieved_classes[:k] == query_class))

        nn_similarity[i] = sim_matrix[i, order[0]]
        nn_correct[i] = bool(retrieved_classes[0] == query_class)

    per_class_topk = {k: {} for k in k_values}
    for k in k_values:
        for cid in set(class_ids.tolist()):
            mask = (class_ids == cid) & scoreable
            per_class_topk[k][cid] = (
                float(per_instance_correct[k][mask].mean()) if mask.sum() > 0 else np.nan
            )

    overall_topk = {}
    for k in k_values:
        overall_topk[k] = (
            float(per_instance_correct[k][scoreable].mean()) if scoreable.sum() > 0 else np.nan
        )

    # Chance-level baseline: what each class's top-k accuracy would be if
    # neighbors were random rather than nearest-by-embedding, given only its
    # count in the pool. Computed for every class (not just scoreable ones)
    # so it's available if min_class_count is later loosened; overall_chance
    # is aggregated with the identical per-instance weighting as
    # overall_topk so "observed - chance" is a fair excess-over-chance figure.
    per_class_chance = {k: {} for k in k_values}
    for k in k_values:
        for cid in set(class_ids.tolist()):
            per_class_chance[k][cid] = _chance_topk_for_class(class_counts[cid], n, k)

    overall_chance = {}
    for k in k_values:
        if scoreable.sum() > 0:
            chance_per_scoreable_instance = np.array(
                [per_class_chance[k][cid] for cid in class_ids[scoreable]]
            )
            overall_chance[k] = float(chance_per_scoreable_instance.mean())
        else:
            overall_chance[k] = float("nan")

    unscoreable_classes = sorted({cid for cid in set(class_ids.tolist()) if class_counts[cid] < min_class_count})

    return _build_results_dict(
        per_class_topk, overall_topk, class_counts, unscoreable_classes,
        n, int(scoreable.sum()), k_values,
        nn_similarity=nn_similarity[scoreable], nn_correct=nn_correct[scoreable],
        per_class_chance=per_class_chance, overall_chance=overall_chance,
    )


def _build_results_dict(per_class_topk, overall_topk, class_counts, unscoreable_classes,
                         num_instances, num_scoreable, k_values, nn_similarity=None, nn_correct=None,
                         per_class_chance=None, overall_chance=None):
    return {
        "per_class_topk": per_class_topk,
        "overall_topk": overall_topk,
        "class_counts": class_counts,
        "unscoreable_classes": unscoreable_classes,
        "num_instances": num_instances,
        "num_scoreable": num_scoreable,
        "k_values": tuple(k_values),
        "nn_similarity": nn_similarity if nn_similarity is not None else np.array([]),
        "nn_correct": nn_correct if nn_correct is not None else np.array([]),
        "per_class_chance": per_class_chance if per_class_chance is not None else {k: {} for k in k_values},
        "overall_chance": overall_chance if overall_chance is not None else {k: float("nan") for k in k_values},
    }


# ------------------------------------------------------------------------------------------
# ------------------------------------------- Display ----------------------------------------
# ------------------------------------------------------------------------------------------

def print_topk_results(results, category_name_dict, model_name, dataset_name):
    print("\n" + "=" * 80)
    print(f"EXPERIMENT 2 RESULTS: {model_name} ({dataset_name})")
    print("=" * 80)

    print(f"Total GT instances:      {results['num_instances']}")
    print(f"Scoreable instances:     {results['num_scoreable']}  "
          f"(excludes classes with < min_class_count total instances)")

    for k in results["k_values"]:
        obs = results['overall_topk'][k]
        chance = results['overall_chance'][k]
        excess = obs - chance if not (np.isnan(obs) or np.isnan(chance)) else float("nan")
        print(f"Top-{k} accuracy:          {obs:.4f}   (chance: {chance:.4f}, excess: {excess:+.4f})")

    print("\nPer-class top-k accuracy (obs / chance / excess):")
    header = f"{'Class':<22}{'Count':>8}"
    for k in results["k_values"]:
        header += f"{('Top-' + str(k) + ' obs'):>12}{('chance'):>10}{('excess'):>10}"
    print("-" * len(header))
    print(header)
    print("-" * len(header))

    scored_ids = sorted(
        (cid for cid in results["class_counts"] if cid not in results["unscoreable_classes"]),
        key=lambda cid: category_name_dict.get(cid, str(cid)).lower(),
    )
    for cid in scored_ids:
        name = category_name_dict.get(cid, str(cid))
        row = f"{name:<22}{results['class_counts'][cid]:>8}"
        for k in results["k_values"]:
            obs = results['per_class_topk'][k].get(cid, float('nan'))
            chance = results['per_class_chance'][k].get(cid, float('nan'))
            excess = obs - chance if not (np.isnan(obs) or np.isnan(chance)) else float("nan")
            row += f"{obs:>12.3f}{chance:>10.3f}{excess:>+10.3f}"
        print(row)

    if results["unscoreable_classes"]:
        print("\nUnscoreable classes (< min_class_count instances in dataset):")
        for cid in results["unscoreable_classes"]:
            name = category_name_dict.get(cid, str(cid))
            print(f"  {name} (count={results['class_counts'][cid]})")

    print("=" * 80)


def save_topk_csv(results, category_name_dict, csv_path):
    fieldnames = ["class_id", "class_name", "count", "scoreable"]
    for k in results["k_values"]:
        fieldnames += [f"top_{k}_accuracy", f"top_{k}_chance", f"top_{k}_excess"]

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for cid, count in sorted(results["class_counts"].items(), key=lambda kv: category_name_dict.get(kv[0], str(kv[0])).lower()):
            scoreable = cid not in results["unscoreable_classes"]
            row = {
                "class_id": cid,
                "class_name": category_name_dict.get(cid, str(cid)),
                "count": count,
                "scoreable": scoreable,
            }
            for k in results["k_values"]:
                obs = results["per_class_topk"][k].get(cid, float("nan")) if scoreable else None
                chance = results["per_class_chance"][k].get(cid, float("nan"))
                row[f"top_{k}_accuracy"] = obs if obs is not None else ""
                row[f"top_{k}_chance"] = chance
                row[f"top_{k}_excess"] = (obs - chance) if (obs is not None and not np.isnan(obs)) else ""
            writer.writerow(row)

        writer.writerow({})
        overall_row = {"class_id": "OVERALL", "class_name": "", "count": results["num_scoreable"], "scoreable": ""}
        for k in results["k_values"]:
            obs = results["overall_topk"][k]
            chance = results["overall_chance"][k]
            overall_row[f"top_{k}_accuracy"] = obs
            overall_row[f"top_{k}_chance"] = chance
            overall_row[f"top_{k}_excess"] = obs - chance if not (np.isnan(obs) or np.isnan(chance)) else ""
        writer.writerow(overall_row)


# ------------------------------------------------------------------------------------------
# --------------------------------------- Skip Log ------------------------------------------
# ------------------------------------------------------------------------------------------
#
# gsam/sam3's dense-feature masked-pooling path can silently drop a GT
# instance if it resizes down to 0 cells on the encoder's feature grid
# (object too small relative to that scale's stride) -- unlike the
# clip/siglip crop-based path, which only drops truly-empty masks. Left
# unmeasured, this means different models are scored on different subsets
# of the same GT pool, biased toward dropping the smallest objects, and
# unevenly so across models. These functions turn embed_gt_masks's
# optional `skip_log` (see gsam_masks_helper.py) into a reportable summary.

def summarize_skips(skip_log):
    """skip_log: list of {"image_id", "class_id", "reason"} dicts (or None/empty)."""
    by_class = Counter(rec["class_id"] for rec in skip_log)
    by_reason = Counter(rec["reason"] for rec in skip_log)
    return {"total": len(skip_log), "by_class": dict(by_class), "by_reason": dict(by_reason)}


def print_skip_summary(skip_log, category_name_dict, num_instances_attempted):
    """
    num_instances_attempted: total GT instances handed to embed_gt_masks
    (i.e. len(pool["embeddings"]) + len(skip_log)) -- the denominator for
    "what fraction of the intended pool did this model actually score."
    """
    if not skip_log:
        return

    summary = summarize_skips(skip_log)
    pct = 100 * summary["total"] / num_instances_attempted if num_instances_attempted > 0 else float("nan")

    print("\n" + "-" * 80)
    print(f"SKIPPED GT INSTANCES: {summary['total']} / {num_instances_attempted} "
          f"({pct:.1f}%) could not be embedded")
    print("-" * 80)

    print("By class:")
    for cid, cnt in sorted(summary["by_class"].items(), key=lambda kv: -kv[1]):
        name = category_name_dict.get(cid, str(cid))
        print(f"  {name:<22}{cnt:>4} skipped")

    print("By reason:")
    for reason, cnt in sorted(summary["by_reason"].items(), key=lambda kv: -kv[1]):
        print(f"  {cnt:>4}  {reason}")

    print("-" * 80)


def save_skip_log_csv(skip_log, category_name_dict, csv_path):
    """Raw per-instance skip records, for tracing exactly which GT instances
    a given model dropped -- useful to confirm skips concentrate in specific
    small-object classes (e.g. buoy/float) across coastal vs. terrestrial
    datasets rather than being spread evenly."""
    fieldnames = ["image_id", "class_id", "class_name", "reason"]

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rec in skip_log:
            writer.writerow({
                "image_id": rec.get("image_id"),
                "class_id": rec["class_id"],
                "class_name": category_name_dict.get(rec["class_id"], str(rec["class_id"])),
                "reason": rec["reason"],
            })


def plot_topk_histogram(results, category_name_dict, model_name, dataset_name, k=1, save_path=None):
    """
    Two-panel figure:
      left  -- bar chart of per-class top-k accuracy (the "histogram" of
               accuracy across classes), sorted worst to best.
      right -- true histogram of nearest-neighbor cosine similarity, split
               into "correct retrieval" vs "incorrect retrieval" -- this is
               the more diagnostic view: it shows whether the embedding
               space actually separates same-class from different-class
               neighbors, or whether correct/incorrect retrievals sit at
               similar similarity values (i.e. the model is guessing).
    """
    if k not in results["per_class_topk"]:
        raise ValueError(f"k={k} was not computed. Available k values: {results['k_values']}")

    scored_ids = [cid for cid in results["class_counts"] if cid not in results["unscoreable_classes"]]
    scored_ids.sort(key=lambda cid: results["per_class_topk"][k].get(cid, float("nan")))
    names = [category_name_dict.get(cid, str(cid)) for cid in scored_ids]
    accs = [results["per_class_topk"][k][cid] for cid in scored_ids]
    chances = [results["per_class_chance"][k].get(cid, float("nan")) for cid in scored_ids]

    fig, axes = plt.subplots(1, 2, figsize=(16, max(6, 0.35 * len(names))))

    ax = axes[0]
    ax.barh(names, accs, color="#4C72B0", label="observed", zorder=2)
    # Per-class chance level, marked directly on each bar -- this is what
    # makes class-imbalance effects visible: a class near its chance mark
    # is being "explained" by its share of the pool, not by the embedding.
    ax.scatter(chances, names, marker="|", s=400, linewidths=2, color="#333333",
               label="chance level", zorder=3)
    ax.axvline(results["overall_topk"][k], color="red", linestyle="--", linewidth=1.5,
               label=f"overall top-{k} = {results['overall_topk'][k]:.3f}")
    ax.axvline(results["overall_chance"][k], color="#333333", linestyle=":", linewidth=1.5,
               label=f"overall chance = {results['overall_chance'][k]:.3f}")
    ax.set_xlim(0, 1)
    ax.set_xlabel(f"Top-{k} accuracy")
    ax.set_title(f"Per-class top-{k} accuracy vs. chance\n{model_name} ({dataset_name})")
    ax.legend(loc="lower right", fontsize=9)

    ax = axes[1]
    nn_sim = results["nn_similarity"]
    nn_correct = results["nn_correct"]
    if len(nn_sim) > 0:
        bins = np.linspace(-1, 1, 41)
        ax.hist(nn_sim[nn_correct], bins=bins, alpha=0.6, label="correct (top-1)", color="#55A868")
        ax.hist(nn_sim[~nn_correct], bins=bins, alpha=0.6, label="incorrect (top-1)", color="#C44E52")
        ax.set_xlabel("Nearest-neighbor cosine similarity")
        ax.set_ylabel("Count")
        ax.set_title("Nearest-neighbor similarity: correct vs incorrect")
        ax.legend()
    else:
        ax.text(0.5, 0.5, "No scoreable instances", ha="center", va="center")

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        print(f"Saved histogram to: {save_path}")
    plt.show()
    plt.close(fig)
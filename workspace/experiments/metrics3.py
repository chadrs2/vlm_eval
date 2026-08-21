import numpy as np

# Experiment 3 reuses Experiment 2's pool accumulation, results-dict shape,
# and every display/plotting/confusion-matrix function as-is -- only the
# retrieval computation itself (querying a word bank instead of leave-one-out
# against the rest of the pool) is Experiment-3-specific, so that's the only
# thing defined in this file. Re-exported here so run_experiment3.py can
# import everything it needs from one place.
from metrics2 import (
    DEFAULT_K_VALUES,
    new_embedding_pool,
    add_batch_to_pool,
    _build_results_dict,
    print_topk_results,
    save_topk_csv,
    plot_topk_histogram,
    print_skip_summary,
    save_skip_log_csv,
    compute_confusion_matrix,
    save_confusion_matrix_csv,
    plot_confusion_matrix,
)

# ------------------------------------------------------------------------------------------
# --------------------------- Top-K Retrieval, Image -> Text (Exp 3) ------------------------
# ------------------------------------------------------------------------------------------
#
# Experiment 3 asks a different question again: given a *perfect* GT mask
# crop and nothing else, does the embedding put it near its OWN class's
# text prompt out of a fixed word bank -- rather than near other GT mask
# crops (Experiment 2). The candidates being ranked are the word bank's
# class embeddings, not other queries, so unlike compute_topk_retrieval
# there's no leave-one-out (the word bank doesn't contain the query) and no
# per-class instance-count effect on chance level: every class gets exactly
# one word-bank slot regardless of how many GT instances of it exist in the
# dataset, so chance is just k / (num word bank classes) for every class,
# uniformly.
#
# Reuses compute_topk_retrieval's `_build_results_dict` shape exactly (via
# metrics2), so every Experiment 2 display/plotting/confusion-matrix
# function re-exported above works unchanged on Experiment 3 results too --
# pass experiment_label="EXPERIMENT 3" to print_topk_results for an accurate
# header.

def compute_topk_text_retrieval(pool, word_bank, k_values=DEFAULT_K_VALUES):
    """
    Leave-one-out-style top-k retrieval, but querying a fixed word bank of
    class text embeddings instead of the rest of the pool.

    pool: an embedding pool as built by new_embedding_pool/add_batch_to_pool
    -- the GT mask visual embeddings (the queries).

    word_bank: {"embeddings": (C, D) array, "class_ids": length-C array/list},
    one text embedding per candidate class, as returned by
    helper.embed_text_classes (paired with the class ids you passed it).

    A query is "scoreable" only if its true class has an entry in the word
    bank at all -- there's no min_class_count knob here (that's an
    Experiment-2-only concept about *other instances* in the pool; the word
    bank already has exactly one slot per class it contains).
    """
    embeddings = np.asarray(pool["embeddings"], dtype=np.float64)
    class_ids = np.asarray(pool["class_ids"])
    n = len(class_ids)

    word_bank_embeddings = np.asarray(word_bank["embeddings"], dtype=np.float64)
    word_bank_class_ids = np.asarray(word_bank["class_ids"])
    num_word_bank_classes = len(word_bank_class_ids)

    if n == 0 or num_word_bank_classes == 0:
        return _build_results_dict({}, {}, {}, [], n, 0, k_values)

    img_norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    img_norms[img_norms == 0] = 1e-12
    normed_img = embeddings / img_norms

    wb_norms = np.linalg.norm(word_bank_embeddings, axis=1, keepdims=True)
    wb_norms[wb_norms == 0] = 1e-12
    normed_wb = word_bank_embeddings / wb_norms

    sim_matrix = normed_img @ normed_wb.T  # (n, C) -- no self-similarity to mask out

    class_counts = {}
    for cid in class_ids:
        class_counts[cid] = class_counts.get(cid, 0) + 1

    in_word_bank = set(word_bank_class_ids.tolist())
    scoreable = np.array([cid in in_word_bank for cid in class_ids])
    # "Unscoreable" here means "true class isn't in the word bank at all",
    # not the Experiment-2 sense of "too few instances in the pool".
    unscoreable_classes = sorted({cid for cid in set(class_ids.tolist()) if cid not in in_word_bank})

    max_k = min(max(k_values), num_word_bank_classes)

    per_instance_correct = {k: np.zeros(n, dtype=bool) for k in k_values}
    nn_similarity = np.full(n, np.nan)
    nn_correct = np.zeros(n, dtype=bool)
    nn_predicted_class = np.full(n, -1, dtype=word_bank_class_ids.dtype)

    for i in range(n):
        if not scoreable[i]:
            continue
        order = np.argsort(-sim_matrix[i])[:max_k]
        retrieved_classes = word_bank_class_ids[order]
        query_class = class_ids[i]

        for k in k_values:
            kk = min(k, num_word_bank_classes)
            per_instance_correct[k][i] = bool(np.any(retrieved_classes[:kk] == query_class))

        nn_similarity[i] = sim_matrix[i, order[0]]
        nn_correct[i] = bool(retrieved_classes[0] == query_class)
        nn_predicted_class[i] = retrieved_classes[0]

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

    # Chance level: k distinct classes drawn uniformly at random out of a
    # fixed C-class word bank (no leave-one-out adjustment needed, since the
    # word bank never contains the query itself) gives P(hit) = k / C
    # exactly -- the same figure for every class, unlike Experiment 2 where
    # chance tracked each class's share of the pool.
    per_class_chance = {k: {} for k in k_values}
    for k in k_values:
        kk = min(k, num_word_bank_classes)
        chance = kk / num_word_bank_classes
        for cid in set(class_ids.tolist()):
            per_class_chance[k][cid] = chance

    overall_chance = {k: (min(k, num_word_bank_classes) / num_word_bank_classes) for k in k_values}

    return _build_results_dict(
        per_class_topk, overall_topk, class_counts, unscoreable_classes,
        n, int(scoreable.sum()), k_values,
        nn_similarity=nn_similarity[scoreable], nn_correct=nn_correct[scoreable],
        per_class_chance=per_class_chance, overall_chance=overall_chance,
        nn_query_class=class_ids[scoreable], nn_predicted_class=nn_predicted_class[scoreable],
    )
"""Curriculum order generation for TinyLibrary.

An "order" is the full multi-epoch sequence of manifest indices a run will
consume. Staged and random runs traverse every sample once per epoch;
competence-based runs make the same number of draws with replacement and can
therefore differ in sample coverage and realized token mass.

Orders (difficulty source):
    developmental       inferred book-level age band, 3_5 -> 12
    antidevelopmental   reversed age bands
    readability         per-sample Flesch-Kincaid grade
    perplexity          per-sample teacher-LM NLL (needs manifest_scored.jsonl)
    random              no curriculum (baseline)
    randomblock         control for the block-structure confound: blocks match
                        the developmental band stages in token mass, order and
                        epoch-to-epoch repetition, but content is a fixed
                        random partition (staged-only)

Schedules:
    staged      4 sequential stages (age bands, or quantile buckets for
                continuous scores), shuffled within stage
    competence  competence-based sampling (Platanios et al., 2019): at draw t,
                sample uniformly from the easiest c(t) fraction,
                c(t) = sqrt(t/T * (1 - c0^2) + c0^2)
"""
import numpy as np

AGE_BANDS = ["3_5", "6_8", "9_12", "12"]
N_STAGES = 4


def difficulty(records, order):
    """Per-sample difficulty scores; lower = easier = earlier."""
    if order in ("developmental", "antidevelopmental"):
        scores = np.array([AGE_BANDS.index(r["age_band"]) for r in records], dtype=np.float64)
    elif order == "readability":
        scores = np.array([r["fk_grade"] for r in records])
    elif order == "perplexity":
        scores = np.array([r["nll"] for r in records])
    elif order == "qwenppl":
        # out-of-family teacher (Qwen3-0.6B) NLL — teacher-robustness check
        scores = np.array([r["nll_qwen"] for r in records])
    elif order == "gemmappl":
        # second out-of-family teacher (Gemma-3-270M) — closer to the SmolLM2
        # ordering (rho .96 vs .91); dose-response point for benefit transfer
        scores = np.array([r["nll_gemma"] for r in records])
    elif order == "random":
        scores = np.zeros(len(records))
    else:
        raise ValueError(order)
    if order == "antidevelopmental":
        scores = -scores
    return scores


def _stages(scores, rng):
    """Split indices into N_STAGES easy->hard groups.

    Discrete scores (age bands) map to their natural groups; continuous scores
    are cut at quantiles so stages have equal size. Ties are broken randomly.
    """
    jitter = rng.random(len(scores)) * 1e-9
    ranks = np.argsort(scores + jitter, kind="stable")
    uniq = np.unique(scores)
    if len(uniq) <= N_STAGES:  # discrete: keep natural groups (unequal sizes)
        return [np.flatnonzero(scores == u) for u in np.sort(uniq)]
    return np.array_split(ranks, N_STAGES)


def _tokens(rec):
    return rec.get("n_tokens") or int(rec["n_words"] * 1.35)


def build_order(records, order, schedule, epochs, seed, competence_c0=0.1):
    """Return the full index sequence for a run (len = n_samples * epochs)."""
    rng = np.random.default_rng(seed)
    n = len(records)
    total = n * epochs

    if order == "random":
        return np.concatenate([rng.permutation(n) for _ in range(epochs)])

    if order == "randomblock":
        # Same block skeleton as developmental/staged (band token masses,
        # easy->hard order, identical blocks every epoch, shuffled within),
        # but block membership is a seed-fixed random partition. Separates
        # "long repeated homogeneous blocks" from "age-band content".
        if schedule != "staged":
            raise ValueError("randomblock is a staged-only control")
        perm = rng.permutation(n)
        cum_tokens = np.cumsum([_tokens(records[i]) for i in perm])
        band_mass = [sum(_tokens(r) for r in records if r["age_band"] == b)
                     for b in AGE_BANDS]
        cuts = np.searchsorted(cum_tokens, np.cumsum(band_mass)[:-1])
        blocks = np.split(perm, cuts)
        return np.concatenate([np.concatenate([rng.permutation(b) for b in blocks])
                               for _ in range(epochs)])

    scores = difficulty(records, order)

    if schedule == "staged":
        # Each epoch traverses all stages easy->hard, shuffled within stage.
        epochs_out = []
        for _ in range(epochs):
            parts = [rng.permutation(stage) for stage in _stages(scores, rng)]
            epochs_out.append(np.concatenate(parts))
        return np.concatenate(epochs_out)

    if schedule == "competence":
        jitter = rng.random(n) * 1e-9
        sorted_idx = np.argsort(scores + jitter, kind="stable")
        t = np.arange(total, dtype=np.float64)
        c = np.sqrt(t / total * (1 - competence_c0 ** 2) + competence_c0 ** 2)
        pool_sizes = np.maximum((c * n).astype(np.int64), 1)
        draws = (rng.random(total) * pool_sizes).astype(np.int64)
        return sorted_idx[draws]

    raise ValueError(schedule)


def summarize_order(records, order_seq, n_chunks=10):
    """Mean age-band index per chunk of the run — sanity check that the
    curriculum actually moves easy->hard."""
    band_idx = np.array([AGE_BANDS.index(r["age_band"]) for r in records])
    chunks = np.array_split(band_idx[order_seq], n_chunks)
    return [round(float(c.mean()), 3) for c in chunks]

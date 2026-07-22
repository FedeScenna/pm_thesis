"""
Outcome-oriented predictive process monitoring for the recruiting log
=====================================================================

Prefix-based hire / not-hire prediction, following:

* Teinemaa et al. (2019) — outcome-oriented predictive process monitoring:
  a *single classifier* trained on prefixes with a lossy *aggregation* (activity
  frequency) encoding is the most reliable configuration; AUC-ROC is the primary
  metric under class imbalance.
* Weytjens & De Weerdt (2021, sec. 5.5-5.6) — *strict temporal splitting*: keep in
  the training set only cases that COMPLETE before the separation time, and carry
  the still-running cases into the test set as their observed prefixes (the
  "point-in-time, predict cases not yet completed" scenario).
* Ceravolo et al. (2024) — imbalance handling (resampling vs. cost-sensitive).

The public entry points used by the notebook are, in order:

    load_event_log -> build_case_table -> temporal_split -> build_prefix_dataset
    -> (per encoding) build_features -> run_grid -> plots / write_summary

Everything that must be *fit on the training set only* (activity vocabulary,
bigram vocabulary, static label encoders, SMOTE-NC, scalers, models) is fit on the
train split exclusively; validation/test are only transformed.
"""
from __future__ import annotations

import os
import sys
import time
import warnings
from collections import Counter
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
SEED = 42

# Column names in data/event_log_consolidated.csv
CASE_COL = "Case_id"
ACT_COL = "Step"
TIME_COL = "timestamp"
LABEL_COL = "hired"
STATIC_COLS = ["Recruiting Agency", "Region", "Country", "Job Family", "Job Family Group"]
# Columns that directly encode the outcome -> NEVER used as features (leakage).
LEAKY_COLS = ["Rejected", "Disposition Reason", "All Stages for Candidate Current and Completed"]

# Candidate locations of the consolidated event log (worktree has no data/ folder;
# the real data lives in the main working tree three levels up).
DATA_CANDIDATES = [
    os.path.join("data", "event_log_consolidated.csv"),
    os.path.join("..", "..", "..", "data", "event_log_consolidated.csv"),
    r"C:\Users\feder\OneDrive\Documents\pm_thesis\data\event_log_consolidated.csv",
]

ENCODINGS = ["boolean", "frequency", "bigram"]
STRATEGIES = ["none", "smotenc", "classweight"]
MODELS = ["rf", "xgb", "lgbm"]


def resolve_data_path(explicit: str | None = None) -> str:
    """Return the first existing consolidated-log path."""
    cands = ([explicit] if explicit else []) + DATA_CANDIDATES
    for p in cands:
        if p and os.path.exists(p):
            return p
    raise FileNotFoundError(
        "Could not locate event_log_consolidated.csv. Tried: " + ", ".join(str(c) for c in cands)
    )


# ----------------------------------------------------------------------------
# 1. Load & case table
# ----------------------------------------------------------------------------
def load_event_log(path: str | None = None, nrows: int | None = None) -> pd.DataFrame:
    """Load the consolidated event log, keep only the columns we need, drop NaN
    activities, parse timestamps and sort by (case, time).

    Only feature-relevant columns are read; the leaky columns are never loaded.
    """
    path = resolve_data_path(path)
    usecols = [CASE_COL, ACT_COL, TIME_COL, LABEL_COL] + STATIC_COLS
    df = pd.read_csv(path, usecols=usecols, low_memory=False, nrows=nrows)
    # Drop rows with missing activity (they pass a na=False string filter and break pm4py).
    df = df.dropna(subset=[ACT_COL]).copy()
    df[TIME_COL] = pd.to_datetime(df[TIME_COL], errors="coerce")
    df = df.dropna(subset=[TIME_COL])
    # Recruiting Agency: blank == direct sourcing (no agency).
    df["Recruiting Agency"] = df["Recruiting Agency"].fillna("Direct").astype(str)
    # Stable ordering within a case by timestamp; keep original order as tiebreaker.
    df = df.sort_values([CASE_COL, TIME_COL], kind="mergesort").reset_index(drop=True)
    return df


def build_case_table(events: pd.DataFrame) -> pd.DataFrame:
    """One row per case: start, end, label y, and the (case-constant) static attrs."""
    agg = {TIME_COL: ["min", "max"], LABEL_COL: "max"}
    for c in STATIC_COLS:
        agg[c] = "first"
    g = events.groupby(CASE_COL).agg(agg)
    g.columns = ["start", "end", "y"] + STATIC_COLS
    g["y"] = g["y"].astype(int)
    return g.reset_index()


# ----------------------------------------------------------------------------
# 2. Strict temporal split (Weytjens & De Weerdt 2021, sec. 5.5-5.6)
# ----------------------------------------------------------------------------
@dataclass
class SplitResult:
    cases: pd.DataFrame          # case table + columns: split, straddler, cutoff
    sep1: pd.Timestamp
    sep2: pd.Timestamp
    ratios: dict


def temporal_split(cases: pd.DataFrame, q1: float = 0.60, q2: float = 0.80,
                   verbose: bool = True) -> SplitResult:
    """Chronological 60/20/20 split by case START time with STRICT temporal
    splitting.

    NOTE: this diverges from the 80/20 split common in the benchmark literature
    (Teinemaa et al. 2019), so absolute numbers are NOT directly comparable to
    published benchmarks — the point-in-time, leakage-free evaluation is the goal.

    * sep1, sep2 = 60th / 80th percentiles of case start time.
    * train = cases COMPLETED before sep1  (end < sep1)     -> no leakage into val.
    * val   = cases starting in [sep1, sep2) AND end < sep2.
    * test  = cases starting >= sep2 (full trace)  +  straddlers that start before
              sep2 but end at/after sep2, observed only up to sep2 (running cases).
    Cases that start before sep1 but finish after sep1 (train-period straddlers)
    are dropped from train/val to keep the split strict; those overlapping sep2 are
    the test straddlers.
    """
    c = cases.copy()
    sep1 = c["start"].quantile(q1)
    sep2 = c["start"].quantile(q2)

    split = np.full(len(c), "drop", dtype=object)
    straddler = np.zeros(len(c), dtype=bool)
    cutoff = pd.Series(pd.NaT, index=c.index)

    start, end = c["start"], c["end"]
    # train: fully completed before val period
    m_train = end < sep1
    # val: starts in the val window and completes before the test period
    m_val = (start >= sep1) & (start < sep2) & (end < sep2)
    # test regular: starts in the test period (last 20%)
    m_test_reg = start >= sep2
    # test straddlers: running at sep2 (started before, still open at sep2)
    m_test_str = (start < sep2) & (end >= sep2)

    split[m_train.values] = "train"
    split[m_val.values] = "val"
    split[m_test_reg.values] = "test"
    split[m_test_str.values] = "test"
    straddler[m_test_str.values] = True
    cutoff[m_test_str.values] = sep2  # observe only events before sep2

    c["split"] = split
    c["straddler"] = straddler
    c["cutoff"] = cutoff
    c = c[c["split"] != "drop"].reset_index(drop=True)

    ratios = {}
    for s in ["train", "val", "test"]:
        sub = c[c["split"] == s]
        ratios[s] = {"n": len(sub), "pos": int(sub["y"].sum()),
                     "rate": float(sub["y"].mean()) if len(sub) else float("nan")}

    if verbose:
        print(f"Separation times: sep1={sep1}  sep2={sep2}")
        for s in ["train", "val", "test"]:
            r = ratios[s]
            print(f"  {s:5s}: n={r['n']:>7d}  hired={r['pos']:>5d}  hire_rate={r['rate']:.4%}")
        n_str = int(c["straddler"].sum())
        print(f"  test straddlers (running at sep2, observed as prefix): {n_str}")
        # drift warning
        base = ratios["train"]["rate"]
        for s in ["val", "test"]:
            r = ratios[s]["rate"]
            if base > 0 and abs(r - base) / base > 0.50:
                print(f"  [WARN] severe hire-rate drift in {s}: {r:.4%} vs train {base:.4%} "
                      f"({(r-base)/base:+.0%} relative)")
    return SplitResult(cases=c, sep1=sep1, sep2=sep2, ratios=ratios)


# ----------------------------------------------------------------------------
# 3. Gap-based prefix generation (Teinemaa gap filtering)
# ----------------------------------------------------------------------------
@dataclass
class PrefixSplit:
    seqs: list          # list of np.ndarray[int16] activity codes (the prefix)
    static: np.ndarray  # (n, len(STATIC_COLS)) object array of raw static values
    y: np.ndarray       # (n,) int
    prefix_len: np.ndarray
    case_id: np.ndarray


def _cut_lengths(L: int, gap: int, max_prefix: int) -> list[int]:
    """Prefix lengths for a trace of length L: 1, 1+gap, 1+2*gap, ... capped at
    min(L, max_prefix); the final observed length is always included."""
    L = int(min(L, max_prefix))
    ks = list(range(1, L + 1, gap))
    if not ks or ks[-1] != L:
        ks.append(L)
    return ks


def build_prefix_dataset(events: pd.DataFrame, split: SplitResult,
                         gap: int = 3, max_prefix: int = 20,
                         verbose: bool = True) -> dict[str, PrefixSplit]:
    """Generate prefixes per case according to its split.

    * train / val / regular-test cases -> gap-based prefixes over the full trace.
    * test straddlers -> a SINGLE prefix = events observed before the cutoff (sep2),
      i.e. one prediction per running case (the point-in-time snapshot).
    """
    # compact activity codebook (storage only; the encoders derive their vocabulary
    # from the TRAIN split, so this global codebook is not an information leak).
    acts = pd.Index(sorted(events[ACT_COL].unique()))
    act2code = {a: i for i, a in enumerate(acts)}

    case_meta = split.cases.set_index(CASE_COL)
    keep_cases = set(case_meta.index)

    buckets: dict[str, dict] = {s: {"seqs": [], "static": [], "y": [], "plen": [], "cid": []}
                                for s in ["train", "val", "test"]}

    ev = events[events[CASE_COL].isin(keep_cases)]
    static_lookup = case_meta[STATIC_COLS]

    t0 = time.time()
    for cid, grp in ev.groupby(CASE_COL, sort=False):
        meta = case_meta.loc[cid]
        s = meta["split"]
        y = int(meta["y"])
        static_vals = static_lookup.loc[cid].to_numpy()

        times = grp[TIME_COL].to_numpy()
        codes = grp[ACT_COL].map(act2code).to_numpy(dtype=np.int16)

        if meta["straddler"]:
            cutoff = meta["cutoff"]
            obs = codes[times < np.datetime64(cutoff)]
            if len(obs) == 0:
                continue
            obs = obs[:max_prefix]
            ks = [len(obs)]
            seq_full = obs
        else:
            seq_full = codes
            ks = _cut_lengths(len(codes), gap, max_prefix)

        for k in ks:
            buckets[s]["seqs"].append(seq_full[:k].copy())
            buckets[s]["static"].append(static_vals)
            buckets[s]["y"].append(y)
            buckets[s]["plen"].append(k)
            buckets[s]["cid"].append(cid)

    out = {}
    for s, b in buckets.items():
        out[s] = PrefixSplit(
            seqs=b["seqs"],
            static=np.array(b["static"], dtype=object) if b["static"] else np.empty((0, len(STATIC_COLS)), object),
            y=np.array(b["y"], dtype=int),
            prefix_len=np.array(b["plen"], dtype=int),
            case_id=np.array(b["cid"], dtype=object),
        )
    if verbose:
        print(f"Prefix generation ({time.time()-t0:.1f}s), gap={gap}, max_prefix={max_prefix}:")
        for s in ["train", "val", "test"]:
            ps = out[s]
            rate = ps.y.mean() if len(ps.y) else float("nan")
            print(f"  {s:5s}: {len(ps.y):>8d} prefixes  hire_rate={rate:.4%}")
    out["_act2code"] = act2code
    out["_code2act"] = {v: k for k, v in act2code.items()}
    return out


# ----------------------------------------------------------------------------
# 4. Encodings (vocabularies fit on TRAIN prefixes only)
# ----------------------------------------------------------------------------
UNK = "<UNK>"


@dataclass
class FeatureBundle:
    X: dict            # {'train':X, 'val':X, 'test':X} float32 arrays
    feature_names: list
    categorical_cols: list   # names of nominal (label-encoded) columns for SMOTE-NC
    cat_indices: list        # their column indices


def _fit_activity_vocab(train: PrefixSplit) -> list[int]:
    seen = set()
    for s in train.seqs:
        seen.update(int(x) for x in s)
    return sorted(seen)


def _encode_controlflow(split: PrefixSplit, kind: str, vocab, code2act) -> tuple[np.ndarray, list]:
    if kind in ("boolean", "frequency"):
        col_of = {c: i for i, c in enumerate(vocab)}
        X = np.zeros((len(split.seqs), len(vocab)), dtype=np.float32)
        for r, seq in enumerate(split.seqs):
            if kind == "boolean":
                for c in set(int(x) for x in seq):
                    j = col_of.get(c)
                    if j is not None:
                        X[r, j] = 1.0
            else:  # frequency (Teinemaa aggregation)
                for c, n in Counter(int(x) for x in seq).items():
                    j = col_of.get(c)
                    if j is not None:
                        X[r, j] = n
        names = [f"act::{code2act[c]}" for c in vocab]
        return X, names
    raise ValueError(kind)


def _fit_bigram_vocab(train: PrefixSplit, topk: int) -> list[tuple]:
    cnt = Counter()
    for seq in train.seqs:
        for a, b in zip(seq[:-1], seq[1:]):
            cnt[(int(a), int(b))] += 1
    return [pair for pair, _ in cnt.most_common(topk)]


def _encode_bigram(split: PrefixSplit, vocab, code2act) -> tuple[np.ndarray, list]:
    col_of = {p: i for i, p in enumerate(vocab)}
    X = np.zeros((len(split.seqs), len(vocab)), dtype=np.float32)
    for r, seq in enumerate(split.seqs):
        for a, b in zip(seq[:-1], seq[1:]):
            j = col_of.get((int(a), int(b)))
            if j is not None:
                X[r, j] += 1.0
    names = [f"bg::{code2act[a]}>{code2act[b]}" for (a, b) in vocab]
    return X, names


def _fit_static_maps(train: PrefixSplit) -> list[dict]:
    maps = []
    for j in range(train.static.shape[1]):
        vals = pd.unique(train.static[:, j])
        m = {v: i for i, v in enumerate(sorted(vals, key=lambda x: str(x)))}
        m[UNK] = len(m)
        maps.append(m)
    return maps


def _encode_static(split: PrefixSplit, maps: list[dict]) -> tuple[np.ndarray, list]:
    n = split.static.shape[0]
    X = np.zeros((n, len(STATIC_COLS)), dtype=np.float32)
    for j, m in enumerate(maps):
        unk = m[UNK]
        X[:, j] = [m.get(v, unk) for v in split.static[:, j]]
    names = [f"{c}_enc" for c in STATIC_COLS]
    return X, names


def build_features(data: dict, encoding: str, bigram_topk: int = 200) -> FeatureBundle:
    """Assemble the feature matrices for the three splits for one control-flow
    encoding, concatenating the label-encoded static case attributes.

    The label-encoded static attributes are NOMINAL codes (agency, country, job
    family, ...). They are the columns that MUST be flagged categorical for
    SMOTE-NC — interpolating them as if numeric (e.g. agency = 3.5) is the bug this
    pipeline guards against. Activity aggregation features (indicators / counts /
    bigram counts) are genuinely numeric and stay continuous.
    """
    train, val, test = data["train"], data["val"], data["test"]
    code2act = data["_code2act"]

    if encoding in ("boolean", "frequency"):
        vocab = _fit_activity_vocab(train)
        cf = {s: _encode_controlflow(data[s], encoding, vocab, code2act) for s in ("train", "val", "test")}
    elif encoding == "bigram":
        vocab = _fit_bigram_vocab(train, bigram_topk)
        cf = {s: _encode_bigram(data[s], vocab, code2act) for s in ("train", "val", "test")}
    else:
        raise ValueError(encoding)

    static_maps = _fit_static_maps(train)
    st = {s: _encode_static(data[s], static_maps) for s in ("train", "val", "test")}

    cf_names = cf["train"][1]
    st_names = st["train"][1]
    feature_names = cf_names + st_names
    categorical_cols = list(st_names)          # nominal statics only
    cat_indices = [feature_names.index(c) for c in categorical_cols]

    X = {s: np.hstack([cf[s][0], st[s][0]]).astype(np.float32) for s in ("train", "val", "test")}
    return FeatureBundle(X=X, feature_names=feature_names,
                         categorical_cols=categorical_cols, cat_indices=cat_indices)


# ----------------------------------------------------------------------------
# 5. Imbalance handling
# ----------------------------------------------------------------------------
def verify_smotenc_mask(bundle: FeatureBundle) -> list[int]:
    """Programmatically verify the SMOTE-NC categorical mask covers every
    label-encoded static column, in particular the Recruiting Agency column that a
    previous version of this code omitted. Returns the validated index list.
    """
    names = bundle.feature_names
    idx = list(bundle.cat_indices)
    # every declared categorical column present
    for c in bundle.categorical_cols:
        assert names.index(c) in idx, f"categorical column {c!r} missing from SMOTE-NC mask"
    # explicit agency check
    agency = [i for i, n in enumerate(names) if "recruiting agency" in n.lower()]
    assert agency, "no Recruiting Agency column found in features"
    assert all(a in idx for a in agency), "Recruiting Agency column NOT in SMOTE-NC categorical mask"
    print(f"  [SMOTE-NC] categorical mask verified: {len(idx)} cols "
          f"{[names[i] for i in idx]} (agency @ {agency})")
    return idx


def apply_smotenc(Xtr: np.ndarray, ytr: np.ndarray, cat_indices: list[int],
                  seed: int = SEED) -> tuple[np.ndarray, np.ndarray]:
    from imblearn.over_sampling import SMOTENC
    sm = SMOTENC(categorical_features=cat_indices, random_state=seed, k_neighbors=5)
    Xr, yr = sm.fit_resample(Xtr, ytr)
    # imbalanced-learn may return a DataFrame; keep everything as plain arrays so the
    # downstream estimators are never fit with feature names then predicted without.
    return np.asarray(Xr, dtype=np.float32), np.asarray(yr)


def pos_weight_from(ytr: np.ndarray) -> float:
    """scale_pos_weight = n_negative / n_positive on the training set."""
    n_pos = int((ytr == 1).sum())
    n_neg = int((ytr == 0).sum())
    return n_neg / max(n_pos, 1)


# ----------------------------------------------------------------------------
# 6. Models + validation-only tuning
# ----------------------------------------------------------------------------
def _param_space(model: str, fast: bool) -> dict:
    if model == "rf":
        return {
            "n_estimators": [150] if fast else [200, 300, 400],
            "max_depth": [None, 10, 20, 30],
            "max_features": ["sqrt", 0.3, 0.5],
            "min_samples_leaf": [1, 5, 20],
        }
    if model == "xgb":
        return {
            "n_estimators": [200] if fast else [200, 400, 600],
            "max_depth": [3, 6, 9],
            "learning_rate": [0.03, 0.1, 0.3],
            "subsample": [0.7, 1.0],
            "colsample_bytree": [0.7, 1.0],
            "min_child_weight": [1, 5],
        }
    if model == "lgbm":
        return {
            "n_estimators": [300] if fast else [300, 600],
            "num_leaves": [31, 63, 127],
            "learning_rate": [0.03, 0.1],
            "subsample": [0.7, 1.0],
            "colsample_bytree": [0.7, 1.0],
            "min_child_samples": [20, 50],
        }
    raise ValueError(model)


def make_model(model: str, strategy: str, pos_weight: float, params: dict):
    """Instantiate a classifier wired for the chosen imbalance strategy.

    For 'classweight': RF uses class_weight='balanced'; XGB/LGBM use
    scale_pos_weight = n_neg/n_pos. The value actually handed to the estimator is
    asserted below (guards against computing pos_weight but never passing it).
    """
    use_weight = strategy == "classweight"
    if model == "rf":
        from sklearn.ensemble import RandomForestClassifier
        clf = RandomForestClassifier(
            random_state=SEED, n_jobs=-1,
            class_weight=("balanced" if use_weight else None), **params)
        return clf
    if model == "xgb":
        from xgboost import XGBClassifier
        spw = pos_weight if use_weight else 1.0
        clf = XGBClassifier(
            random_state=SEED, n_jobs=-1, tree_method="hist",
            eval_metric="logloss", scale_pos_weight=spw, **params)
        if use_weight:
            assert abs(clf.get_params()["scale_pos_weight"] - pos_weight) < 1e-9, \
                "XGB scale_pos_weight not applied"
        return clf
    if model == "lgbm":
        from lightgbm import LGBMClassifier
        spw = pos_weight if use_weight else 1.0
        clf = LGBMClassifier(
            random_state=SEED, n_jobs=-1, verbose=-1, scale_pos_weight=spw, **params)
        if use_weight:
            assert abs(clf.get_params()["scale_pos_weight"] - pos_weight) < 1e-9, \
                "LGBM scale_pos_weight not applied"
        return clf
    raise ValueError(model)


def _sample_params(space: dict, rng: np.random.Generator) -> dict:
    return {k: v[int(rng.integers(len(v)))] for k, v in space.items()}


def tune_and_fit(model: str, strategy: str, bundle: FeatureBundle,
                 ytr: np.ndarray, Xtr: np.ndarray, Xval: np.ndarray, yval: np.ndarray,
                 pos_weight: float, n_iter: int = 10, fast: bool = False,
                 search_subsample: int = 60000, verbose: bool = True):
    """Random search selected on the VALIDATION set only; refit best on full train.

    Xtr/ytr are the (already resampled, if SMOTE-NC) training features.
    """
    from sklearn.metrics import roc_auc_score
    space = _param_space(model, fast)
    rng = np.random.default_rng(SEED)

    # subsample train for the search phase only (speed); refit best on full train.
    if search_subsample and len(ytr) > search_subsample:
        idx = rng.choice(len(ytr), size=search_subsample, replace=False)
        Xs, ys = Xtr[idx], ytr[idx]
    else:
        Xs, ys = Xtr, ytr

    best = {"auc": -1, "params": None}
    seen = set()
    for _ in range(n_iter):
        p = _sample_params(space, rng)
        key = tuple(sorted((k, str(v)) for k, v in p.items()))
        if key in seen:
            continue
        seen.add(key)
        clf = make_model(model, strategy, pos_weight, p)
        clf.fit(Xs, ys)
        auc = roc_auc_score(yval, clf.predict_proba(Xval)[:, 1])
        if auc > best["auc"]:
            best = {"auc": auc, "params": p}
    # refit best on full training set
    clf = make_model(model, strategy, pos_weight, best["params"])
    clf.fit(Xtr, ytr)
    if verbose:
        spw = clf.get_params().get("scale_pos_weight", "n/a")
        print(f"    {model}/{strategy}: best val AUC={best['auc']:.4f} "
              f"params={best['params']} scale_pos_weight={spw}")
    return clf, best


# ----------------------------------------------------------------------------
# 7. Evaluation
# ----------------------------------------------------------------------------
def evaluate(clf, Xte: np.ndarray, yte: np.ndarray) -> dict:
    from sklearn.metrics import (roc_auc_score, average_precision_score, f1_score,
                                 balanced_accuracy_score, precision_score,
                                 recall_score, confusion_matrix)
    proba = clf.predict_proba(Xte)[:, 1]
    pred = (proba >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(yte, pred, labels=[0, 1]).ravel()
    return {
        "auc_roc": roc_auc_score(yte, proba),
        "auc_pr": average_precision_score(yte, proba),
        "f1": f1_score(yte, pred, zero_division=0),
        "balanced_acc": balanced_accuracy_score(yte, pred),
        "precision": precision_score(yte, pred, zero_division=0),
        "recall": recall_score(yte, pred, zero_division=0),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "_proba": proba,
    }


def run_grid(data: dict, n_iter: int = 10, fast: bool = False,
             bigram_topk: int = 200, search_subsample: int = 60000,
             encodings=ENCODINGS, strategies=STRATEGIES, models=MODELS,
             verbose: bool = True) -> tuple[pd.DataFrame, dict]:
    """Full grid: |encodings| x |strategies| x |models|.

    Returns a tidy results DataFrame (one row per config) and an artifacts dict
    holding the test labels, prefix lengths and per-config predicted probabilities
    (used by the plotting helpers). The test set is only touched here, at final
    evaluation.
    """
    yval = data["val"].y
    yte = data["test"].y
    plen_te = data["test"].prefix_len

    rows = []
    artifacts = {"y_test": yte, "prefix_len_test": plen_te, "proba": {}, "feature_importance": {}}

    for enc in encodings:
        if verbose:
            print(f"\n=== Encoding: {enc} ===")
        bundle = build_features(data, enc, bigram_topk=bigram_topk)
        Xtr0, ytr0 = bundle.X["train"], data["train"].y
        Xval, Xte = bundle.X["val"], bundle.X["test"]
        artifacts.setdefault("feature_names", {})[enc] = bundle.feature_names

        for strat in strategies:
            # build the training matrix for this strategy
            if strat == "smotenc":
                cat_idx = verify_smotenc_mask(bundle)
                t0 = time.time()
                Xtr, ytr = apply_smotenc(Xtr0, ytr0, cat_idx)
                if verbose:
                    print(f"  [SMOTE-NC] {len(ytr0)} -> {len(ytr)} rows "
                          f"(hire_rate {ytr0.mean():.3%} -> {ytr.mean():.3%}), {time.time()-t0:.1f}s")
            else:
                Xtr, ytr = Xtr0, ytr0
            pw = pos_weight_from(ytr0)  # computed on ORIGINAL train distribution
            if strat == "classweight" and verbose:
                print(f"  [class weights] scale_pos_weight = n_neg/n_pos = {pw:.3f}")

            for model in models:
                clf, best = tune_and_fit(model, strat, bundle, ytr, Xtr, Xval, yval, pw,
                                         n_iter=n_iter, fast=fast,
                                         search_subsample=search_subsample, verbose=verbose)
                met = evaluate(clf, Xte, yte)
                cfg = f"{enc}|{strat}|{model}"
                artifacts["proba"][cfg] = met.pop("_proba")
                # gain-based importance
                artifacts["feature_importance"][cfg] = _gain_importance(clf, model, bundle.feature_names)
                rows.append({
                    "encoding": enc, "strategy": strat, "model": model,
                    "val_auc": best["auc"],
                    "scale_pos_weight": (pw if strat == "classweight" and model in ("xgb", "lgbm") else np.nan),
                    **{k: v for k, v in met.items()},
                })

    res = pd.DataFrame(rows).sort_values("auc_roc", ascending=False).reset_index(drop=True)
    return res, artifacts


def _gain_importance(clf, model: str, feature_names: list) -> pd.Series:
    try:
        if model == "xgb":
            booster = clf.get_booster()
            score = booster.get_score(importance_type="gain")
            imp = np.zeros(len(feature_names))
            for k, v in score.items():
                imp[int(k[1:])] = v  # 'f123' -> 123
        elif model == "lgbm":
            imp = clf.booster_.feature_importance(importance_type="gain")
        else:  # rf: impurity-based
            imp = clf.feature_importances_
        return pd.Series(imp, index=feature_names).sort_values(ascending=False)
    except Exception as e:  # pragma: no cover
        warnings.warn(f"importance failed for {model}: {e}")
        return pd.Series(dtype=float)


# ----------------------------------------------------------------------------
# 8. Plots (Spanish labels) & 9. Summary
# ----------------------------------------------------------------------------
def _savefig(fig, stem: str, figdir: str):
    os.makedirs(figdir, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(figdir, f"{stem}.{ext}"), dpi=300, bbox_inches="tight")


def plot_roc(res: pd.DataFrame, artifacts: dict, figdir: str = "figs"):
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve
    y = artifacts["y_test"]
    fig, ax = plt.subplots(figsize=(6, 6))
    for model in sorted(res["model"].unique()):
        sub = res[res["model"] == model].sort_values("auc_roc", ascending=False).iloc[0]
        cfg = f"{sub['encoding']}|{sub['strategy']}|{model}"
        fpr, tpr, _ = roc_curve(y, artifacts["proba"][cfg])
        ax.plot(fpr, tpr, label=f"{model.upper()} ({sub['encoding']}/{sub['strategy']}) AUC={sub['auc_roc']:.3f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Azar")
    ax.set_xlabel("Tasa de falsos positivos")
    ax.set_ylabel("Tasa de verdaderos positivos")
    ax.set_title("Curva ROC — mejor configuración por modelo")
    ax.legend(loc="lower right", fontsize=9)
    _savefig(fig, "outcome_prefix_roc", figdir)
    return fig


def plot_pr(res: pd.DataFrame, artifacts: dict, figdir: str = "figs"):
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_curve
    y = artifacts["y_test"]
    fig, ax = plt.subplots(figsize=(6, 6))
    for model in sorted(res["model"].unique()):
        sub = res[res["model"] == model].sort_values("auc_roc", ascending=False).iloc[0]
        cfg = f"{sub['encoding']}|{sub['strategy']}|{model}"
        prec, rec, _ = precision_recall_curve(y, artifacts["proba"][cfg])
        ax.plot(rec, prec, label=f"{model.upper()} ({sub['encoding']}/{sub['strategy']}) AP={sub['auc_pr']:.3f}")
    base = y.mean()
    ax.axhline(base, ls="--", color="k", lw=1, label=f"Base ({base:.3f})")
    ax.set_xlabel("Exhaustividad (Recall)")
    ax.set_ylabel("Precisión")
    ax.set_title("Curva Precisión-Exhaustividad — mejor configuración por modelo")
    ax.legend(loc="upper right", fontsize=9)
    _savefig(fig, "outcome_prefix_pr", figdir)
    return fig


def plot_feature_importance(res: pd.DataFrame, artifacts: dict, top: int = 20, figdir: str = "figs"):
    import matplotlib.pyplot as plt
    best = res.iloc[0]
    cfg = f"{best['encoding']}|{best['strategy']}|{best['model']}"
    imp = artifacts["feature_importance"][cfg].head(top)[::-1]
    fig, ax = plt.subplots(figsize=(7, 8))
    ax.barh(range(len(imp)), imp.values)
    ax.set_yticks(range(len(imp)))
    ax.set_yticklabels(imp.index, fontsize=8)
    ax.set_xlabel("Importancia (ganancia)")
    ax.set_title(f"Importancia de variables — mejor modelo\n({cfg}, AUC={best['auc_roc']:.3f})")
    _savefig(fig, "outcome_prefix_importance", figdir)
    return fig


def plot_auc_vs_prefix(res: pd.DataFrame, artifacts: dict, figdir: str = "figs"):
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_auc_score
    best = res.iloc[0]
    cfg = f"{best['encoding']}|{best['strategy']}|{best['model']}"
    y = artifacts["y_test"]
    plen = artifacts["prefix_len_test"]
    proba = artifacts["proba"][cfg]
    rows = []
    for k in sorted(np.unique(plen)):
        m = plen == k
        if m.sum() >= 30 and len(np.unique(y[m])) == 2:
            rows.append((k, roc_auc_score(y[m], proba[m]), int(m.sum())))
    if not rows:
        return None
    ks, aucs, ns = zip(*rows)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(ks, aucs, "o-")
    ax.set_xlabel("Longitud de prefijo (nº de eventos observados)")
    ax.set_ylabel("AUC-ROC")
    ax.set_title(f"AUC-ROC por longitud de prefijo — mejor configuración\n({cfg})")
    ax.grid(alpha=0.3)
    _savefig(fig, "outcome_prefix_auc_vs_prefix", figdir)
    return fig


def _df_to_md(df: pd.DataFrame, index: bool = False) -> str:
    """Render a DataFrame as a GitHub-flavoured markdown table (no `tabulate` dep)."""
    d = df.copy()
    if index:
        d = d.reset_index()

    def fmt(v):
        if isinstance(v, float):
            return f"{v:.4f}"
        return "" if pd.isna(v) else str(v)

    cols = list(d.columns)
    head = "| " + " | ".join(map(str, cols)) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    rows = ["| " + " | ".join(fmt(v) for v in row) + " |" for row in d.itertuples(index=False, name=None)]
    return "\n".join([head, sep] + rows)


def write_summary(res: pd.DataFrame, path: str = "OUTCOME_PREFIX_SUMMARY.md"):
    best = res.iloc[0]
    # SMOTE-NC vs class weights: compare mean AUC across configs
    by_strat = res.groupby("strategy")["auc_roc"].agg(["mean", "max"]).sort_values("mean", ascending=False)
    best_smote = res[res["strategy"] == "smotenc"]["auc_roc"].max()
    best_cw = res[res["strategy"] == "classweight"]["auc_roc"].max()
    best_none = res[res["strategy"] == "none"]["auc_roc"].max()
    preferable = "SMOTE-NC" if best_smote >= best_cw else "ponderación de clases (class weights)"

    lines = [
        "# Resumen — Predicción de resultado (contratado / no contratado)\n",
        "Monitorización predictiva orientada a resultados, basada en prefijos "
        "(Teinemaa et al. 2019) con partición temporal estricta "
        "(Weytjens & De Weerdt 2021) y tratamiento del desbalance (Ceravolo et al. 2024).\n",
        "## Mejor configuración global\n",
        f"- **Codificación:** {best['encoding']}",
        f"- **Estrategia de desbalance:** {best['strategy']}",
        f"- **Modelo:** {best['model'].upper()}",
        f"- **AUC-ROC (test):** {best['auc_roc']:.4f}",
        f"- **AUC-PR:** {best['auc_pr']:.4f} · **F1:** {best['f1']:.4f} · "
        f"**Balanced acc.:** {best['balanced_acc']:.4f} · "
        f"**Precisión:** {best['precision']:.4f} · **Recall:** {best['recall']:.4f}\n",
        "## ¿SMOTE-NC o ponderación de clases?\n",
        f"- Mejor AUC-ROC sin tratamiento: {best_none:.4f}",
        f"- Mejor AUC-ROC con SMOTE-NC: {best_smote:.4f}",
        f"- Mejor AUC-ROC con class weights: {best_cw:.4f}",
        f"- **Preferible para este log:** {preferable}\n",
        "### AUC-ROC medio por estrategia\n",
        _df_to_md(by_strat, index=True),
        "\n## Tabla completa (ordenada por AUC-ROC)\n",
        _df_to_md(res.drop(columns=[c for c in res.columns if c.startswith("_")]), index=False),
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Wrote {path}")
    return path

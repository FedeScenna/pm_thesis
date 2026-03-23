"""
process_map.py — DFG generation via pm4py + Graphviz.
Returns PNG bytes for st.image().

Windows temp-file note: use delete=False + manual os.unlink.
dot.exe cannot write to a file that is still held open by the
NamedTemporaryFile handle (Windows file-locking behaviour).
"""

import os
import tempfile
import random

import pandas as pd
import pm4py
import streamlit as st
from pm4py.visualization.dfg import visualizer as dfg_vis


def build_pm4py_subset(
    df: pd.DataFrame,
    case_ids: set,
    max_cases: int = 20_000,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Filter df to case_ids, downsample to max_cases if needed,
    rename columns to pm4py standard names, and localise timestamps to UTC.
    """
    subset = df[df['Case_id'].isin(case_ids)].copy()

    unique_cases = subset['Case_id'].unique()
    if len(unique_cases) > max_cases:
        rng = random.Random(random_state)
        sampled = set(rng.sample(list(unique_cases), max_cases))
        subset = subset[subset['Case_id'].isin(sampled)]

    subset = subset.rename(columns={
        'Case_id':      'case:concept:name',
        'Step':         'concept:name',
        'timestamp':    'time:timestamp',
        'Completed By': 'org:resource',
    })

    # pm4py requires timezone-aware timestamps (UTC)
    subset['time:timestamp'] = pd.to_datetime(subset['time:timestamp'], utc=True)

    return subset[['case:concept:name', 'concept:name', 'time:timestamp', 'org:resource']]


def generate_dfg_png(
    df: pd.DataFrame,
    case_ids: set,
    top_n_edges: int = 50,
    max_cases: int = 20_000,
) -> bytes:
    """
    Build a DFG from the filtered log and render it as PNG bytes.

    Raises:
        RuntimeError: if Graphviz dot binary is not found on PATH.
    """
    subset = build_pm4py_subset(df, case_ids, max_cases)
    log = pm4py.convert_to_event_log(subset)

    dfg, start_activities, end_activities = pm4py.discover_dfg(log)

    # Keep only top_n_edges by frequency for readability
    top_edges = dict(
        sorted(dfg.items(), key=lambda x: x[1], reverse=True)[:top_n_edges]
    )

    gviz = dfg_vis.apply(
        top_edges,
        log=log,
        parameters={
            dfg_vis.Variants.FREQUENCY.value.Parameters.FORMAT: 'png',
        },
    )

    # Windows-safe: close the NamedTemporaryFile before dot.exe tries to write to it
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
        tmp_path = tmp.name
    # tmp handle is now closed; dot.exe can write to tmp_path

    try:
        dfg_vis.save(gviz, tmp_path)
        with open(tmp_path, 'rb') as f:
            return f.read()
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


@st.cache_data(show_spinner="Generating process map…")
def cached_dfg_png(
    _df: pd.DataFrame,
    case_ids_frozenset: frozenset,
    top_n_edges: int = 50,
    max_cases: int = 20_000,
) -> bytes:
    """
    Cached wrapper. Uses frozenset as the hashable cache key.
    _df uses underscore prefix to skip Streamlit's DataFrame hashing.
    """
    return generate_dfg_png(_df, set(case_ids_frozenset), top_n_edges, max_cases)

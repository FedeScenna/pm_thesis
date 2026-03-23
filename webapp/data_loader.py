"""
data_loader.py — Cached CSV loading and variant computation.
All heavy I/O happens here; every function is cached by Streamlit.
"""

import pandas as pd
import streamlit as st

# Absolute path to the event log (sibling to this webapp folder)
DATA_PATH = r"C:\Users\feder\OneDrive\Documents\pm_thesis\data\event_log_consolidated.csv"

_DTYPE_DICT = {
    'hired': 'Int8', 'Rejected': 'Int8', 'CW': 'Int8', 'Evergreen': 'Int8',
    'Region': 'Int16', 'Country': 'Int16',
    'Job Family': 'Int16', 'Job Family Group': 'Int16',
}


@st.cache_data(show_spinner="Loading event log (200 MB) — first load only…")
def load_event_log() -> pd.DataFrame:
    """
    Read, clean, and return the full event log DataFrame.
    Filters out numeric-only Step codes (~15 codes such as '407', '6748').
    Result is cached for the server's lifetime.
    """
    df = pd.read_csv(
        DATA_PATH,
        low_memory=False,
        dtype=_DTYPE_DICT,
        parse_dates=['timestamp'],
    )
    df = df[~df['Step'].str.match(r'^\d+$', na=False)].copy()
    df.sort_values(['Case_id', 'timestamp'], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


@st.cache_data(show_spinner="Computing activity sequences per case…")
def get_case_sequences(_df: pd.DataFrame) -> pd.Series:
    """
    Return a Series: index = Case_id, value = tuple of activity sequence.
    Cached separately so variant filtering is O(set lookup) on re-runs,
    not O(5.8M rows) each time.

    Leading underscore on _df skips Streamlit's expensive DataFrame hashing.
    """
    return _df.groupby('Case_id')['Step'].apply(tuple)


@st.cache_data(show_spinner="Computing variant frequencies…")
def compute_variants(_df: pd.DataFrame) -> pd.Series:
    """
    Return a Series: index = tuple (variant), value = case count, sorted desc.
    """
    seqs = get_case_sequences(_df)
    return seqs.value_counts()


def filter_by_variants(
    case_sequences: pd.Series,
    selected_variants: list,
) -> set:
    """
    Return the set of Case_ids whose activity sequence is in selected_variants.
    Not cached — cheap after case_sequences is already cached.
    """
    selected_set = set(map(tuple, selected_variants))
    mask = case_sequences.isin(selected_set)
    return set(case_sequences[mask].index)

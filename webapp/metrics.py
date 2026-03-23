"""
metrics.py — Pure pandas descriptive metric computations.
All functions accept an already-filtered DataFrame and return
plain dicts or DataFrames ready for Plotly Express.
"""

import pandas as pd
import numpy as np


def compute_kpis(df: pd.DataFrame) -> dict:
    """KPI summary for the filtered subset."""
    case_attrs = df.drop_duplicates('Case_id')
    hire_rate = case_attrs['hired'].astype(float).mean()
    return {
        'n_cases':      df['Case_id'].nunique(),
        'n_events':     len(df),
        'n_activities': df['Step'].nunique(),
        'n_resources':  df['Completed By'].nunique(),
        'hire_rate':    hire_rate if not pd.isna(hire_rate) else 0.0,
    }


def compute_activity_freq(df: pd.DataFrame, top_n: int = 30) -> pd.DataFrame:
    """Top-N activities by event count."""
    counts = df['Step'].value_counts().head(top_n).reset_index()
    counts.columns = ['activity', 'count']
    return counts.sort_values('count')  # ascending for horizontal bar


def compute_trace_length_dist(df: pd.DataFrame) -> pd.DataFrame:
    """One row per case: Case_id, trace_length."""
    tl = df.groupby('Case_id').size().reset_index(name='trace_length')
    return tl


def compute_case_duration_dist(df: pd.DataFrame, clip_days: int = 365) -> pd.DataFrame:
    """One row per case: Case_id, duration_days (clipped at clip_days)."""
    times = df.groupby('Case_id')['timestamp'].agg(['min', 'max'])
    times['duration_days'] = (times['max'] - times['min']).dt.total_seconds() / 86400
    times['duration_days'] = times['duration_days'].clip(upper=clip_days)
    return times[['duration_days']].reset_index()


def compute_monthly_volume(df: pd.DataFrame) -> pd.DataFrame:
    """Event count per calendar month, filtered to >= 2022."""
    monthly = (
        df[df['timestamp'] >= '2022-01-01']
        .groupby(df['timestamp'].dt.to_period('M'))
        .size()
        .reset_index(name='event_count')
    )
    monthly['year_month'] = monthly['timestamp'].astype(str)
    return monthly[['year_month', 'event_count']]


def compute_rolling_hire_rate(df: pd.DataFrame, window: int = 3) -> pd.DataFrame:
    """
    Monthly hire rate with rolling smoothing.
    Returns DataFrame[year_month, n_cases, hire_rate, hire_rate_rolling].
    """
    case_attrs = df.drop_duplicates('Case_id')[['Case_id', 'hired', 'timestamp']].copy()
    case_attrs = case_attrs[case_attrs['timestamp'] >= '2022-01-01']
    case_attrs['month'] = case_attrs['timestamp'].dt.to_period('M')

    monthly = (
        case_attrs.groupby('month')
        .agg(n_cases=('Case_id', 'count'),
             hire_rate=('hired', lambda x: x.astype(float).mean()))
        .reset_index()
    )
    monthly['hire_rate_rolling'] = (
        monthly['hire_rate']
        .rolling(window, center=True, min_periods=1)
        .mean()
    )
    monthly['year_month'] = monthly['month'].astype(str)
    return monthly[['year_month', 'n_cases', 'hire_rate', 'hire_rate_rolling']]


def compute_hire_rate_by_dim(
    df: pd.DataFrame,
    dim_col: str,
    top_n: int = 15,
) -> pd.DataFrame:
    """
    Hire rate per unique value of dim_col (e.g. 'Region', 'Job Family Group').
    Returns top_n by hire rate, with n_cases for context.
    Integer-encoded values are kept as-is (no decode map available).
    """
    case_attrs = df.drop_duplicates('Case_id')[[dim_col, 'hired']].copy()
    case_attrs[dim_col] = case_attrs[dim_col].astype(str)
    result = (
        case_attrs.groupby(dim_col)
        .agg(n_cases=('hired', 'count'),
             hire_rate=('hired', lambda x: x.astype(float).mean()))
        .reset_index()
        .dropna(subset=['hire_rate'])
        .sort_values('hire_rate', ascending=False)
        .head(top_n)
    )
    return result

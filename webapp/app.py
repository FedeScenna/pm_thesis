"""
app.py — Streamlit entrypoint for the Recruiting Process Mining Dashboard.

Run:
    cd C:\\Users\\feder\\OneDrive\\Documents\\pm_thesis\\webapp
    streamlit run app.py

Requirements (in processmining conda env):
    pip install streamlit plotly
    conda install -c conda-forge graphviz   # for process map
"""

import streamlit as st
import plotly.express as px
import numpy as np

from data_loader import (
    load_event_log,
    get_case_sequences,
    compute_variants,
    filter_by_variants,
)
from metrics import (
    compute_kpis,
    compute_activity_freq,
    compute_trace_length_dist,
    compute_case_duration_dist,
    compute_monthly_volume,
    compute_rolling_hire_rate,
    compute_hire_rate_by_dim,
)
from process_map import cached_dfg_png

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Recruiting Process Mining",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("⚙️ Recruiting Process Mining Dashboard")

# ── Data load (cached) ────────────────────────────────────────────────────────
df = load_event_log()
case_sequences = get_case_sequences(df)
variant_counts = compute_variants(df)

# ── Sidebar: variant filter ───────────────────────────────────────────────────
with st.sidebar:
    st.header("Variant Filter")

    top_n = st.slider(
        "Show top N variants",
        min_value=5, max_value=100, value=20, step=5,
        help="Number of most-frequent variants to display in the filter list.",
    )

    top_variants = variant_counts.head(top_n)

    # Build label→tuple mapping — use string labels as options (tuples are
    # not reliably serialised by Streamlit's multiselect widget).
    def _make_label(v: tuple, count: int) -> str:
        seq = " → ".join(str(a) for a in v)
        seq = seq[:88] + "…" if len(seq) > 88 else seq
        return f"{seq}  ({count:,})"

    label_to_variant: dict[str, tuple] = {
        _make_label(v, cnt): v
        for v, cnt in top_variants.items()
    }
    all_labels = list(label_to_variant.keys())

    selected_labels = st.multiselect(
        "Select variants",
        options=all_labels,
        default=all_labels,
        help="Only cases matching the selected variants will appear in all charts.",
    )
    selected_variants = [label_to_variant[lbl] for lbl in selected_labels]

    # Guard: if user deselects everything, fall back to all top-N
    if not selected_variants:
        selected_variants = list(top_variants.index)
        st.caption("⚠️ No variants selected — showing all top-N.")

    st.markdown("---")
    filtered_case_ids = filter_by_variants(case_sequences, selected_variants)
    st.caption(
        f"**{len(selected_variants)}** variant(s) selected  \n"
        f"**{len(filtered_case_ids):,}** matching cases"
    )

# ── Filtered DataFrame ────────────────────────────────────────────────────────
df_f = df[df['Case_id'].isin(filtered_case_ids)]

# ── KPI row (always visible above tabs) ──────────────────────────────────────
kpis = compute_kpis(df_f)
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Cases",       f"{kpis['n_cases']:,}")
c2.metric("Events",      f"{kpis['n_events']:,}")
c3.metric("Activities",  kpis['n_activities'])
c4.metric("Resources",   kpis['n_resources'])
c5.metric("Hire Rate",   f"{kpis['hire_rate']:.1%}")

st.markdown("---")

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_overview, tab_dist, tab_time, tab_hire, tab_variants, tab_map = st.tabs([
    "Overview", "Distributions", "Time Analysis", "Hire Rate", "Variants", "Process Map",
])

# ── Tab: Overview ─────────────────────────────────────────────────────────────
with tab_overview:
    st.subheader("Activity Frequency (top 30)")
    act_freq = compute_activity_freq(df_f, top_n=30)
    fig = px.bar(
        act_freq, x='count', y='activity', orientation='h',
        labels={'count': 'Event count', 'activity': 'Activity'},
        height=700,
        color='count',
        color_continuous_scale='Blues',
    )
    fig.update_layout(coloraxis_showscale=False, yaxis={'categoryorder': 'total ascending'})
    st.plotly_chart(fig, use_container_width=True)

# ── Tab: Distributions ────────────────────────────────────────────────────────
with tab_dist:
    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Trace Length Distribution")
        tl = compute_trace_length_dist(df_f)
        p95 = int(np.percentile(tl['trace_length'], 95))
        fig_tl = px.histogram(
            tl, x='trace_length',
            nbins=60,
            labels={'trace_length': 'Events per case', 'count': 'Cases'},
            title=f"p50={int(tl['trace_length'].median())}  p95={p95}",
        )
        fig_tl.update_traces(marker_color='steelblue')
        st.plotly_chart(fig_tl, use_container_width=True)

    with col_b:
        st.subheader("Case Duration Distribution (clipped at 365 days)")
        dur = compute_case_duration_dist(df_f, clip_days=365)
        p50_dur = dur['duration_days'].median()
        fig_dur = px.histogram(
            dur, x='duration_days',
            nbins=60,
            labels={'duration_days': 'Duration (days)', 'count': 'Cases'},
            title=f"Median: {p50_dur:.1f} days",
        )
        fig_dur.update_traces(marker_color='darkorange')
        st.plotly_chart(fig_dur, use_container_width=True)

# ── Tab: Time Analysis ────────────────────────────────────────────────────────
with tab_time:
    st.subheader("Monthly Event Volume (from 2022)")
    vol = compute_monthly_volume(df_f)
    fig_vol = px.line(
        vol, x='year_month', y='event_count', markers=True,
        labels={'year_month': 'Month', 'event_count': 'Events'},
    )
    fig_vol.update_xaxes(tickangle=45)
    st.plotly_chart(fig_vol, use_container_width=True)

    st.subheader("Rolling 3-Month Hire Rate (from 2022)")
    hr = compute_rolling_hire_rate(df_f, window=3)
    fig_hr = px.line(
        hr, x='year_month', y=['hire_rate', 'hire_rate_rolling'],
        labels={
            'year_month': 'Month',
            'value': 'Hire Rate',
            'variable': 'Series',
        },
    )
    fig_hr.update_xaxes(tickangle=45)
    newnames = {'hire_rate': 'Monthly', 'hire_rate_rolling': '3-month rolling avg'}
    fig_hr.for_each_trace(lambda t: t.update(name=newnames.get(t.name, t.name)))
    st.plotly_chart(fig_hr, use_container_width=True)

# ── Tab: Hire Rate ────────────────────────────────────────────────────────────
with tab_hire:
    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Hire Rate by Region (integer-encoded)")
        hr_region = compute_hire_rate_by_dim(df_f, 'Region', top_n=15)
        fig_r = px.bar(
            hr_region, x='hire_rate', y='Region', orientation='h',
            hover_data=['n_cases'],
            labels={'hire_rate': 'Hire Rate', 'Region': 'Region (code)'},
            color='hire_rate',
            color_continuous_scale='Greens',
        )
        fig_r.update_layout(coloraxis_showscale=False,
                            yaxis={'categoryorder': 'total ascending'})
        fig_r.update_xaxes(tickformat='.1%')
        st.plotly_chart(fig_r, use_container_width=True)

    with col_b:
        st.subheader("Hire Rate by Job Family Group (integer-encoded)")
        hr_jfg = compute_hire_rate_by_dim(df_f, 'Job Family Group', top_n=15)
        fig_j = px.bar(
            hr_jfg, x='hire_rate', y='Job Family Group', orientation='h',
            hover_data=['n_cases'],
            labels={'hire_rate': 'Hire Rate', 'Job Family Group': 'Job Family Group (code)'},
            color='hire_rate',
            color_continuous_scale='Purples',
        )
        fig_j.update_layout(coloraxis_showscale=False,
                            yaxis={'categoryorder': 'total ascending'})
        fig_j.update_xaxes(tickformat='.1%')
        st.plotly_chart(fig_j, use_container_width=True)

# ── Tab: Variants ─────────────────────────────────────────────────────────────
with tab_variants:
    st.subheader("Variant Frequency (top 30 overall)")
    top30 = variant_counts.head(30).reset_index()
    top30.columns = ['variant_tuple', 'case_count']
    top30['variant_label'] = top30['variant_tuple'].apply(
        lambda v: (" → ".join(v))[:80] + ("…" if len(" → ".join(v)) > 80 else "")
    )
    top30['selected'] = top30['variant_tuple'].apply(lambda v: v in set(selected_variants))
    top30 = top30.sort_values('case_count')

    fig_v = px.bar(
        top30, x='case_count', y='variant_label', orientation='h',
        color='selected',
        color_discrete_map={True: 'steelblue', False: 'lightgrey'},
        labels={'case_count': 'Case count', 'variant_label': 'Variant'},
        height=700,
    )
    fig_v.update_layout(
        showlegend=True,
        legend_title="Currently selected",
        yaxis={'categoryorder': 'total ascending'},
    )
    st.plotly_chart(fig_v, use_container_width=True)

    total_cases = variant_counts.sum()
    top30_cases = variant_counts.head(30).sum()
    st.caption(
        f"Top 30 variants cover {top30_cases:,} / {total_cases:,} cases "
        f"({top30_cases/total_cases:.1%}). "
        f"Total unique variants: {len(variant_counts):,}."
    )

# ── Tab: Process Map ──────────────────────────────────────────────────────────
with tab_map:
    st.subheader("Directly-Follows Graph (DFG)")
    st.caption(
        "Built from up to 20,000 randomly sampled cases matching the current variant filter. "
        "Click **Generate** after changing the variant selection or edge count."
    )

    col_ctrl1, col_ctrl2 = st.columns([2, 1])
    with col_ctrl1:
        top_n_edges = st.slider(
            "Max edges shown", min_value=10, max_value=100, value=50, step=5,
        )
    with col_ctrl2:
        generate = st.button("Generate Process Map", type="primary")

    if generate:
        with st.spinner(f"Discovering DFG on up to 20K cases, top {top_n_edges} edges…"):
            try:
                png_bytes = cached_dfg_png(
                    df,
                    frozenset(filtered_case_ids),
                    top_n_edges=top_n_edges,
                    max_cases=20_000,
                )
                st.image(png_bytes, use_container_width=True)
                st.caption(
                    f"{len(filtered_case_ids):,} cases matched · "
                    f"up to 20K used for DFG · top {top_n_edges} edges displayed."
                )
            except Exception as exc:
                st.error(
                    f"**Process map failed:** {exc}\n\n"
                    "Make sure Graphviz is installed and `dot` is on your PATH.\n\n"
                    "**Fix:** `conda install -c conda-forge graphviz`  \n"
                    "Then restart the Streamlit server."
                )
    else:
        st.info("Press **Generate Process Map** to render the DFG for the current filter.")

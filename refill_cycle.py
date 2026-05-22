"""Refill Cycle — chronic price-↑ patients overdue for a refill.

For patients who bought a chronic price-↑ SKU at a pilot store in the
post-window, look at their 6-month history (across the whole chain) to
estimate per-drug refill cycles, project expected refill dates, and surface
who is overdue.
"""

from __future__ import annotations

from datetime import date, timedelta

import altair as alt
import pandas as pd
import streamlit as st

from data import (
    COHORT_HISTORY_START,
    PILOT_STORES,
    QUANT_POST_END,
    QUANT_POST_START,
    REFILL_GRACE_DAYS,
    _load_cohort_history_snapshot,
    chronic_pi_drug_ids,
    compute_refill_cycle,
)

# ---------- palette (shared with Bill Progression) --------------------------

COLOR_REFILLED = "#27AE60"
COLOR_OVERDUE = "#E74C3C"
COLOR_ONTIME = "#F39C12"
COLOR_NOEST = "#95A5A6"
STATUS_ORDER = ["refilled", "on_time", "overdue", "no_estimate"]
STATUS_COLOR = {
    "refilled": COLOR_REFILLED,
    "on_time": COLOR_ONTIME,
    "overdue": COLOR_OVERDUE,
    "no_estimate": COLOR_NOEST,
}
STATUS_LABEL = {
    "refilled": "✅ Refilled",
    "on_time": "⏱ On-time (not yet due)",
    "overdue": "⚠️ Overdue",
    "no_estimate": "❔ Insufficient history",
}


# ============================================================================
# Page setup
# ============================================================================

st.title("🔁 Chronic Refill Cycle")
st.caption(
    "Are chronic price-↑ patients refilling on time? **Cohort** = patients who "
    "bought a chronic price-↑ SKU at the selected pilot store(s) during the date "
    "range. **Refill history** is checked across **all chain stores** (so a "
    "patient who refilled at a non-pilot store also counts as refilled)."
)

hist = _load_cohort_history_snapshot()
if hist is None or hist.empty:
    st.error(
        "`data_snapshots/cohort_chronic_history.parquet` is missing or empty. "
        "Run `python refresh_snapshots.py` locally with VPN to regenerate."
    )
    st.stop()

chronic_ids = chronic_pi_drug_ids()
if not chronic_ids:
    st.error("Chronic price-↑ SKU lookup unavailable. Run `python refresh_snapshots.py`.")
    st.stop()


# ----- Sidebar --------------------------------------------------------------
st.sidebar.header("Refill filters")
c1, c2 = st.sidebar.columns(2)
post_from = c1.date_input("Post from", value=QUANT_POST_START, key="rc_post_from")
post_to = c2.date_input("Post to", value=QUANT_POST_END, key="rc_post_to")
st.sidebar.caption(f"Cohort window: {post_from} → {post_to}")

selected_pilots = st.sidebar.multiselect(
    "Pilot store(s)",
    options=list(PILOT_STORES),
    default=list(PILOT_STORES),
    help="Patients are admitted to the cohort only if they bought a chronic price-↑ SKU at one of these stores in the date range.",
)
if not selected_pilots:
    st.warning("Pick at least one pilot store.")
    st.stop()

st.sidebar.markdown("---")
st.sidebar.caption(
    f"📂 6-month history window in snapshot: {COHORT_HISTORY_START} → today"
)


# ----- Build cohort + run analysis -----------------------------------------

@st.cache_data(ttl=900, show_spinner=False)
def build_cohort_and_metrics(
    post_from: date, post_to: date, pilots: tuple[str, ...]
) -> tuple[set[int], pd.DataFrame]:
    h = _load_cohort_history_snapshot()
    h = h[h["bill_flag"].astype(str).str.lower() == "gross"].copy()
    h["day"] = pd.to_datetime(h["created_at"]).dt.date
    cohort_mask = (
        h["store_name"].isin(set(pilots))
        & (h["day"] >= post_from)
        & (h["day"] <= post_to)
        & h["drug_id"].astype("Int64").isin(chronic_ids)
    )
    cohort = set(int(x) for x in h.loc[cohort_mask, "patient_id"].dropna().unique())
    if not cohort:
        return cohort, pd.DataFrame()
    sub = h[h["patient_id"].astype("Int64").isin(cohort)].copy()
    metrics = compute_refill_cycle(sub, post_from, date.today())
    return cohort, metrics


with st.spinner("Building cohort & estimating refill cycles…"):
    cohort, metrics = build_cohort_and_metrics(
        post_from, post_to, tuple(sorted(selected_pilots))
    )

if not cohort or metrics.empty:
    st.info("No patients match the cohort definition with the current filters.")
    st.stop()


# ----- Hero KPI strip -------------------------------------------------------

n_patients = len(cohort)
n_pairs = len(metrics)
patients_overdue = set(metrics.loc[metrics["status"] == "overdue", "patient_id"].astype(int))
pct_overdue = len(patients_overdue) / n_patients * 100 if n_patients else 0
pct_refilled = (metrics["status"] == "refilled").mean() * 100 if n_pairs else 0

st.markdown("---")
k1, k2, k3, k4 = st.columns(4)
k1.metric("👥 Cohort patients", f"{n_patients:,}")
k2.metric("💊 Patient-drug pairs", f"{n_pairs:,}",
          help="Number of (patient × chronic price-↑ SKU) pairs with ≥1 historical purchase.")
k3.metric("✅ On-track refill rate",
          f"{pct_refilled:.1f}%",
          help="Share of patient-drug pairs where the patient bought the drug again at any chain store post-go-live.")
k4.metric("⚠️ Patients overdue (1+ drug)",
          f"{len(patients_overdue):,}",
          f"{pct_overdue:.1f}% of cohort",
          delta_color="inverse")


# ----- Section A — Funnel by status ---------------------------------------

st.markdown("---")
st.subheader("Section A · How does the cohort split?")

status_counts = (
    metrics["status"].value_counts().reindex(STATUS_ORDER, fill_value=0).reset_index()
)
status_counts.columns = ["status", "count"]
status_counts["label"] = status_counts["status"].map(STATUS_LABEL)
status_counts["pct"] = status_counts["count"] / status_counts["count"].sum() * 100

bar = (
    alt.Chart(status_counts)
    .mark_bar()
    .encode(
        x=alt.X("count:Q", title="Patient-drug pairs"),
        y=alt.Y("label:N", title=None,
                sort=[STATUS_LABEL[s] for s in STATUS_ORDER]),
        color=alt.Color("status:N",
                        scale=alt.Scale(domain=STATUS_ORDER,
                                        range=[STATUS_COLOR[s] for s in STATUS_ORDER]),
                        legend=None),
        tooltip=["label", "count", alt.Tooltip("pct", format=".1f", title="%")],
    )
    .properties(height=180)
)
st.altair_chart(bar, width="stretch")
st.caption(
    "`refilled` = patient bought the drug again after their last pre-go-live purchase. "
    "`on_time` = no post purchase yet, but they're not past expected refill date. "
    "`overdue` = past expected refill date by >7 days, no purchase recorded anywhere in chain. "
    "`no_estimate` = only one historical purchase — can't estimate a cycle."
)


# ----- Section B — Overdue patients (headline) ----------------------------

st.markdown("---")
st.subheader("Section B · Overdue patients")
st.caption(
    f"Patients overdue by >{REFILL_GRACE_DAYS} days on ≥1 chronic price-↑ SKU, "
    "with no refill anywhere in the chain. **The actionable list for outreach.**"
)

overdue = metrics[metrics["status"] == "overdue"].copy()
if overdue.empty:
    st.success("🎉 No overdue patients with current filters.")
else:
    by_patient = (
        overdue.groupby("patient_id")
        .agg(
            n_overdue_drugs=("drug_id", "nunique"),
            max_days_overdue=("days_overdue", "max"),
            avg_days_overdue=("days_overdue", "mean"),
            last_pre_date=("last_pre_date", "max"),
            last_pre_store=("last_pre_store", "first"),
            most_overdue_drug=("drug_name",
                               lambda s: overdue.loc[s.index].sort_values("days_overdue", ascending=False)["drug_name"].iloc[0]),
        )
        .reset_index()
        .sort_values("max_days_overdue", ascending=False)
    )
    by_patient["avg_days_overdue"] = by_patient["avg_days_overdue"].round(0).astype(int)
    by_patient["max_days_overdue"] = by_patient["max_days_overdue"].astype(int)

    c1, c2, c3 = st.columns(3)
    c1.metric("Patients overdue", f"{len(by_patient):,}")
    c2.metric("Median days overdue (per patient max)", f"{int(by_patient['max_days_overdue'].median())}")
    c3.metric("Total overdue drug-instances", f"{len(overdue):,}")

    st.dataframe(
        by_patient.head(200),
        width="stretch",
        hide_index=True,
        column_config={
            "patient_id": st.column_config.NumberColumn("Patient ID", format="%d", pinned=True),
            "n_overdue_drugs": st.column_config.NumberColumn("# Overdue drugs"),
            "max_days_overdue": st.column_config.NumberColumn("Max days overdue"),
            "avg_days_overdue": st.column_config.NumberColumn("Avg days overdue"),
            "most_overdue_drug": "Most-overdue drug",
            "last_pre_store": "Last pre store",
            "last_pre_date": st.column_config.DateColumn("Last pre date"),
        },
    )
    st.download_button(
        "📥 Download full overdue list (CSV)",
        data=by_patient.to_csv(index=False).encode("utf-8"),
        file_name="overdue_chronic_patients.csv",
        mime="text/csv",
    )


# ----- Section C — Days overdue distribution ------------------------------

st.markdown("---")
st.subheader("Section C · How far overdue?")

dist_src = metrics[metrics["status"].isin(["refilled", "on_time", "overdue"])].copy()
dist_src["days_overdue"] = dist_src["days_overdue"].astype(float)

if dist_src.empty:
    st.info("No data for distribution chart.")
else:
    hist_chart = (
        alt.Chart(dist_src)
        .mark_bar()
        .encode(
            x=alt.X("days_overdue:Q",
                    bin=alt.Bin(maxbins=40),
                    title="Days past expected refill date (negative = not yet due)"),
            y=alt.Y("count():Q", title="Patient-drug pairs"),
            color=alt.Color("status:N",
                            scale=alt.Scale(domain=STATUS_ORDER,
                                            range=[STATUS_COLOR[s] for s in STATUS_ORDER]),
                            legend=alt.Legend(title="Status")),
            tooltip=["status", "count()"],
        )
        .properties(height=260)
    )
    rule = (
        alt.Chart(pd.DataFrame({"x": [0]}))
        .mark_rule(color="#2C3E50", strokeDash=[4, 3])
        .encode(x="x:Q")
    )
    st.altair_chart(hist_chart + rule, width="stretch")
    st.caption(
        "x = days past expected refill date (`today − expected`). "
        "Bars to the right of the dashed line are due-or-overdue; "
        "the red slice is what's actually overdue with no recorded refill."
    )


# ----- Section D — Drug-level overdue rate --------------------------------

st.markdown("---")
st.subheader("Section D · Which chronic SKUs lapse most?")

drug_summary = (
    metrics.groupby(["drug_id", "drug_name"])
    .agg(
        cohort=("patient_id", "nunique"),
        overdue=("status", lambda s: (s == "overdue").sum()),
        refilled=("status", lambda s: (s == "refilled").sum()),
    )
    .reset_index()
)
drug_summary["overdue_rate"] = drug_summary["overdue"] / drug_summary["cohort"] * 100
top_drugs = (
    drug_summary[drug_summary["cohort"] >= 5]
    .sort_values("overdue", ascending=False)
    .head(20)
)

if top_drugs.empty:
    st.info("Too few patients per drug to plot reliable rates.")
else:
    drug_bar = (
        alt.Chart(top_drugs)
        .mark_bar(color=COLOR_OVERDUE)
        .encode(
            x=alt.X("overdue:Q", title="Patients overdue"),
            y=alt.Y("drug_name:N", sort="-x", title=None),
            tooltip=[
                "drug_name",
                alt.Tooltip("cohort", title="Cohort patients (this drug)"),
                alt.Tooltip("overdue", title="Overdue"),
                alt.Tooltip("overdue_rate", title="Overdue rate (%)", format=".1f"),
            ],
        )
        .properties(height=max(220, 22 * len(top_drugs)))
    )
    st.altair_chart(drug_bar, width="stretch")
    st.caption(
        "Top 20 chronic price-↑ SKUs by absolute overdue-patient count "
        "(filtered to drugs with ≥5 patients in the cohort, so percentages are meaningful)."
    )


# ----- Section E — Cycle estimate quality ---------------------------------

st.markdown("---")
with st.expander("Section E · How reliable are the cycle estimates?", expanded=False):
    bin_def = pd.cut(
        metrics["n_pre_purchases"],
        bins=[0, 1, 2, 3, 4, 1000],
        labels=["1 purchase (no est.)", "2", "3", "4", "5+"],
        right=True,
    )
    counts = bin_def.value_counts().sort_index().reset_index()
    counts.columns = ["Pre purchases", "Patient-drug pairs"]
    st.dataframe(counts, width="stretch", hide_index=True)
    st.caption(
        "More pre-purchases per (patient × drug) → more stable cycle estimate. "
        "Pairs with only 1 purchase get `status = no_estimate` and are excluded "
        "from the overdue list."
    )


# ----- Section F — Patient drill-down (timeline) --------------------------

st.markdown("---")
with st.expander("Section F · Patient timeline drill-down", expanded=False):
    overdue_pids = sorted(set(int(x) for x in metrics.loc[metrics["status"] == "overdue", "patient_id"]))
    pids_pick = st.multiselect(
        "Pick patient ID(s)",
        options=overdue_pids[:200],
        max_selections=5,
        help="Pick up to 5 overdue patients to see their purchase timeline. "
             "Top 200 overdue patients shown.",
    )
    if pids_pick:
        h_pick = _load_cohort_history_snapshot()
        h_pick = h_pick[h_pick["bill_flag"].astype(str).str.lower() == "gross"]
        h_pick = h_pick[h_pick["patient_id"].astype("Int64").isin(set(pids_pick))].copy()
        h_pick["day"] = pd.to_datetime(h_pick["created_at"]).dt.date
        h_pick = h_pick.sort_values(["patient_id", "drug_name", "day"])

        # Expected-refill markers from metrics
        markers = metrics[metrics["patient_id"].astype(int).isin(pids_pick)].copy()
        markers["day"] = pd.to_datetime(markers["expected_refill_date"]).dt.date

        purchase_chart = (
            alt.Chart(h_pick)
            .mark_circle(size=110, opacity=0.85)
            .encode(
                x=alt.X("day:T", title="Date"),
                y=alt.Y("drug_name:N", title=None),
                color=alt.Color("patient_id:N", legend=alt.Legend(title="Patient")),
                tooltip=["patient_id", "drug_name", "day", "store_name", "net_quantity"],
            )
            .properties(height=max(180, 28 * h_pick["drug_name"].nunique()))
        )
        go_live = (
            alt.Chart(pd.DataFrame({"x": [pd.Timestamp("2026-05-06")]}))
            .mark_rule(color="#2C3E50", strokeDash=[4, 3])
            .encode(x="x:T")
        )
        expected_marks = (
            alt.Chart(markers.dropna(subset=["day"]))
            .mark_point(shape="diamond", size=140, color=COLOR_OVERDUE, opacity=0.8)
            .encode(x="day:T", y="drug_name:N",
                    tooltip=["patient_id", "drug_name",
                             alt.Tooltip("days_overdue", title="Days overdue")])
        )
        st.altair_chart(purchase_chart + go_live + expected_marks,
                        width="stretch")
        st.caption(
            "Circles = actual purchases (any chain store). Red diamond = expected refill date. "
            "Dashed line = 2026-05-06 go-live."
        )


# ----- Footer ---------------------------------------------------------------

st.markdown("---")
st.caption(
    f"Snapshot date range: {COHORT_HISTORY_START} → today. "
    f"Refresh via `python refresh_snapshots.py` (requires VPN)."
)

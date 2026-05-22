"""Bill-by-bill Progression — chronic price-↑ SKUs silently dropped.

For each chronic-cohort patient, compare their latest post-window bill
against the previous 1st/2nd/3rd most-recent bills to identify chronic
price-↑ SKUs they used to buy but didn't this time.
"""

from __future__ import annotations

from datetime import date

import altair as alt
import pandas as pd
import streamlit as st

from data import (
    COHORT_HISTORY_START,
    PILOT_STORES,
    QUANT_POST_END,
    QUANT_POST_START,
    _load_cohort_history_snapshot,
    chronic_pi_drug_ids,
    compute_bill_progression,
    load_skus,
)

DROP_COLOR = "#E74C3C"
NEUTRAL_COLOR = "#2E86C1"


# ============================================================================
# Page setup
# ============================================================================

st.title("📉 Chronic Bill-by-Bill Progression")
st.caption(
    "What chronic price-↑ SKUs are patients silently dropping from their "
    "basket? For each cohort patient we compare the **latest** post-window "
    "bill against their previous **1st / 2nd / 3rd** bills (across the whole "
    "chain). A SKU present earlier but missing in the latest bill is a *drop*."
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
st.sidebar.header("Progression filters")
c1, c2 = st.sidebar.columns(2)
post_from = c1.date_input("Post from", value=QUANT_POST_START, key="bp_post_from")
post_to = c2.date_input("Post to", value=QUANT_POST_END, key="bp_post_to")
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
    f"📂 History window in snapshot: {COHORT_HISTORY_START} → today"
)


# ----- Build cohort + run analysis -----------------------------------------

@st.cache_data(ttl=900, show_spinner=False)
def build_progression(
    post_from: date, post_to: date, pilots: tuple[str, ...]
) -> tuple[set[int], pd.DataFrame, pd.DataFrame]:
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
        return cohort, pd.DataFrame(), pd.DataFrame()
    sub = h[h["patient_id"].astype("Int64").isin(cohort)].copy()
    per_patient, per_drug = compute_bill_progression(sub, post_from, post_to, chronic_ids)
    return cohort, per_patient, per_drug


with st.spinner("Comparing latest bill vs prior bills…"):
    cohort, per_patient, per_drug = build_progression(
        post_from, post_to, tuple(sorted(selected_pilots))
    )

if not cohort or per_patient.empty:
    st.info("No patients match the cohort definition with the current filters.")
    st.stop()


# ----- Hero KPI strip -------------------------------------------------------

n_cohort = len(cohort)
n_with_prior = int((per_patient["prev1_date"].notna()).sum())
patients_dropped_any = int(
    ((per_patient["dropped_vs_prev1"] > 0)
     | (per_patient["dropped_vs_prev2"] > 0)
     | (per_patient["dropped_vs_prev3"] > 0)).sum()
)
total_drop_instances = int(
    per_patient[["dropped_vs_prev1", "dropped_vs_prev2", "dropped_vs_prev3"]].sum().sum()
)
pct_drop = (patients_dropped_any / n_cohort * 100) if n_cohort else 0

st.markdown("---")
k1, k2, k3, k4 = st.columns(4)
k1.metric("👥 Cohort patients", f"{n_cohort:,}")
k2.metric("📋 Patients with ≥1 prior bill", f"{n_with_prior:,}")
k3.metric("📉 Dropped ≥1 chronic SKU (vs last bill)",
          f"{int((per_patient['dropped_vs_prev1'] > 0).sum()):,}",
          help="Latest bill is missing ≥1 chronic price-↑ SKU that was on their immediately-prior bill.")
k4.metric("🔻 Total drop instances (all 3 windows)",
          f"{total_drop_instances:,}",
          help="Sum of dropped-SKU events across vs_prev1, vs_prev2, vs_prev3.")


# ----- Section A — Dropped counts by lookback window ----------------------

st.markdown("---")
st.subheader("Section A · How many SKUs got dropped, by lookback window?")

agg = pd.DataFrame({
    "lookback": ["vs Previous bill", "vs 2nd-previous bill", "vs 3rd-previous bill"],
    "patients": [
        int((per_patient["dropped_vs_prev1"] > 0).sum()),
        int((per_patient["dropped_vs_prev2"] > 0).sum()),
        int((per_patient["dropped_vs_prev3"] > 0).sum()),
    ],
    "drop_instances": [
        int(per_patient["dropped_vs_prev1"].sum()),
        int(per_patient["dropped_vs_prev2"].sum()),
        int(per_patient["dropped_vs_prev3"].sum()),
    ],
})
melted = agg.melt(id_vars="lookback", var_name="metric", value_name="value")
melted["metric"] = melted["metric"].map({
    "patients": "Patients with ≥1 drop",
    "drop_instances": "Total drop instances",
})
chart_a = (
    alt.Chart(melted)
    .mark_bar()
    .encode(
        x=alt.X("lookback:N", title=None,
                sort=["vs Previous bill", "vs 2nd-previous bill", "vs 3rd-previous bill"]),
        y=alt.Y("value:Q", title="Count"),
        color=alt.Color("metric:N",
                        scale=alt.Scale(domain=["Patients with ≥1 drop", "Total drop instances"],
                                        range=[DROP_COLOR, NEUTRAL_COLOR]),
                        legend=alt.Legend(orient="top", title=None)),
        xOffset="metric:N",
        tooltip=["lookback", "metric", "value"],
    )
    .properties(height=260)
)
st.altair_chart(chart_a, width="stretch")
st.caption(
    "A drop is persistent if it shows up across multiple lookbacks — the SKU was on "
    "prev1 AND prev2 AND prev3 but vanished from the latest bill."
)


# ----- Section B — Drop heatmap (drug × lookback) -------------------------

st.markdown("---")
st.subheader("Section B · Which chronic SKUs are getting dropped?")

if per_drug.empty:
    st.info("No drops to plot.")
else:
    top_drugs = per_drug.head(30).copy()
    hm_long = top_drugs.melt(
        id_vars=["drug_id", "drug_name"],
        value_vars=["vs_prev1", "vs_prev2", "vs_prev3"],
        var_name="lookback",
        value_name="patients_dropped",
    )
    hm_long["lookback"] = hm_long["lookback"].map({
        "vs_prev1": "Latest vs prev1",
        "vs_prev2": "Latest vs prev2",
        "vs_prev3": "Latest vs prev3",
    })
    heatmap = (
        alt.Chart(hm_long)
        .mark_rect()
        .encode(
            x=alt.X("lookback:N", title=None,
                    sort=["Latest vs prev1", "Latest vs prev2", "Latest vs prev3"]),
            y=alt.Y("drug_name:N", title=None,
                    sort=top_drugs["drug_name"].tolist()),
            color=alt.Color("patients_dropped:Q",
                            scale=alt.Scale(scheme="reds"),
                            title="Patients dropped"),
            tooltip=["drug_name", "lookback", "patients_dropped"],
        )
        .properties(height=max(280, 22 * len(top_drugs)))
    )
    text = (
        alt.Chart(hm_long)
        .mark_text(fontSize=11, fontWeight=600)
        .encode(
            x=alt.X("lookback:N",
                    sort=["Latest vs prev1", "Latest vs prev2", "Latest vs prev3"]),
            y=alt.Y("drug_name:N", sort=top_drugs["drug_name"].tolist()),
            text=alt.Text("patients_dropped:Q"),
            color=alt.condition("datum.patients_dropped > 8",
                                alt.value("white"), alt.value("#2C3E50")),
        )
    )
    st.altair_chart(heatmap + text, width="stretch")
    st.caption(
        "Rows = top 30 chronic SKUs by total drop count. "
        "A row with high values across all 3 columns is being persistently dropped."
    )


# ----- Section C — Patient-level table ------------------------------------

st.markdown("---")
st.subheader("Section C · Patient-level drops")

# Augment with dropped-drug names (concat top per row).
skus = load_skus()
name_lookup = dict(zip(skus["drug_id"].astype(int), skus["drug_name"]))


def drug_list_to_names(ids: list[int]) -> str:
    if not isinstance(ids, list) or not ids:
        return ""
    names = [name_lookup.get(int(d), str(d)) for d in ids[:5]]
    if len(ids) > 5:
        names.append(f"+{len(ids) - 5} more")
    return ", ".join(names)


tbl = per_patient.copy()
tbl["dropped_vs_prev1_names"] = tbl["dropped_vs_prev1_drugs"].apply(drug_list_to_names)
tbl["dropped_vs_prev2_names"] = tbl["dropped_vs_prev2_drugs"].apply(drug_list_to_names)
tbl["dropped_vs_prev3_names"] = tbl["dropped_vs_prev3_drugs"].apply(drug_list_to_names)
tbl = tbl[[
    "patient_id", "latest_bill_date", "latest_store", "n_chronic_pi_latest",
    "dropped_vs_prev1", "dropped_vs_prev1_names",
    "dropped_vs_prev2", "dropped_vs_prev2_names",
    "dropped_vs_prev3", "dropped_vs_prev3_names",
    "prev1_date", "prev2_date", "prev3_date",
]].sort_values("dropped_vs_prev1", ascending=False)

st.dataframe(
    tbl.head(200),
    width="stretch",
    hide_index=True,
    column_config={
        "patient_id": st.column_config.NumberColumn("Patient ID", format="%d", pinned=True),
        "latest_bill_date": st.column_config.DateColumn("Latest bill"),
        "latest_store": "Latest store",
        "n_chronic_pi_latest": st.column_config.NumberColumn("# chronic-PI on latest"),
        "dropped_vs_prev1": st.column_config.NumberColumn("Dropped vs prev1"),
        "dropped_vs_prev1_names": "Dropped (vs prev1)",
        "dropped_vs_prev2": st.column_config.NumberColumn("Dropped vs prev2"),
        "dropped_vs_prev2_names": "Dropped (vs prev2)",
        "dropped_vs_prev3": st.column_config.NumberColumn("Dropped vs prev3"),
        "dropped_vs_prev3_names": "Dropped (vs prev3)",
        "prev1_date": st.column_config.DateColumn("Prev1 date"),
        "prev2_date": st.column_config.DateColumn("Prev2 date"),
        "prev3_date": st.column_config.DateColumn("Prev3 date"),
    },
)
st.download_button(
    "📥 Download full patient progression (CSV)",
    data=tbl.to_csv(index=False).encode("utf-8"),
    file_name="bill_progression_patients.csv",
    mime="text/csv",
)


# ----- Section D — Drug-level rollup --------------------------------------

st.markdown("---")
with st.expander("Section D · Drug-level drop summary", expanded=False):
    if per_drug.empty:
        st.info("No drug-level drops.")
    else:
        # Cohort patients who bought each chronic-PI drug in ANY prior bill
        h_buyers = _load_cohort_history_snapshot()
        h_buyers = h_buyers[h_buyers["bill_flag"].astype(str).str.lower() == "gross"]
        h_buyers = h_buyers[h_buyers["patient_id"].astype("Int64").isin(cohort)]
        h_buyers = h_buyers[h_buyers["drug_id"].astype("Int64").isin(chronic_ids)]
        h_buyers["day"] = pd.to_datetime(h_buyers["created_at"]).dt.date
        prior_buyers = (
            h_buyers[h_buyers["day"] < post_from]
            .groupby("drug_id")["patient_id"].nunique()
            .reset_index()
            .rename(columns={"patient_id": "prior_buyer_patients"})
        )
        rollup = per_drug.merge(prior_buyers, on="drug_id", how="left")
        rollup["prior_buyer_patients"] = rollup["prior_buyer_patients"].fillna(0).astype(int)
        rollup["drop_rate_vs_prev1"] = (
            rollup["vs_prev1"] / rollup["prior_buyer_patients"].replace(0, pd.NA) * 100
        ).round(1)
        st.dataframe(
            rollup,
            width="stretch",
            hide_index=True,
            column_config={
                "drug_id": st.column_config.NumberColumn("Drug ID", format="%d"),
                "drug_name": st.column_config.Column("Drug name", pinned=True),
                "vs_prev1": st.column_config.NumberColumn("Dropped (vs prev1)"),
                "vs_prev2": st.column_config.NumberColumn("Dropped (vs prev2)"),
                "vs_prev3": st.column_config.NumberColumn("Dropped (vs prev3)"),
                "total_dropped": st.column_config.NumberColumn("Total drop instances"),
                "prior_buyer_patients": st.column_config.NumberColumn("Prior cohort buyers"),
                "drop_rate_vs_prev1": st.column_config.NumberColumn("Drop rate vs prev1 (%)"),
            },
        )


# ----- Section E — Patient timeline drill-down ----------------------------

st.markdown("---")
with st.expander("Section E · Patient timeline drill-down", expanded=False):
    top_droppers = (
        per_patient.sort_values("dropped_vs_prev1", ascending=False)
        .loc[lambda d: d["dropped_vs_prev1"] > 0, "patient_id"]
        .astype(int)
        .head(200)
        .tolist()
    )
    pids_pick = st.multiselect(
        "Pick patient ID(s)",
        options=top_droppers,
        max_selections=5,
        help="Top 200 patients who dropped the most SKUs vs their previous bill.",
    )
    if pids_pick:
        h_pick = _load_cohort_history_snapshot()
        h_pick = h_pick[h_pick["bill_flag"].astype(str).str.lower() == "gross"]
        h_pick = h_pick[h_pick["patient_id"].astype("Int64").isin(set(pids_pick))].copy()
        h_pick["day"] = pd.to_datetime(h_pick["created_at"]).dt.date
        h_pick = h_pick.sort_values(["patient_id", "drug_name", "day"])

        timeline = (
            alt.Chart(h_pick)
            .mark_circle(size=130, opacity=0.85)
            .encode(
                x=alt.X("day:T", title="Date"),
                y=alt.Y("drug_name:N", title=None),
                color=alt.Color("patient_id:N", legend=alt.Legend(title="Patient")),
                tooltip=["patient_id", "drug_name", "day", "store_name", "net_quantity"],
            )
            .properties(height=max(220, 28 * h_pick["drug_name"].nunique()))
        )
        go_live = (
            alt.Chart(pd.DataFrame({"x": [pd.Timestamp("2026-05-06")]}))
            .mark_rule(color="#2C3E50", strokeDash=[4, 3])
            .encode(x="x:T")
        )
        st.altair_chart(timeline + go_live, width="stretch")
        st.caption(
            "Circles = purchases (any chain store). Dashed line = 2026-05-06 go-live. "
            "Gaps between circles for the same drug = patient skipped that refill."
        )


# ----- Footer ---------------------------------------------------------------

st.markdown("---")
st.caption(
    f"Snapshot date range: {COHORT_HISTORY_START} → today. "
    f"Refresh via `python refresh_snapshots.py` (requires VPN). "
    "Bills are keyed by (patient × store × day) — multiple till transactions on the "
    "same day at the same store are treated as one visit."
)

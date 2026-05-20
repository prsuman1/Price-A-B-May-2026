"""Streamlit dashboard for the price-change A/B test."""

from __future__ import annotations

from datetime import date

import altair as alt
import pandas as pd
import streamlit as st

from data import (
    GO_LIVE_DATE, RAG_COLORS, RAG_EMOJI, STORE_MAP,
    _load_partial_bill_snapshot, build_joined, write_clean_bill_ids_csv,
)

# Display palette for the 4 chart segments (amber split into partial/genrtn).
SEGMENT_COLORS = {
    "green": "#2ECC71",
    "amber_partial": "#F39C12",
    "amber_genrtn": "#F1C40F",
    "red": "#E74C3C",
}
SEGMENT_ORDER = ["green", "amber_partial", "amber_genrtn", "red"]
SEGMENT_LABELS = {
    "green": "🟢 Green (no change)",
    "amber_partial": "🟡 Amber — partial bill",
    "amber_genrtn": "🟡 Amber — GENRTN coupon",
    "red": "🔴 Red (walk-away)",
}


def rag_detail(row: pd.Series) -> str:
    """Map per-bill RAG + sub-flags to one of 4 display segments."""
    if row["rag"] == "red":
        return "red"
    if row["rag"] == "green":
        return "green"
    # amber: split by which sub-flag is set; both → genrtn (rarer + more informative).
    if row.get("is_amber_genrtn", False):
        return "amber_genrtn"
    return "amber_partial"

st.title("Price Change A/B — Field Insight Dashboard")

# -----------------------------------------------------------------------------
# Load
# -----------------------------------------------------------------------------
with st.spinner("Loading data…"):
    bundle = build_joined()

survey: pd.DataFrame = bundle["survey"]
skus: pd.DataFrame = bundle["skus"]
sales: pd.DataFrame = bundle["sales"]
bills: pd.DataFrame = bundle["bills"]
missing_serials: list[str] = bundle["missing_serials"]
stats: dict = bundle["stats"]

# Regenerate the human-readable clean-bills sheet on every page render.
clean_bills_path = write_clean_bill_ids_csv(survey)

# -----------------------------------------------------------------------------
# Sidebar filters
# -----------------------------------------------------------------------------
st.sidebar.header("Filters")

min_date = max(GO_LIVE_DATE.date(), survey["submitted_at"].min().date()) if not survey.empty else GO_LIVE_DATE.date()
max_date = survey["submitted_at"].max().date() if not survey.empty else date.today()
date_range = st.sidebar.date_input(
    "Date range (Submitted at)",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date,
)
if isinstance(date_range, tuple) and len(date_range) == 2:
    d_from, d_to = date_range
else:
    d_from, d_to = min_date, max_date

all_stores = sorted(survey["store"].unique().tolist())
sel_stores = st.sidebar.multiselect("Stores", options=all_stores, default=all_stores)

drug_types = sorted(skus["drug_type"].dropna().unique().tolist())
sel_types = st.sidebar.multiselect("SKU drug type", options=drug_types, default=drug_types)

# Apply filters
mask_survey = (
    (survey["submitted_at"].dt.date >= d_from)
    & (survey["submitted_at"].dt.date <= d_to)
    & (survey["store"].isin(sel_stores))
)
fsurvey = survey.loc[mask_survey].copy()

# Restrict SKU set by drug-type filter, then recompute has_price_increased_sku per bill
selected_drug_ids = set(skus.loc[skus["drug_type"].isin(sel_types), "drug_id"].astype(int))

mask_bills = (
    (bills["submitted_at"].dt.date >= d_from)
    & (bills["submitted_at"].dt.date <= d_to)
    & (bills["store"].isin(sel_stores))
)
fbills = bills.loc[mask_bills].copy()

# Recompute has_price_increased_sku based on filtered SKU set (per-line count from sales)
if not sales.empty and not fbills.empty:
    sales_filtered = sales[sales["drug_id"].astype("Int64").isin(selected_drug_ids)]
    increased_serials = set(sales_filtered["serial"].unique())
    fbills["has_price_increased_sku"] = fbills["serial_norm"].isin(increased_serials)
else:
    fbills["has_price_increased_sku"] = False

# -----------------------------------------------------------------------------
# Top KPIs — RAG chips + secondary functional KPIs
# -----------------------------------------------------------------------------
n_survey = len(fsurvey)
n_joined = int(fbills["bill_id"].notna().sum())
n_with_increase = int(fbills["has_price_increased_sku"].sum())
n_walkaways = int(fsurvey["is_walkaway"].sum())
n_genrtn = int(fbills["used_genrtn"].fillna(False).sum())
n_no_id = int((fsurvey["bill_kind"] == "bill_created_no_id").sum())
n_bills_created = int((fsurvey["bill_kind"].isin(["serial", "bill_created_no_id"])).sum())

n_green = int((fsurvey["rag"] == "green").sum())
n_amber = int((fsurvey["rag"] == "amber").sum())
n_amber_partial = int(fsurvey["is_amber_partial"].sum())
n_amber_genrtn = int(fsurvey["is_amber_genrtn"].sum())
n_red = int((fsurvey["rag"] == "red").sum())
pct = (lambda n: f"{(100*n/n_survey):.1f}%" if n_survey else "—")

g, a, r = st.columns(3)
g.metric(f"{RAG_EMOJI['green']} Green (no change)", n_green, pct(n_green))
a.metric(
    f"{RAG_EMOJI['amber']} Amber (partial / GENRTN)",
    n_amber,
    f"{pct(n_amber)} · partial: {n_amber_partial} · genrtn: {n_amber_genrtn}",
)
r.metric(f"{RAG_EMOJI['red']} Red (walk-away)", n_red, pct(n_red))

st.markdown("**Functional KPIs**")
k1, k2, k3, k4, k5, k6 = st.columns(6)
k1.metric("Survey rows", n_survey)
k2.metric("Bills created", n_bills_created)
k3.metric("Bills w/ price-↑ SKU", n_with_increase)
k4.metric("Bills joined to sales", n_joined)
k5.metric("GENRTN bills (sales)", n_genrtn)
k6.metric("Bill created, no ID", n_no_id)

# Funnel widget — show the math behind the headline KPIs
n_serial = int((fsurvey["bill_kind"] == "serial").sum())
n_sync_lag = n_serial - n_joined
funnel_df = pd.DataFrame(
    [
        ("Survey rows (post go-live)", n_survey),
        ("├── Bills created (per outcome column)", n_bills_created),
        ("│   ├── With parseable serial", n_serial),
        ("│   │   ├── Matched in sales (joinable)", n_joined),
        ("│   │   └── Not yet in sales (sync lag)", n_sync_lag),
        ("│   └── No serial entered (bill_created_no_id)", n_no_id),
        ("└── Walk-aways (outcome = 'Bana hi nahin')", n_walkaways),
    ],
    columns=["Step", "Count"],
)
with st.expander("Funnel — how 'survey rows' breaks down into bills / walkaways", expanded=False):
    st.dataframe(funnel_df, hide_index=True, width='stretch')
    extra = []
    if stats.get("typo_fixed_kewt_to_krwt"):
        extra.append(f"`KEWT → KRWT` typo correction applied to **{stats['typo_fixed_kewt_to_krwt']}** survey rows (Khar West).")
    if extra:
        st.caption(" · ".join(extra))

with st.expander("Data freshness — survey serials not yet in sales", expanded=False):
    if missing_serials:
        st.write(f"{len(missing_serials)} survey serial(s) not found in `sales` (likely Redshift sync lag for today's bills):")
        st.code("\n".join(missing_serials), language=None)
    else:
        st.success("All survey serials matched in sales.")

st.download_button(
    "📥 Download clean bill IDs (CSV)",
    data=clean_bills_path.read_bytes(),
    file_name="clean_bill_ids.csv",
    mime="text/csv",
    help="Per-row mapping of what each surveyor bill value got normalized to. "
         "Open in Excel/Sheets to spot any remaining outliers.",
)

# -----------------------------------------------------------------------------
# Tabs
# -----------------------------------------------------------------------------
(
    tab_rag,
    tab_store,
    tab_sent,
    tab_walk,
    tab_partial,
    tab_ctrl,
    tab_drill,
) = st.tabs([
    "RAG by Store × SKU",
    "Store cut",
    "Sentiment cross-tabs",
    "Walk-aways",
    "Partial-bill retention",
    "Control comparison",
    "Drill-down",
])

# Pre-compute fbills["rag_detail"] once for use across multiple tabs.
if not fbills.empty:
    fbills["rag_detail"] = fbills.apply(rag_detail, axis=1)
else:
    fbills["rag_detail"] = pd.Series([], dtype=str)

# ---- RAG by Store × SKU (HEADLINE) ------------------------------------------
with tab_rag:
    st.subheader("Bill health by store, split by price-increased SKU presence")
    st.caption(
        "Two horizontal bars per store — top: bills WITH a price-↑ SKU, bottom: bills WITHOUT. "
        "Segments: 🟢 no change · 🟠 partial bill. "
        "(🔴 Walk-aways have no bill data → see **Walk-aways** tab.)"
    )

    if fbills.empty:
        st.info("No joined bills in this filter.")
    else:
        chart_df = fbills.copy()
        chart_df["sku_presence"] = chart_df["has_price_increased_sku"].map(
            {True: "With price-↑ SKU", False: "Without price-↑ SKU"}
        )
        agg = (
            chart_df.groupby(["store", "sku_presence", "rag_detail"], dropna=False)
            .size().reset_index(name="bills")
        )
        # Headline chart only shows the two bill-level RAG segments.
        # Walk-aways have no bill data → see Walk-aways tab.
        # GENRTN bills → see GENRTN tab.
        chart_segments = ["green", "amber_partial"]
        chart_agg = agg[agg["rag_detail"].isin(chart_segments)]
        chart_agg["segment"] = chart_agg["rag_detail"].map(SEGMENT_LABELS)

        chart = (
            alt.Chart(chart_agg)
            .mark_bar(size=14)
            .encode(
                y=alt.Y("store:N", title=None,
                        sort=alt.EncodingSortField(field="bills", op="sum", order="descending")),
                yOffset=alt.YOffset("sku_presence:N",
                                    sort=["With price-↑ SKU", "Without price-↑ SKU"]),
                x=alt.X("bills:Q", title="Bills"),
                color=alt.Color(
                    "rag_detail:N",
                    scale=alt.Scale(
                        domain=chart_segments,
                        range=[SEGMENT_COLORS[k] for k in chart_segments],
                    ),
                    legend=alt.Legend(title=None, labelExpr=(
                        "{'green':'🟢 No change',"
                        "'amber_partial':'🟠 Partial bill'}[datum.label]"
                    )),
                ),
                tooltip=["store", "sku_presence", "segment", "bills"],
            )
            .properties(height=400)
        )
        st.altair_chart(chart, width='stretch')

        # Cross-tab beneath the chart — same two segments only
        st.markdown("**Cross-tab — bills × store × SKU presence**")
        ct = (
            chart_df[chart_df["rag_detail"].isin(chart_segments)]
            .groupby(["store", "sku_presence", "rag_detail"]).size()
            .unstack("rag_detail", fill_value=0)
        )
        for k in chart_segments:
            if k not in ct.columns: ct[k] = 0
        ct = ct[chart_segments]
        ct.columns = ["🟢 No change", "🟠 Partial bill"]
        ct["TOTAL"] = ct.sum(axis=1)
        st.dataframe(ct.reset_index(), width='stretch')

        # Walk-away + no-ID side note (these don't have SKU presence)
        non_bill = fsurvey[~fsurvey["bill_kind"].eq("serial")]
        if not non_bill.empty:
            st.markdown("**Customer interactions excluded from chart (no SKU info)**")
            totals = fsurvey.groupby("store").size().rename("Total interactions")
            excluded = (
                non_bill.groupby(["store", "bill_kind"]).size()
                .unstack("bill_kind", fill_value=0)
            )
            for k in ["bill_created_no_id", "walkaway"]:
                if k not in excluded.columns: excluded[k] = 0
            extra = (
                totals.to_frame()
                .join(excluded[["bill_created_no_id", "walkaway"]], how="left")
                .fillna(0).astype(int)
            )
            extra.columns = ["Total interactions", "Bill made, ID not entered", "Walk-away"]
            extra = extra.reset_index()
            total_row = pd.DataFrame([{
                "store": "TOTAL",
                "Total interactions": int(extra["Total interactions"].sum()),
                "Bill made, ID not entered": int(extra["Bill made, ID not entered"].sum()),
                "Walk-away": int(extra["Walk-away"].sum()),
            }])
            extra = pd.concat([extra, total_row], ignore_index=True)
            st.dataframe(extra, width='stretch', hide_index=True)

    # Audit expander
    with st.expander("How RAG was assigned (audit)", expanded=False):
        oa = stats.get("outcome_audit", {})
        cols = st.columns(4)
        cols[0].metric("Canonical 🟢", oa.get("n_canonical_green", 0))
        cols[1].metric("Canonical 🟡 partial", oa.get("n_canonical_partial", 0))
        cols[2].metric("Canonical 🔴", oa.get("n_canonical_walkaway", 0))
        cols[3].metric("Free-text outcomes", oa.get("n_freetext_total", 0))

        st.write(
            f"Free-text → 🔴: **{oa.get('n_freetext_to_red', 0)}** · "
            f"→ 🟡 partial: **{oa.get('n_freetext_to_partial', 0)}** · "
            f"→ 🟢 (default): **{oa.get('n_freetext_to_green', 0)}**"
        )
        if oa.get("n_genrtn_overrides_no_change", 0):
            st.warning(
                f"⚠️ **{oa['n_genrtn_overrides_no_change']}** bill(s) where surveyor wrote "
                f"\"No change\" but the till applied GENRTN — reclassified as Amber-genrtn."
            )

        top_ft = oa.get("top_freetext_in_green", {})
        if top_ft:
            st.markdown("**Top 20 free-text outcomes that defaulted to 🟢** (consider extending RED/AMBER patterns):")
            ft_df = pd.DataFrame(
                [(k, v) for k, v in top_ft.items()],
                columns=["Outcome (free text)", "Count"],
            )
            st.dataframe(ft_df, width='stretch', hide_index=True)

# ---- Store cut --------------------------------------------------------------
with tab_store:
    st.subheader("Store-wise rollup")
    st.caption("All counts based on customer interactions (survey rows). RAG counts include walk-aways/no-ID rows; bill-level counts only the serial-joined subset.")
    if fsurvey.empty:
        st.info("No survey rows in this filter.")
    else:
        survey_grp = fsurvey.groupby("store", dropna=False)
        bill_grp = fbills.groupby("store", dropna=False) if not fbills.empty else None

        def pct_yes(series: pd.Series) -> float:
            n = len(series)
            return round(100 * series.str.lower().eq("yes").mean(), 1) if n else 0.0

        rows = []
        for store, g in survey_grp:
            n_g = int((g["rag"] == "green").sum())
            n_a = int((g["rag"] == "amber").sum())
            n_r = int((g["rag"] == "red").sum())
            n_ap = int(g["is_amber_partial"].sum())
            n_ag = int(g["is_amber_genrtn"].sum())
            bg = fbills[fbills["store"] == store] if not fbills.empty else fbills
            n_bills = int(bg["bill_id"].notna().sum())
            n_sku = int(bg["has_price_increased_sku"].sum()) if len(bg) else 0
            pi = bg[bg["has_price_increased_sku"]] if len(bg) else bg
            rows.append({
                "store": store,
                "Customer interactions": len(g),
                "Bills joined": n_bills,
                "Bills w/ price-↑ SKU": n_sku,
                f"{RAG_EMOJI['green']} green": n_g,
                f"{RAG_EMOJI['amber']} amber": n_a,
                "↳ partial": n_ap,
                "↳ GENRTN": n_ag,
                f"{RAG_EMOJI['red']} red": n_r,
                "% spoke (all)": pct_yes(g["spoke_price"]),
                "% spoke (price-↑ bills)": pct_yes(pi["spoke_price"]) if len(pi) else 0.0,
                "% non-verbal (all)": pct_yes(g["nonverbal"]),
                "% non-verbal (price-↑ bills)": pct_yes(pi["nonverbal"]) if len(pi) else 0.0,
            })
        # Totals row
        pi_all = fbills[fbills["has_price_increased_sku"]] if len(fbills) else fbills
        rows.append({
            "store": "TOTAL",
            "Customer interactions": len(fsurvey),
            "Bills joined": int(fbills["bill_id"].notna().sum()),
            "Bills w/ price-↑ SKU": int(fbills["has_price_increased_sku"].sum()) if len(fbills) else 0,
            f"{RAG_EMOJI['green']} green": int((fsurvey["rag"] == "green").sum()),
            f"{RAG_EMOJI['amber']} amber": int((fsurvey["rag"] == "amber").sum()),
            "↳ partial": int(fsurvey["is_amber_partial"].sum()),
            "↳ GENRTN": int(fsurvey["is_amber_genrtn"].sum()),
            f"{RAG_EMOJI['red']} red": int((fsurvey["rag"] == "red").sum()),
            "% spoke (all)": pct_yes(fsurvey["spoke_price"]),
            "% spoke (price-↑ bills)": pct_yes(pi_all["spoke_price"]) if len(pi_all) else 0.0,
            "% non-verbal (all)": pct_yes(fsurvey["nonverbal"]),
            "% non-verbal (price-↑ bills)": pct_yes(pi_all["nonverbal"]) if len(pi_all) else 0.0,
        })
        # Set "store" as index so it stays pinned on the left during horizontal scroll.
        st.dataframe(pd.DataFrame(rows).set_index("store"), width='stretch')

# ---- Sentiment cross-tabs ---------------------------------------------------
with tab_sent:
    st.subheader("Sentiment by RAG bucket — Yes/No fields & multi-select reasons")

    scope = st.radio(
        "Scope",
        ["All bills", "Bills WITH price-↑ SKU only", "Side by side (with vs without)"],
        horizontal=True,
    )

    def _scope_frame(df: pd.DataFrame, with_sku: bool | None) -> pd.DataFrame:
        if with_sku is None: return df
        return df[df["has_price_increased_sku"] == with_sku]

    yn_fields = [
        ("staff_informed", "Staff informed"),
        ("spoke_price", "Spoke about price"),
        ("nonverbal", "Non-verbal reaction"),
        ("checked_bill", "Checked bill"),
    ]

    def explode_field(df: pd.DataFrame, field: str) -> pd.Series:
        s = df[field].fillna("").astype(str).str.split(",").explode().str.strip()
        return s[s.ne("")]

    def render_yn_row(df: pd.DataFrame, label: str):
        st.markdown(f"**{label}** — bills = {len(df)}")
        if df.empty: st.info("No bills."); return
        cols = st.columns(len(yn_fields))
        for col, (field, title) in zip(cols, yn_fields):
            with col:
                vals = df[field].str.lower().replace("", "(blank)")
                ct = pd.crosstab(df["rag"], vals, margins=True)
                ct = ct.reindex(index=["green", "amber", "red", "All"]).fillna(0).astype(int)
                col_order = [c for c in ["yes", "no", "(blank)"] if c in ct.columns] + ["All"]
                ct = ct[col_order]
                ct.index = [f"{RAG_EMOJI.get(i,'')} {i}" if i in RAG_EMOJI else i for i in ct.index]
                st.markdown(f"**{title}**")
                st.dataframe(ct, width='stretch')

    def render_multiselect_row(df: pd.DataFrame, field: str, label: str):
        st.markdown(f"**{label}** — bills = {len(df)}")
        if df.empty: st.info("No bills."); return
        mcols = st.columns(2)
        for mcol, rag in zip(mcols, ["green", "amber"]):
            with mcol:
                bucket = df[df["rag"] == rag]
                values = explode_field(bucket, field)
                top = values.value_counts().head(10)
                st.markdown(f"**{RAG_EMOJI[rag]} {rag.title()}** ({len(bucket)} bills)")
                if top.empty: st.write("_(none)_")
                else: st.bar_chart(top)

    def render_sentiment_block(df: pd.DataFrame, label: str):
        """Single-scope view: Yes/No row, then multi-select rows in order."""
        render_yn_row(df, label)
        st.markdown("**Top reasons & reactions (multi-select)** — top 10 per RAG bucket")
        st.markdown("###### What customer said")
        render_multiselect_row(df, "what_said", label)
        st.markdown("###### Reaction type")
        render_multiselect_row(df, "reaction_type", label)

    if scope == "All bills":
        render_sentiment_block(fbills, "All bills")
    elif scope == "Bills WITH price-↑ SKU only":
        render_sentiment_block(_scope_frame(fbills, True), "With price-↑ SKU")
    else:
        df_with = _scope_frame(fbills, True)
        df_without = _scope_frame(fbills, False)

        # Section 1: Yes/No fields — with on top, without below
        st.markdown("### Yes/No fields × RAG bucket")
        render_yn_row(df_with, "With price-↑ SKU")
        st.markdown("")
        render_yn_row(df_without, "Without price-↑ SKU")
        st.markdown("---")

        # Section 2: What customer said
        st.markdown("### What customer said (top 10 per RAG bucket)")
        render_multiselect_row(df_with, "what_said", "With price-↑ SKU")
        st.markdown("")
        render_multiselect_row(df_without, "what_said", "Without price-↑ SKU")
        st.markdown("---")

        # Section 3: Reaction type
        st.markdown("### Reaction type (top 10 per RAG bucket)")
        render_multiselect_row(df_with, "reaction_type", "With price-↑ SKU")
        st.markdown("")
        render_multiselect_row(df_without, "reaction_type", "Without price-↑ SKU")

# ---- Walk-aways (= RED bucket) ----------------------------------------------
with tab_walk:
    st.subheader("Walk-aways by store — 🔴 RED bucket of RAG")
    st.caption(
        "Walk-away = customer left without a bill. Classified purely by the outcome column "
        "(\"Bana hi nahin. Customer chale gaye.\")."
    )
    walk = fsurvey[fsurvey["is_walkaway"]].copy()
    if walk.empty:
        st.info("No walk-aways in this filter.")
    else:
        per_store = walk.groupby("store").agg(
            walkaways=("submission_id", "size"),
            verbal_complaint=("spoke_price", lambda s: int(s.str.lower().eq("yes").sum())),
            nonverbal_reaction=("nonverbal", lambda s: int(s.str.lower().eq("yes").sum())),
        )
        per_store["RED w/ verbal price complaint"] = per_store["verbal_complaint"]
        per_store["RED w/o verbal price complaint"] = per_store["walkaways"] - per_store["verbal_complaint"]
        per_store = per_store.reset_index()
        st.bar_chart(per_store.set_index("store")["walkaways"])
        st.dataframe(per_store, width='stretch')

# ---- Partial-bill retention -------------------------------------------------
with tab_partial:
    st.subheader("Partial-bill cohort: did they keep buying price-↑ SKUs?")
    st.caption(
        "Cohort = patients flagged AMBER-partial on a bill (surveyor said they dropped items). "
        "For each, we check their **4-month pre-go-live** history of price-↑ SKUs across **all "
        "chain stores**, then whether they re-bought the SAME SKUs after the price hike."
    )

    pb_hist = _load_partial_bill_snapshot()
    if pb_hist is None or pb_hist.empty:
        st.warning(
            "`data_snapshots/partial_bill_history.parquet` missing or empty. "
            "Run `python refresh_snapshots.py` locally with VPN to generate it."
        )
    else:
        partial_bills = fsurvey[fsurvey["is_amber_partial"].fillna(False)].copy()
        st.markdown(f"**Partial-bill cohort:** {pb_hist['patient_id'].nunique():,} unique patients · {len(pb_hist):,} purchase rows of price-↑ SKUs across 4-month pre + post.")

        gross = pb_hist[pb_hist["bill_flag"] == "gross"].copy()
        go_live = GO_LIVE_DATE.date()
        gross["period"] = gross["day"].apply(lambda d: "pre" if d < go_live else "post")

        # Per-patient × per-SKU pivot
        per_ps = (
            gross.groupby(["patient_id", "drug_id", "drug_name", "period"])
            ["net_quantity"].sum().unstack("period", fill_value=0).reset_index()
        )
        per_ps["pre"] = per_ps.get("pre", 0)
        per_ps["post"] = per_ps.get("post", 0)
        per_ps["continued"] = (per_ps["pre"] > 0) & (per_ps["post"] > 0)
        per_ps["lapsed"] = (per_ps["pre"] > 0) & (per_ps["post"] <= 0)

        # Per-patient summary
        per_patient = (
            per_ps.groupby("patient_id").agg(
                n_pi_skus_pre=("pre", lambda s: int((s > 0).sum())),
                n_pi_skus_continued=("continued", "sum"),
                n_pi_skus_lapsed=("lapsed", "sum"),
                total_qty_pre=("pre", "sum"),
                total_qty_post=("post", "sum"),
            ).reset_index()
        )
        per_patient["sku_retention_rate"] = (
            per_patient["n_pi_skus_continued"] /
            per_patient["n_pi_skus_pre"].replace(0, pd.NA)
        ).round(3)
        per_patient = per_patient.sort_values("n_pi_skus_pre", ascending=False)

        # Headline metrics
        hc1, hc2, hc3 = st.columns(3)
        hc1.metric("Patients with pre-go-live price-↑ history",
                    int((per_patient["n_pi_skus_pre"] > 0).sum()))
        any_continued = int((per_patient["n_pi_skus_continued"] > 0).sum())
        hc2.metric("Of those, ≥1 SKU continued post", any_continued)
        all_lapsed = int(((per_patient["n_pi_skus_pre"] > 0) & (per_patient["n_pi_skus_continued"] == 0)).sum())
        hc3.metric("All price-↑ SKUs lapsed post", all_lapsed)

        st.markdown("**Per-patient summary**")
        st.dataframe(per_patient, width='stretch', hide_index=True)

        # Per-SKU retention rollup
        st.markdown("**Per-SKU retention (within the partial-bill cohort)**")
        per_sku = (
            per_ps.groupby(["drug_id", "drug_name"]).agg(
                n_patients_pre=("pre", lambda s: int((s > 0).sum())),
                n_patients_continued=("continued", "sum"),
                total_qty_pre=("pre", "sum"),
                total_qty_post=("post", "sum"),
            ).reset_index()
        )
        per_sku = per_sku[per_sku["n_patients_pre"] > 0]
        per_sku["sku_retention_rate"] = (
            per_sku["n_patients_continued"] / per_sku["n_patients_pre"]
        ).round(3)
        per_sku = per_sku.sort_values("n_patients_pre", ascending=False)
        st.dataframe(
            per_sku, width='stretch', hide_index=True,
            column_config={"drug_name": st.column_config.Column(pinned=True)},
        )

        st.download_button(
            "📥 Download partial-bill cohort retention (CSV)",
            data=per_patient.to_csv(index=False).encode("utf-8"),
            file_name="partial_bill_retention_per_patient.csv",
            mime="text/csv",
        )

# ---- Control comparison -----------------------------------------------------
with tab_ctrl:
    st.subheader("Bills WITH vs WITHOUT a price-increased SKU — RAG matrix")
    st.caption(
        "Headline question: do customers buying price-↑ SKUs react differently than the control? "
        "Each row sums to 100% across green/amber/red."
    )
    if fbills.empty:
        st.info("No joined bills in this filter.")
    else:
        groups = {
            "WITH price-↑ SKU": fbills[fbills["has_price_increased_sku"]],
            "WITHOUT price-↑ SKU": fbills[~fbills["has_price_increased_sku"]],
        }
        rows = []
        for label, g in groups.items():
            n = len(g)
            ng = int((g["rag"] == "green").sum())
            na = int((g["rag"] == "amber").sum())
            nr = int((g["rag"] == "red").sum())
            ap = int(g["is_amber_partial"].sum())
            ag = int(g["is_amber_genrtn"].sum())
            avg_rev = round(g["revenue"].mean(), 2) if n else 0.0
            f = (lambda x: f"{x} ({100*x/n:.1f}%)" if n else "—")
            rows.append({
                "Group": label,
                "Bills": n,
                f"{RAG_EMOJI['green']} Green": f(ng),
                f"{RAG_EMOJI['amber']} Amber": f(na),
                "  ↳ partial": ap,
                "  ↳ GENRTN": ag,
                f"{RAG_EMOJI['red']} Red": f(nr),
                "% spoke about price": round(100 * g["spoke_price"].str.lower().eq("yes").mean(), 1) if n else 0.0,
                "% non-verbal reaction": round(100 * g["nonverbal"].str.lower().eq("yes").mean(), 1) if n else 0.0,
                "Avg revenue": avg_rev,
            })
        st.dataframe(pd.DataFrame(rows), width='stretch', hide_index=True)

# ---- Drill-down -------------------------------------------------------------
with tab_drill:
    st.subheader("Bill-level drill-down")
    rag_filter = st.multiselect(
        "Filter by RAG",
        options=["green", "amber", "red"],
        default=["green", "amber", "red"],
        format_func=lambda r: f"{RAG_EMOJI[r]} {r.title()}",
    )
    drill_src = fbills[fbills["rag"].isin(rag_filter)].copy()
    show_cols = [
        "store", "serial_norm", "bill_id", "submitted_at", "bill_date",
        "rag", "is_amber_partial", "is_amber_genrtn",
        "has_price_increased_sku", "n_lines", "n_price_increase_lines",
        "revenue", "used_genrtn", "promo_code_any",
        "staff_informed", "spoke_price", "nonverbal",
        "outcome", "what_said", "reaction_type", "notes",
    ]
    drill = drill_src[show_cols].copy()
    st.dataframe(drill, width='stretch', height=500)
    st.download_button(
        "Download CSV",
        data=drill.to_csv(index=False).encode("utf-8"),
        file_name="price_change_drilldown.csv",
        mime="text/csv",
    )

# -----------------------------------------------------------------------------
# Footer — open items
# -----------------------------------------------------------------------------
st.markdown("---")
st.caption(
    "Open items to confirm with the team: "
    "(1) survey **Khar** is mapped to sales **Khar West** — confirm vs. Kharghar / Kharghar Sector 12. "
    "(2) survey **Goregaon** (with serial prefix `GORE`) is mapped to sales **Goregaon**, not "
    "**Goregaon West S.V. Road** — confirm. "
    "(3) `GENRTN` count above appears low — confirm the actual coupon code used in stores."
)

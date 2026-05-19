"""Quant Insight — Pilot vs Non-Pilot store A/B test.

Pilot stores received the 2026-05-06 price increase. Non-pilot = paired
control. Page is organised into 4 tabs:
  1. Summary       — All Pilot vs All Paired Non-Pilot vs All Other Stores
  2. Store vs Store — per-pair pilot ↔ non-pilot
  3. Elasticity    — SKU-level price sensitivity
  4. Drill-down    — bill-level table + CSV
"""

from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

from data import (
    ASSORTMENT_ETHICAL,
    ASSORTMENT_GENERIC,
    PAIRED_NON_PILOT_STORES,
    PILOT_STORES,
    QUANT_POST_END,
    QUANT_POST_START,
    QUANT_PRE_END,
    QUANT_PRE_START,
    STORE_PAIRS,
    _load_patients_snapshot,
    _load_sales_snapshot,
    _load_store_totals_snapshot,
    aggregate_totals,
    build_pair_table,
    compute_elasticity,
    compute_kpis,
    compute_repeat_churn,
    fetch_quant_sales,
    load_skus,
)

# ----- KPI catalogue ----------------------------------------------------------

KPI_DIRECTION = {
    # ↑ good
    "revenue": +1, "gm": +1, "gm_pct": +1, "aov": +1, "aom": +1, "acm": +1,
    "frequency": +1, "bill_count": +1, "unique_customers": +1,
    "total_quantity": +1, "items_per_bill": +1, "revenue_per_patient": +1,
    "repeat_patient_count": +1, "new_patient_count": +1,
    "repeat_rate": +1,
    # ↓ good
    "generic_mix_units_pct": -1, "generic_mix_revenue_pct": -1,
    "promo_usage_rate": -1, "promo_share_revenue": -1,
    "return_rate": -1, "churn_rate": -1,
}

KPI_LABELS = {
    "revenue": "Revenue (₹)",
    "gm": "Gross Margin (₹)",
    "gm_pct": "GM %",
    "aov": "Avg Order Value (₹/bill)",
    "aom": "Avg Order Margin (₹/bill)",
    "acm": "Avg Customer Margin (₹/patient)",
    "bill_count": "Bills",
    "unique_customers": "Patients",
    "frequency": "Frequency (bills/patient)",
    "total_quantity": "Total units",
    "items_per_bill": "Items per bill",
    "revenue_per_patient": "Revenue per patient (₹)",
    "repeat_patient_count": "Repeat patients (≥2 bills in window)",
    "new_patient_count": "New patients (first-ever in chain)",
    "generic_mix_units_pct": "Generic mix % (units)",
    "generic_mix_revenue_pct": "Generic mix % (revenue)",
    "promo_usage_rate": "GENRTN promo usage rate",
    "promo_share_revenue": "GENRTN discount share of revenue",
    "return_rate": "Return rate",
    "repeat_rate": "8-day repeat rate",
    "churn_rate": "8-day churn rate",
}

# KPI groupings for the section dividers
KPI_GROUPS = [
    ("Volume",       ["bill_count", "unique_customers", "total_quantity",
                       "repeat_patient_count", "new_patient_count"]),
    ("Revenue & Margin", ["revenue", "gm", "gm_pct"]),
    ("Per bill / per patient", ["aov", "aom", "acm", "items_per_bill",
                                 "revenue_per_patient", "frequency"]),
    ("Mix",          ["generic_mix_units_pct", "generic_mix_revenue_pct"]),
    ("Promo / Returns", ["promo_usage_rate", "promo_share_revenue", "return_rate"]),
]

PILOT_COLOR = "#2E86C1"
NPP_COLOR = "#7F8C8D"
OTHER_COLOR = "#BDC3C7"
GOOD_COLOR = "#27AE60"
BAD_COLOR = "#C0392B"


# ----- formatters -------------------------------------------------------------

def fmt_value(kpi: str, v) -> str:
    if v is None or pd.isna(v): return "—"
    if kpi.endswith("_pct") or kpi.endswith("_rate"):
        return f"{v*100:.1f}%"
    if kpi in ("revenue", "gm", "aov", "aom", "acm", "revenue_per_patient"):
        return f"₹{v:,.0f}"
    if kpi in ("bill_count", "unique_customers", "total_quantity",
                "repeat_patient_count", "new_patient_count"):
        return f"{int(v):,}"
    return f"{v:.2f}"


def fmt_delta(v) -> str:
    if v is None or pd.isna(v): return "—"
    return f"{v:+.1f}%"


def color_did(did, kpi: str) -> str:
    if did is None or pd.isna(did): return "color: #95a5a6"
    direction = KPI_DIRECTION.get(kpi, 0)
    if direction == 0: return ""
    good = (did > 0 and direction > 0) or (did < 0 and direction < 0)
    return f"color: {GOOD_COLOR}; font-weight: 600" if good else f"color: {BAD_COLOR}; font-weight: 600"


# =============================================================================
# Page setup
# =============================================================================

st.title("Quant Insight — Pilot vs Non-Pilot")
st.caption(
    "**Pilot** = 6 Mumbai stores where the price increase was applied on 2026-05-06. "
    "**Non-Pilot** = paired control stores (unchanged). **DiD** = Pilot Δ% − Non-Pilot Δ%."
)

skus = load_skus()
sku_ids = set(skus["drug_id"].astype(int))
patient_first_seen = _load_patients_snapshot()
store_totals = _load_store_totals_snapshot()
sales_snapshot = _load_sales_snapshot()

if sales_snapshot is None:
    st.error("`data_snapshots/sales.parquet` missing. Run `python refresh_snapshots.py` locally with VPN.")
    st.stop()

# ----- Sidebar ----------------------------------------------------------------
st.sidebar.header("Quant filters")
c1, c2 = st.sidebar.columns(2)
pre_from = c1.date_input("Pre from", value=QUANT_PRE_START)
pre_to = c2.date_input("Pre to", value=QUANT_PRE_END)
c3, c4 = st.sidebar.columns(2)
post_from = c3.date_input("Post from", value=QUANT_POST_START)
post_to = c4.date_input("Post to", value=QUANT_POST_END)

st.sidebar.caption(f"Pre  : {pre_from} → {pre_to}\nPost : {post_from} → {post_to}")

# Bill-set toggle (page-level)
st.markdown("### Bill set")
bill_set_label = st.radio(
    "Which bills to analyse?",
    ["Price-↑ bills", "Non-price-↑ bills", "All bills"],
    horizontal=True,
    label_visibility="collapsed",
    help=(
        "**Price-↑ bills**: bills containing at least one of the 346 price-increased SKUs (treatment universe). "
        "**Non-price-↑ bills**: bills with ZERO price-↑ SKUs (control bills). "
        "**All bills**: no bill-level filter."
    ),
)
bill_filter = {"Price-↑ bills": "with_pi", "Non-price-↑ bills": "without_pi", "All bills": "all"}[bill_set_label]


# ----- Per-store data loaders -------------------------------------------------

@st.cache_data(ttl=1800, show_spinner=False)
def load_store(store: str, date_from: str, date_to: str) -> pd.DataFrame:
    """Line-level rows for one store between two dates (snapshot-first)."""
    return fetch_quant_sales((store,), date_from, date_to)


def kpis_for_store_list(stores: tuple[str, ...], date_from, date_to) -> dict:
    """Compute KPIs for a UNION of stores by concatenating line-level data."""
    dfs = [load_store(s, date_from.isoformat(), date_to.isoformat()) for s in stores]
    df = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
    return compute_kpis(df, sku_ids, patient_first_seen, bill_filter)


# ============================================================================
# Tabs
# ============================================================================

tab_summary, tab_pair, tab_elast, tab_drill = st.tabs([
    "🏪 Summary",
    "🔀 Store vs Store",
    "📈 Elasticity",
    "🔍 Drill-down",
])


# ----- Tab 1: Summary --------------------------------------------------------
with tab_summary:
    st.subheader("Pilot vs Paired Non-Pilot vs All Other Stores")
    st.caption(
        "**All Pilot** (6 stores) · **Paired Non-Pilot** (6 paired control stores) · "
        "**All-Other Non-Pilot** (~195 chain stores not in the experiment, from per-store-per-day aggregates). "
        "DiD = Pilot Δ% − comparator Δ%."
    )

    # Compute KPIs for the 3 groups × pre/post.
    with st.spinner("Computing group KPIs…"):
        pilot_pre = kpis_for_store_list(PILOT_STORES, pre_from, pre_to)
        pilot_post = kpis_for_store_list(PILOT_STORES, post_from, post_to)
        npp_pre = kpis_for_store_list(PAIRED_NON_PILOT_STORES, pre_from, pre_to)
        npp_post = kpis_for_store_list(PAIRED_NON_PILOT_STORES, post_from, post_to)

        if store_totals is not None:
            all_stores_in_chain = set(store_totals["store_name"].unique())
            pilot_set = set(PILOT_STORES)
            other_stores = tuple(sorted(all_stores_in_chain - pilot_set))
            other_pre = aggregate_totals(store_totals, other_stores, pre_from, pre_to, bill_filter)
            other_post = aggregate_totals(store_totals, other_stores, post_from, post_to, bill_filter)
        else:
            other_pre, other_post = {}, {}

    def delta_pct(post, pre):
        return ((post - pre) / pre * 100) if pre else None

    # Hero KPIs row
    st.markdown("---")
    hero_kpis = [("revenue", "Revenue"), ("gm", "Gross Margin"),
                 ("aov", "Avg Order Value"), ("unique_customers", "Patients")]
    hcols = st.columns(len(hero_kpis))
    for col, (k, label) in zip(hcols, hero_kpis):
        pilot_d = delta_pct(pilot_post.get(k, 0), pilot_pre.get(k, 0))
        npp_d = delta_pct(npp_post.get(k, 0), npp_pre.get(k, 0))
        did = (pilot_d - npp_d) if (pilot_d is not None and npp_d is not None) else None
        col.metric(
            label,
            fmt_value(k, pilot_post.get(k)),
            f"DiD vs paired: {fmt_delta(did)}",
        )
    st.caption(f"Pilot KPI value shown is the post-window total. Delta = DiD against the 6 paired non-pilot stores.")

    # Full comparison table per group of KPIs
    st.markdown("---")
    for group_name, kpis in KPI_GROUPS:
        st.markdown(f"### {group_name}")
        rows = []
        for k in kpis:
            ppre, ppost = pilot_pre.get(k), pilot_post.get(k)
            npre, npost = npp_pre.get(k), npp_post.get(k)
            opre, opost = other_pre.get(k), other_post.get(k)
            p_delta = delta_pct(ppost, ppre)
            n_delta = delta_pct(npost, npre)
            o_delta = delta_pct(opost, opre) if opost is not None else None
            did_paired = (p_delta - n_delta) if (p_delta is not None and n_delta is not None) else None
            did_other = (p_delta - o_delta) if (p_delta is not None and o_delta is not None) else None
            rows.append({
                "KPI": KPI_LABELS.get(k, k),
                "Pilot Pre": fmt_value(k, ppre),
                "Pilot Post": fmt_value(k, ppost),
                "Pilot Δ%": fmt_delta(p_delta),
                "Non-Pilot Pre": fmt_value(k, npre),
                "Non-Pilot Post": fmt_value(k, npost),
                "Non-Pilot Δ%": fmt_delta(n_delta),
                "All-Other Δ%": fmt_delta(o_delta),
                "DiD vs Paired": fmt_delta(did_paired),
                "DiD vs All-Other": fmt_delta(did_other),
                "All-Other Pre": fmt_value(k, opre),
                "All-Other Post": fmt_value(k, opost),
                "_kpi": k,
                "_did_paired": did_paired,
                "_did_other": did_other,
            })
        df_disp = pd.DataFrame(rows)
        kpi_series = df_disp["_kpi"].reset_index(drop=True)
        dp_series = df_disp["_did_paired"].reset_index(drop=True)
        do_series = df_disp["_did_other"].reset_index(drop=True)
        display = df_disp.drop(columns=["_kpi", "_did_paired", "_did_other"]).set_index("KPI")
        did_paired_pos = display.columns.get_loc("DiD vs Paired")
        did_other_pos = display.columns.get_loc("DiD vs All-Other")

        def styler(row, _dp=dp_series, _do=do_series, _ks=kpi_series, _dpp=did_paired_pos, _dop=did_other_pos):
            idx = row.name  # this is the index label (KPI name) when using set_index
            i = display.index.get_loc(idx) if idx in display.index else 0
            styles = [""] * len(row)
            styles[_dpp] = color_did(_dp.iloc[i], _ks.iloc[i])
            styles[_dop] = color_did(_do.iloc[i], _ks.iloc[i])
            return styles

        st.dataframe(display.style.apply(styler, axis=1), width='stretch')

    # Caveats for the All-Other column
    if store_totals is None:
        st.warning("`data_snapshots/store_totals.parquet` missing — All-Other column unavailable.")
    else:
        st.caption(
            "⚠️ All-Other Δ% is computed from per-store-per-day aggregates (no patient-level cross-store deduplication). "
            "Patient counts in that column may double-count customers who visit multiple stores. "
            "`new_patient_count` is unavailable for All-Other (requires line-level data; shows 0)."
        )


# ----- Tab 2: Store vs Store -------------------------------------------------
with tab_pair:
    st.subheader("Pair-by-pair: Pilot ↔ Non-Pilot")

    with st.spinner("Loading pair data…"):
        pair_data = {}
        for p in STORE_PAIRS:
            pair_data[p["pair"]] = {
                "pilot_pre":  load_store(p["pilot"], pre_from.isoformat(), pre_to.isoformat()),
                "pilot_post": load_store(p["pilot"], post_from.isoformat(), post_to.isoformat()),
                "npp_pre":    load_store(p["non_pilot"], pre_from.isoformat(), pre_to.isoformat()),
                "npp_post":   load_store(p["non_pilot"], post_from.isoformat(), post_to.isoformat()),
            }

    all_rows = []
    for p in STORE_PAIRS:
        d = pair_data[p["pair"]]
        pt = build_pair_table(d["pilot_pre"], d["pilot_post"], d["npp_pre"], d["npp_post"],
                              sku_ids, patient_first_seen, bill_filter)
        pt["pair"] = p["pair"]
        pt["pair_label"] = f"P{p['pair']}: {p['pilot']} ↔ {p['non_pilot']}"
        all_rows.append(pt)
    combined = pd.concat(all_rows, ignore_index=True)

    # Add repeat / churn rows per pair
    rc_rows = []
    for p in STORE_PAIRS:
        d = pair_data[p["pair"]]
        t_rc = compute_repeat_churn(d["pilot_pre"], d["pilot_post"], sku_ids if bill_filter == "with_pi" else None)
        c_rc = compute_repeat_churn(d["npp_pre"], d["npp_post"], sku_ids if bill_filter == "with_pi" else None)
        label = f"P{p['pair']}: {p['pilot']} ↔ {p['non_pilot']}"
        for k in ("repeat_rate", "churn_rate"):
            rc_rows.append({
                "kpi": k, "pair": p["pair"], "pair_label": label,
                "test_pre": None, "test_post": t_rc[k], "test_delta_pct": None,
                "ctrl_pre": None, "ctrl_post": c_rc[k], "ctrl_delta_pct": None,
                "did": (t_rc[k] - c_rc[k]) * 100,
            })
    combined = pd.concat([combined, pd.DataFrame(rc_rows)], ignore_index=True)

    # Per-group rendering for the pair view too
    for group_name, group_kpis in KPI_GROUPS + [("Guardrails", ["return_rate", "repeat_rate", "churn_rate"])]:
        st.markdown(f"#### {group_name}")
        sub = combined[combined["kpi"].isin(group_kpis)].copy().reset_index(drop=True)
        if sub.empty: continue
        sub["KPI"] = sub["kpi"].map(KPI_LABELS)
        sub["Pilot Pre"] = sub.apply(lambda r: fmt_value(r["kpi"], r["test_pre"]), axis=1)
        sub["Pilot Post"] = sub.apply(lambda r: fmt_value(r["kpi"], r["test_post"]), axis=1)
        sub["Pilot Δ%"] = sub["test_delta_pct"].apply(fmt_delta)
        sub["Non-Pilot Pre"] = sub.apply(lambda r: fmt_value(r["kpi"], r["ctrl_pre"]), axis=1)
        sub["Non-Pilot Post"] = sub.apply(lambda r: fmt_value(r["kpi"], r["ctrl_post"]), axis=1)
        sub["Non-Pilot Δ%"] = sub["ctrl_delta_pct"].apply(fmt_delta)
        sub["DiD"] = sub["did"].apply(fmt_delta)
        cols = ["KPI", "pair_label", "Pilot Pre", "Pilot Post", "Pilot Δ%",
                "Non-Pilot Pre", "Non-Pilot Post", "Non-Pilot Δ%", "DiD"]
        disp = sub[cols].rename(columns={"pair_label": "Pair"})
        kpi_series = sub["kpi"]
        did_series = sub["did"]
        did_pos = disp.columns.get_loc("DiD")
        def styler(row, _ds=did_series, _ks=kpi_series, _dp=did_pos):
            i = row.name
            styles = [""] * len(row)
            styles[_dp] = color_did(_ds.iloc[i], _ks.iloc[i])
            return styles
        st.dataframe(
            disp.style.apply(styler, axis=1),
            width='stretch',
            hide_index=True,
            column_config={"KPI": st.column_config.Column(pinned=True)},
        )


# ----- Tab 3: Elasticity -----------------------------------------------------
with tab_elast:
    st.subheader("SKU-level elasticity (Pilot stores)")
    st.caption(
        "For each of the 346 price-↑ SKUs sold across PILOT stores, compare pre- and post-window units & realised price. "
        "Negative own-price elasticity = customers buy fewer when price goes up."
    )

    pilot_pre_all = pd.concat(
        [load_store(s, pre_from.isoformat(), pre_to.isoformat()) for s in PILOT_STORES],
        ignore_index=True,
    )
    pilot_post_all = pd.concat(
        [load_store(s, post_from.isoformat(), post_to.isoformat()) for s in PILOT_STORES],
        ignore_index=True,
    )

    elast = compute_elasticity(pilot_pre_all, pilot_post_all, sku_ids)
    elast_with_data = elast[(elast["pre_units"] > 0) | (elast["post_units"] > 0)].copy()
    elast_with_data = elast_with_data.merge(
        skus[["drug_id", "drug_name", "old_price", "new_price"]],
        on="drug_id", how="left",
    )
    st.markdown(f"**Universe:** {len(sku_ids)} price-↑ SKUs · with sales in either window: **{len(elast_with_data)}**")

    display_e = elast_with_data.dropna(subset=["own_elasticity"]).copy()
    display_e = display_e[display_e["own_elasticity"].between(-50, 50)]
    display_e = display_e.sort_values("own_elasticity").head(20)
    st.markdown("**Top 20 most price-sensitive SKUs (most-negative own-price elasticity)**")
    st.dataframe(
        display_e[["drug_id", "drug_name", "old_price", "new_price",
                    "pre_units", "post_units", "pre_avg_price", "post_avg_price",
                    "units_delta_pct", "price_delta_pct", "own_elasticity"]],
        width='stretch', hide_index=True,
    )

    # Cross-price elasticity at the chain level (pilot stores)
    gen_pre = pilot_pre_all[pilot_pre_all["bill_flag"] == "gross"]
    gen_pre_g = gen_pre[gen_pre["assortment_classification_id"].isin(ASSORTMENT_GENERIC)]
    gen_post = pilot_post_all[pilot_post_all["bill_flag"] == "gross"]
    gen_post_g = gen_post[gen_post["assortment_classification_id"].isin(ASSORTMENT_GENERIC)]
    eth_pre_u = float(gen_pre[gen_pre["assortment_classification_id"].isin(ASSORTMENT_ETHICAL)]["net_quantity"].sum())
    eth_post_u = float(gen_post[gen_post["assortment_classification_id"].isin(ASSORTMENT_ETHICAL)]["net_quantity"].sum())
    g_pre_u = float(gen_pre_g["net_quantity"].sum())
    g_post_u = float(gen_post_g["net_quantity"].sum())
    g_pre_p = (float(gen_pre_g["revenue_value"].sum()) / g_pre_u) if g_pre_u else 0
    g_post_p = (float(gen_post_g["revenue_value"].sum()) / g_post_u) if g_post_u else 0
    eth_d = ((eth_post_u - eth_pre_u) / eth_pre_u * 100) if eth_pre_u else None
    gp_d = ((g_post_p - g_pre_p) / g_pre_p * 100) if g_pre_p else None
    cross = (eth_d / gp_d) if (eth_d is not None and gp_d) else None

    cc1, cc2, cc3 = st.columns(3)
    cc1.metric("%Δ Ethical units (Pilot)", fmt_delta(eth_d))
    cc2.metric("%Δ Generic avg price (Pilot)", fmt_delta(gp_d))
    cc3.metric("Cross-price elasticity (Gen→Eth)",
               f"{cross:+.2f}" if cross is not None and not pd.isna(cross) else "—",
               help="Positive = customers substituted toward ethical when generic price rose.")


# ----- Tab 4: Drill-down -----------------------------------------------------
with tab_drill:
    st.subheader("Bill-level drill-down")
    pair_labels = [f"P{p['pair']} · {p['pilot']} (pilot) vs {p['non_pilot']}" for p in STORE_PAIRS]
    drill_pair_label = st.selectbox("Pick a pair", options=pair_labels)
    dp = next(p for p, lbl in zip(STORE_PAIRS, pair_labels) if lbl == drill_pair_label)
    side = st.radio("Side", ["Pilot", "Non-Pilot"], horizontal=True)
    window = st.radio("Window", ["Pre", "Post"], horizontal=True, index=1)
    store = dp["pilot"] if side == "Pilot" else dp["non_pilot"]
    d_from = pre_from if window == "Pre" else post_from
    d_to = pre_to if window == "Pre" else post_to

    df_view = load_store(store, d_from.isoformat(), d_to.isoformat())
    if df_view.empty:
        st.info("No rows.")
    else:
        bill_agg = df_view[df_view["bill_flag"] == "gross"].groupby("bill_id").agg(
            store=("store_name", "first"),
            date=("created_at", "min"),
            n_lines=("drug_id", "size"),
            units=("net_quantity", "sum"),
            revenue=("revenue_value", "sum"),
            promo_discount=("promo_discount", "sum"),
            cogs=("purchase_rate", lambda s: float((df_view.loc[s.index, "net_quantity"] * s).sum())),
            promo_codes=("promo_code", lambda s: ",".join(sorted({x for x in s if x}))),
            patient_id=("patient_id", "first"),
        ).reset_index()
        bill_agg["gm"] = bill_agg["revenue"] - bill_agg["cogs"] - bill_agg["promo_discount"]
        bill_agg["used_genrtn"] = bill_agg["promo_codes"].str.upper().str.contains("GENRTN", na=False)
        st.dataframe(bill_agg, width='stretch', height=400)
        st.download_button(
            "Download CSV",
            data=bill_agg.to_csv(index=False).encode("utf-8"),
            file_name=f"quant_drilldown_p{dp['pair']}_{side}_{window}.csv",
            mime="text/csv",
        )

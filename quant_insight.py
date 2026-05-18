"""Quant Insight page — store-pair A/B test (test vs control, pre vs post).

Reads only Redshift sales (no survey CSV). Identifies price-↑ SKUs from the
SKU CSV. Tells the story in seven sections:
  0. Hero — chain-wide DiD KPIs
  1. Setup — pair parity check
  2. Headline — Revenue & Margin
  3. Generic Mix (CORE)
  4. Customer Behaviour
  5. Guardrails
  6. SKU-level elasticity
  7. Drill-down
"""

from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

from data import (
    ASSORTMENT_ETHICAL,
    ASSORTMENT_GENERIC,
    QUANT_BASELINE_END,
    QUANT_BASELINE_START,
    QUANT_POST_END,
    QUANT_POST_START,
    QUANT_PRE_END,
    QUANT_PRE_START,
    STORE_PAIRS,
    build_pair_table,
    compute_elasticity,
    compute_kpis,
    compute_repeat_churn,
    fetch_quant_sales,
    load_skus,
)

# ----- helpers ----------------------------------------------------------------

# KPI direction → green when DiD has this sign. None = neutral.
KPI_DIRECTION = {
    "revenue": +1,
    "gm": +1,
    "gm_pct": +1,
    "aov": +1,
    "aom": +1,
    "acm": +1,
    "frequency": +1,
    "bill_count": +1,
    "unique_customers": +1,
    "generic_mix_units_pct": -1,   # ↓ in test = substitution to ethical (the predicted side-effect)
    "generic_mix_revenue_pct": -1,
    "promo_usage_rate": -1,        # ↑ promo usage = customer resistance
    "promo_share_revenue": -1,
    "return_rate": -1,             # ↑ returns = bad
    "repeat_rate": +1,
    "churn_rate": -1,
}

KPI_LABELS = {
    "revenue": "Revenue (₹)",
    "gm": "Gross Margin (₹)",
    "gm_pct": "GM %",
    "aov": "Avg Order Value (₹/bill)",
    "aom": "Avg Order Margin (₹/bill)",
    "acm": "Avg Customer Margin (₹/customer)",
    "bill_count": "Bills",
    "unique_customers": "Unique customers",
    "frequency": "Frequency (bills/customer)",
    "generic_mix_units_pct": "Generic mix % (units)",
    "generic_mix_revenue_pct": "Generic mix % (revenue)",
    "promo_usage_rate": "GENRTN promo usage rate",
    "promo_share_revenue": "GENRTN discount share of revenue",
    "return_rate": "Return rate",
    "repeat_rate": "8-day repeat rate",
    "churn_rate": "8-day churn rate",
}


def fmt_value(kpi: str, v: float) -> str:
    if v is None or pd.isna(v): return "—"
    if kpi.endswith("_pct") or kpi.endswith("_rate"):
        return f"{v*100:.1f}%"
    if kpi in ("revenue", "gm", "aom", "acm"):
        return f"₹{v:,.0f}"
    if kpi in ("bill_count", "unique_customers"):
        return f"{int(v):,}"
    return f"{v:.2f}"


def fmt_delta(v: float) -> str:
    if v is None or pd.isna(v): return "—"
    return f"{v:+.1f}%"


def color_did(did: float, kpi: str) -> str:
    """Return a CSS colour for the DiD cell based on KPI direction."""
    if did is None or pd.isna(did): return "color: #7f8c8d"
    direction = KPI_DIRECTION.get(kpi, 0)
    if direction == 0: return ""
    good = (did > 0 and direction > 0) or (did < 0 and direction < 0)
    return "color: #27ae60; font-weight: 600" if good else "color: #c0392b; font-weight: 600"


def render_kpi_table(rows: list[dict], kpis: list[str], title: str | None = None):
    """Render one consolidated table per pair × KPI with DiD highlighted."""
    if title: st.markdown(f"#### {title}")
    df_disp = pd.DataFrame(rows)
    df_disp = df_disp[df_disp["kpi"].isin(kpis)].reset_index(drop=True)
    if df_disp.empty:
        st.info("No data."); return
    df_disp["KPI"] = df_disp["kpi"].map(KPI_LABELS)
    df_disp["Test Pre"] = df_disp.apply(lambda r: fmt_value(r["kpi"], r["test_pre"]), axis=1)
    df_disp["Test Post"] = df_disp.apply(lambda r: fmt_value(r["kpi"], r["test_post"]), axis=1)
    df_disp["Test Δ%"] = df_disp["test_delta_pct"].apply(fmt_delta)
    df_disp["Ctrl Pre"] = df_disp.apply(lambda r: fmt_value(r["kpi"], r["ctrl_pre"]), axis=1)
    df_disp["Ctrl Post"] = df_disp.apply(lambda r: fmt_value(r["kpi"], r["ctrl_post"]), axis=1)
    df_disp["Ctrl Δ%"] = df_disp["ctrl_delta_pct"].apply(fmt_delta)
    df_disp["DiD"] = df_disp["did"].apply(fmt_delta)

    show_cols = ["pair_label", "KPI", "Test Pre", "Test Post", "Test Δ%",
                 "Ctrl Pre", "Ctrl Post", "Ctrl Δ%", "DiD"]
    out = df_disp[show_cols].rename(columns={"pair_label": "Pair"}).reset_index(drop=True)
    kpi_series = df_disp["kpi"].reset_index(drop=True)
    did_series = df_disp["did"].reset_index(drop=True)

    def styler(row):
        idx = row.name
        styles = [""] * len(row)
        styles[show_cols.index("DiD")] = color_did(did_series.iloc[idx], kpi_series.iloc[idx])
        return styles

    st.dataframe(out.style.apply(styler, axis=1), width='stretch', hide_index=True)


@st.cache_data(ttl=1800, show_spinner=False)
def load_pair_data(pair: dict, pre_from: str, pre_to: str,
                   post_from: str, post_to: str) -> dict:
    """Fetch all 4 quadrants for a pair. Cached by pair-test name + dates."""
    return {
        "test_pre": fetch_quant_sales((pair["test"],), pre_from, pre_to),
        "test_post": fetch_quant_sales((pair["test"],), post_from, post_to),
        "ctrl_pre": fetch_quant_sales((pair["control"],), pre_from, pre_to),
        "ctrl_post": fetch_quant_sales((pair["control"],), post_from, post_to),
    }


# =============================================================================
# Page
# =============================================================================

st.title("Quant Insight — Store-Pair A/B Test")
st.caption(
    "6 Mumbai store pairs were chosen for pre-treatment similarity. **Set A (Test) received the price increase** on 346 SKUs from 2026-05-06; **Set B (Control)** was unchanged. "
    "**All KPIs in Sections 2–5 are scoped to bills containing at least one price-↑ SKU** (the treatment universe). "
    "Sections 1 (parity) and 6 (elasticity) use full or SKU-only data as appropriate. "
    "All windows are symmetric 8 days. Headline number per row is **DiD** (Difference-in-Differences = Test Δ% − Control Δ%)."
)

# Sidebar
st.sidebar.header("Quant filters")
pair_labels = [f"P{p['pair']} · {p['test']} vs {p['control']}" for p in STORE_PAIRS]
selected_labels = st.sidebar.multiselect("Pairs", options=pair_labels, default=pair_labels)
selected_pairs = [p for p, lbl in zip(STORE_PAIRS, pair_labels) if lbl in selected_labels]

c1, c2 = st.sidebar.columns(2)
pre_from = c1.date_input("Pre from", value=QUANT_PRE_START)
pre_to = c2.date_input("Pre to", value=QUANT_PRE_END)
c3, c4 = st.sidebar.columns(2)
post_from = c3.date_input("Post from", value=QUANT_POST_START)
post_to = c4.date_input("Post to", value=QUANT_POST_END)

st.sidebar.caption(f"Pre  : {pre_from} → {pre_to}\nPost : {post_from} → {post_to}")

if not selected_pairs:
    st.warning("Select at least one pair from the sidebar.")
    st.stop()

# Pre-load all pair data
with st.spinner("Loading pair data…"):
    pair_data = {p["pair"]: load_pair_data(
        p, pre_from.isoformat(), pre_to.isoformat(),
        post_from.isoformat(), post_to.isoformat()
    ) for p in selected_pairs}

# Load price-↑ SKU set once — all KPIs are scoped to bills containing ≥1 of these.
skus = load_skus()
sku_ids = set(skus["drug_id"].astype(int))

# Build the pair-KPI long table once for re-use
all_rows = []
for p in selected_pairs:
    d = pair_data[p["pair"]]
    pt = build_pair_table(d["test_pre"], d["test_post"], d["ctrl_pre"], d["ctrl_post"], sku_ids)
    pt["pair"] = p["pair"]
    pt["pair_label"] = f"P{p['pair']}: {p['test']} ↔ {p['control']}"
    all_rows.append(pt)
combined = pd.concat(all_rows, ignore_index=True)

# Add repeat / churn rows per pair (also scoped to customers of price-↑ SKU bills)
rc_rows = []
for p in selected_pairs:
    d = pair_data[p["pair"]]
    t_rc = compute_repeat_churn(d["test_pre"], d["test_post"], sku_ids)
    c_rc = compute_repeat_churn(d["ctrl_pre"], d["ctrl_post"], sku_ids)
    label = f"P{p['pair']}: {p['test']} ↔ {p['control']}"
    for k in ("repeat_rate", "churn_rate"):
        rc_rows.append({
            "kpi": k, "pair": p["pair"], "pair_label": label,
            "test_pre": None, "test_post": t_rc[k], "test_delta_pct": None,
            "ctrl_pre": None, "ctrl_post": c_rc[k], "ctrl_delta_pct": None,
            "did": (t_rc[k] - c_rc[k]) * 100,
        })
combined = pd.concat([combined, pd.DataFrame(rc_rows)], ignore_index=True)

# -----------------------------------------------------------------------------
# Hero — chain-wide DiD on the 4 most important KPIs
# -----------------------------------------------------------------------------

def chain_did(kpi: str) -> float | None:
    sub = combined[combined["kpi"] == kpi]
    return sub["did"].dropna().mean() if not sub["did"].dropna().empty else None

st.markdown("### Headline — chain-wide DiD across selected pairs")
hcols = st.columns(4)
hero_kpis = [("revenue", "Revenue"), ("gm", "Gross Margin"),
             ("gm_pct", "GM %"), ("generic_mix_units_pct", "Generic Mix % (units)")]
for col, (k, label) in zip(hcols, hero_kpis):
    did = chain_did(k)
    arrow = "→"
    if did is not None and not pd.isna(did):
        arrow = "↑" if did > 0 else "↓" if did < 0 else "→"
    direction = KPI_DIRECTION.get(k, 0)
    delta_color = "normal"
    if did is not None and not pd.isna(did):
        good = (did > 0 and direction > 0) or (did < 0 and direction < 0)
        delta_color = "normal" if good else "inverse"
    col.metric(
        f"{label}",
        f"{did:+.1f}%" if did is not None and not pd.isna(did) else "—",
        f"DiD avg across {len(selected_pairs)} pair(s) {arrow}",
    )

st.caption(f"Pre window: **{pre_from} → {pre_to}** (8 days) · Post window: **{post_from} → {post_to}** (8 days). Test = Set A (price ↑). Control = Set B.")

st.divider()

# -----------------------------------------------------------------------------
# Section 1 — The Setup (pair parity check)
# -----------------------------------------------------------------------------

st.subheader("1. The Setup — are the pairs actually comparable?")
st.caption(
    f"Pre-treatment baseline window: **{QUANT_BASELINE_START} → {QUANT_BASELINE_END}** "
    "(~80 days before the test windows). Pairs were chosen for similarity along these dimensions. "
    "If any baseline is wildly off, treat that pair's DiD with caution."
)

@st.cache_data(ttl=3600, show_spinner=False)
def baseline_summary(stores: tuple[str, ...]) -> dict:
    df = fetch_quant_sales(stores, QUANT_BASELINE_START.isoformat(),
                           QUANT_BASELINE_END.isoformat())
    if df.empty:
        return {s: {"revenue": 0, "mau": 0, "generic_sub_pct": 0} for s in stores}
    out = {}
    for s in stores:
        sub = df[df["store_name"] == s]
        gross = sub[sub["bill_flag"] == "gross"]
        rev = float((gross["revenue_value"] - gross["promo_discount"]).sum())
        mau = int(gross["patient_id"].dropna().nunique())
        is_gen = gross["assortment_classification_id"].isin(ASSORTMENT_GENERIC)
        gen_units = float(gross.loc[is_gen, "net_quantity"].sum())
        tot_units = float(gross["net_quantity"].sum())
        out[s] = {
            "revenue": rev, "mau": mau,
            "generic_sub_pct": (gen_units / tot_units * 100) if tot_units else 0,
        }
    return out

all_stores = tuple(sorted({p["test"] for p in selected_pairs} | {p["control"] for p in selected_pairs}))
bl = baseline_summary(all_stores)

setup_rows = []
for p in selected_pairs:
    a, b = p["test"], p["control"]
    setup_rows.append({
        "Pair": f"P{p['pair']}",
        "City": p["city"],
        "Set A (Test)": a,
        "Set B (Control)": b,
        "A revenue (₹L)": f"{bl[a]['revenue']/1e5:.2f}L",
        "B revenue (₹L)": f"{bl[b]['revenue']/1e5:.2f}L",
        "A MAU": f"{bl[a]['mau']:,}",
        "B MAU": f"{bl[b]['mau']:,}",
        "A gen-mix %": f"{bl[a]['generic_sub_pct']:.1f}%",
        "B gen-mix %": f"{bl[b]['generic_sub_pct']:.1f}%",
    })
st.dataframe(pd.DataFrame(setup_rows), width='stretch', hide_index=True)

st.divider()

# -----------------------------------------------------------------------------
# Section 2 — Headline: Revenue & Margin
# -----------------------------------------------------------------------------

st.subheader("2. Headline — did revenue and margin move?")
st.caption(
    "Each row = one pair. **DiD** is the treatment effect: positive (green) means Test outperformed Control on that KPI."
)
render_kpi_table(combined.to_dict("records"),
                 ["revenue", "gm", "gm_pct", "aov", "aom", "acm"])

st.divider()

# -----------------------------------------------------------------------------
# Section 3 — Generic Mix (the CORE story)
# -----------------------------------------------------------------------------

st.subheader("3. Generic Mix — the CORE side-effect")
st.caption(
    "If the price hike pushes customers from Generic → Ethical, generic mix should **drop in Test more than in Control** "
    "(DiD column negative). This is the single most important section."
)
render_kpi_table(combined.to_dict("records"),
                 ["generic_mix_units_pct", "generic_mix_revenue_pct"])

# Per-pair bar chart: Pre vs Post for Test and Control side by side
mix_chart_rows = []
for p in selected_pairs:
    sub = combined[(combined["pair"] == p["pair"]) &
                   (combined["kpi"] == "generic_mix_units_pct")]
    if sub.empty: continue
    r = sub.iloc[0]
    label = f"P{p['pair']}"
    mix_chart_rows.extend([
        {"pair": label, "side": "Test", "period": "Pre",  "value": r["test_pre"] * 100},
        {"pair": label, "side": "Test", "period": "Post", "value": r["test_post"] * 100},
        {"pair": label, "side": "Ctrl", "period": "Pre",  "value": r["ctrl_pre"] * 100},
        {"pair": label, "side": "Ctrl", "period": "Post", "value": r["ctrl_post"] * 100},
    ])
if mix_chart_rows:
    mix_df = pd.DataFrame(mix_chart_rows)
    mix_df["bar"] = mix_df["side"] + "-" + mix_df["period"]
    chart = (
        alt.Chart(mix_df).mark_bar().encode(
            y=alt.Y("pair:N", title=None),
            yOffset=alt.YOffset("bar:N", sort=["Test-Pre", "Test-Post", "Ctrl-Pre", "Ctrl-Post"]),
            x=alt.X("value:Q", title="Generic mix % (units)"),
            color=alt.Color("bar:N",
                            scale=alt.Scale(
                                domain=["Test-Pre", "Test-Post", "Ctrl-Pre", "Ctrl-Post"],
                                range=["#85C1E9", "#1F618D", "#D5DBDB", "#566573"]),
                            legend=alt.Legend(title=None)),
            tooltip=["pair", "side", "period", alt.Tooltip("value:Q", format=".1f")],
        ).mark_bar(size=12).properties(height=max(80, 70 * len(selected_pairs)))
    )
    st.altair_chart(chart, width='stretch')

st.divider()

# -----------------------------------------------------------------------------
# Section 4 — Customer Behaviour
# -----------------------------------------------------------------------------

st.subheader("4. Customer Behaviour — did anything weird happen with bills, frequency, or coupons?")
render_kpi_table(combined.to_dict("records"),
                 ["bill_count", "unique_customers", "frequency",
                  "promo_usage_rate", "promo_share_revenue"])
total_genrtn = combined.loc[combined["kpi"] == "promo_usage_rate", "test_post"].sum() + \
               combined.loc[combined["kpi"] == "promo_usage_rate", "ctrl_post"].sum()
if total_genrtn == 0:
    st.caption("⚠️ GENRTN promo usage is **0% across all selected pairs (post)**. Once the field team starts applying the price-match coupon, these rows will populate.")

st.divider()

# -----------------------------------------------------------------------------
# Section 5 — Guardrails
# -----------------------------------------------------------------------------

st.subheader("5. Guardrails — did anyone walk away?")
st.caption(
    "Return rate, 8-day repeat rate (returning customers), and 8-day churn (lost customers). "
    "**Repeat / churn at 8 days are noisy** — the metric matures at 30 / 60 days. The DiD column shows test vs control percentage-point gap."
)
render_kpi_table(combined.to_dict("records"),
                 ["return_rate", "repeat_rate", "churn_rate"])

st.divider()

# -----------------------------------------------------------------------------
# Section 6 — SKU-level Elasticity
# -----------------------------------------------------------------------------

st.subheader("6. SKU-level elasticity — which drugs drove the result?")
st.caption(
    "For each price-↑ SKU sold across selected Test stores, compare pre- and post-window units & realised price. "
    "**Own-price elasticity = %Δ units / %Δ price.** Negative = customers buy fewer when price goes up (rational). Positive = noise / supply effects."
)

# Aggregate test pre/post across all selected test stores (sku_ids already loaded above)
test_pre_all = pd.concat([pair_data[p["pair"]]["test_pre"] for p in selected_pairs], ignore_index=True)
test_post_all = pd.concat([pair_data[p["pair"]]["test_post"] for p in selected_pairs], ignore_index=True)

elast = compute_elasticity(test_pre_all, test_post_all, sku_ids)
elast_with_data = elast[(elast["pre_units"] > 0) | (elast["post_units"] > 0)].copy()
elast_with_data = elast_with_data.merge(skus[["drug_id", "drug_name", "old_price", "new_price"]],
                                         on="drug_id", how="left")
st.markdown(f"**Universe:** {len(sku_ids)} price-↑ SKUs · with sales in either window: **{len(elast_with_data)}**")

# Top 20 most negative elasticities
display_e = elast_with_data.dropna(subset=["own_elasticity"]).copy()
display_e = display_e[display_e["own_elasticity"].between(-50, 50)]  # drop crazy outliers
display_e = display_e.sort_values("own_elasticity").head(20)
display_e_show = display_e[["drug_id", "drug_name", "old_price", "new_price",
                              "pre_units", "post_units",
                              "pre_avg_price", "post_avg_price",
                              "units_delta_pct", "price_delta_pct", "own_elasticity"]]
st.markdown("**Top 20 most price-sensitive SKUs (most-negative own-price elasticity)**")
st.dataframe(display_e_show, width='stretch', hide_index=True)

# Cross-elasticity: ethical units vs generic price (chain-level)
gen_pre = test_pre_all[test_pre_all["bill_flag"] == "gross"]
gen_pre_g = gen_pre[gen_pre["assortment_classification_id"].isin(ASSORTMENT_GENERIC)]
gen_post = test_post_all[test_post_all["bill_flag"] == "gross"]
gen_post_g = gen_post[gen_post["assortment_classification_id"].isin(ASSORTMENT_GENERIC)]

eth_pre_units = float(gen_pre[gen_pre["assortment_classification_id"].isin(ASSORTMENT_ETHICAL)]["net_quantity"].sum())
eth_post_units = float(gen_post[gen_post["assortment_classification_id"].isin(ASSORTMENT_ETHICAL)]["net_quantity"].sum())
gen_pre_units = float(gen_pre_g["net_quantity"].sum())
gen_post_units = float(gen_post_g["net_quantity"].sum())
gen_pre_price = float(gen_pre_g["revenue_value"].sum() / gen_pre_units) if gen_pre_units else 0
gen_post_price = float(gen_post_g["revenue_value"].sum() / gen_post_units) if gen_post_units else 0

eth_delta_pct = ((eth_post_units - eth_pre_units) / eth_pre_units * 100) if eth_pre_units else None
gen_price_delta_pct = ((gen_post_price - gen_pre_price) / gen_pre_price * 100) if gen_pre_price else None
cross_elast = (eth_delta_pct / gen_price_delta_pct) if (eth_delta_pct is not None and gen_price_delta_pct) else None

cc1, cc2, cc3 = st.columns(3)
cc1.metric("%Δ Ethical units (Test)", fmt_delta(eth_delta_pct))
cc2.metric("%Δ Generic avg price (Test)", fmt_delta(gen_price_delta_pct))
cc3.metric("Cross-price elasticity (Gen→Eth)",
           f"{cross_elast:+.2f}" if cross_elast is not None and not pd.isna(cross_elast) else "—",
           help="Positive = customers substituted toward ethical when generic price rose.")

st.divider()

# -----------------------------------------------------------------------------
# Section 7 — Drill-down
# -----------------------------------------------------------------------------

st.subheader("7. Drill-down — bill-level export")
drill_pair_label = st.selectbox("Pick a pair to inspect", options=pair_labels)
dp = next(p for p, lbl in zip(STORE_PAIRS, pair_labels) if lbl == drill_pair_label)
side = st.radio("Side", ["Test", "Control"], horizontal=True)
window = st.radio("Window", ["Pre", "Post"], horizontal=True, index=1)
key = ("test_" if side == "Test" else "ctrl_") + window.lower()
df_view = pair_data.get(dp["pair"], {}).get(key, pd.DataFrame())

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
        file_name=f"quant_drilldown_pair{dp['pair']}_{side}_{window}.csv",
        mime="text/csv",
    )

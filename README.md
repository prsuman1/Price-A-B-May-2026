# Price Change A/B — Insight Dashboard

Two-page Streamlit dashboard for the 2026-05-06 price-change experiment:

- **🏪 Store Insight** — surveyor RAG view of customer reactions, joined to Redshift sales.
- **📊 Quant Insight** — store-pair A/B test of revenue, GM, generic mix, elasticity etc.

## Run locally

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Two data modes

The dashboard reads sales data in one of two ways:

1. **Snapshot mode (default, no VPN needed).** If `data_snapshots/sales.parquet` exists, the dashboard loads from it. This is the mode used by deployed instances (e.g. Streamlit Cloud) that can't reach `redshift.internal`.
2. **Live Redshift mode (local with VPN).** If the snapshot file is absent, the dashboard falls back to live Redshift queries. Requires `.env` with credentials (gitignored).

## Refreshing the deployed data

Whenever you want the deployed dashboard to reflect newer sales:

```bash
# Locally, with VPN reachable:
.venv/bin/python refresh_snapshots.py     # writes data_snapshots/sales.parquet
git add data_snapshots/sales.parquet
git commit -m "refresh sales snapshot"
git push                                  # Streamlit Cloud auto-redeploys
```

The script pulls all rows for the 14 stores (union of 8 survey stores + 6 extra pair-control stores) from `QUANT_BASELINE_START` (currently 2026-02-06) through today. Current snapshot ≈ 500K rows, ~16 MB.

## Inputs (auto-discovered via glob)

- `Customer Behaviour Survey Replies*.csv` — Google Form export.
- `Final Price Change Proposal*.csv` — 346 SKUs whose price went up.
- `unmatched_numeric_bills.csv` — user-corrected mapping for survey rows where the bill number was entered as a bare numeric value (defensive fallback).
- `data_snapshots/sales.parquet` — Redshift slice for offline runs (committed to git).

## Notes

- Data cached for 30 min (`st.cache_data` TTL). Menu → "Clear cache" + "Rerun" to force refresh.
- `clean_bill_ids.csv` is regenerated on every Store Insight render — open it to spot any surveyor data-quality issues.
- Open items are listed in the footer of the dashboard.

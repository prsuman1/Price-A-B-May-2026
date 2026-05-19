# Price Change A/B — Insight Dashboard

Three-page Streamlit dashboard for the 2026-05-06 price-change experiment:

- **🏪 Store Insight** — surveyor RAG view of customer reactions, joined to Redshift sales.
- **📊 Quant Insight** — pilot vs non-pilot A/B test of revenue, GM, generic mix, elasticity etc.
- **💬 Ask Zeno** — natural-language Q&A over the data via Claude Sonnet 4.6 (OpenRouter).

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

The script writes **three** files into `data_snapshots/`:
- `sales.parquet` (~16 MB) — line-level rows for the 14 pair-related stores.
- `store_totals.parquet` (~2 MB) — per-store-per-day-per-bill-set aggregates for ALL ~201 chain stores. Powers the Quant Summary tab's "All-Other Non-Pilot" comparator.
- `patients_first_seen.parquet` (~0.5 MB) — patient_id → first-ever bill date in chain. Powers the "new patient" KPI.

Combined ~19 MB. Within GitHub's 100 MB/file limit.

## Inputs (auto-discovered via glob)

- `Customer Behaviour Survey Replies*.csv` — Google Form export.
- `Final Price Change Proposal*.csv` — 346 SKUs whose price went up.
- `unmatched_numeric_bills.csv` — user-corrected mapping for survey rows where the bill number was entered as a bare numeric value (defensive fallback).
- `data_snapshots/sales.parquet` — Redshift slice for offline runs (committed to git).

## Chat page

Requires an OpenRouter API key. Add to `.env`:

```
openrouterkey=sk-or-v1-...
```

On Streamlit Cloud, add the same key as a secret named `openrouterkey` in the app settings.

Chat history persists to `chat_history.csv`. The file is committed to git, but Streamlit Cloud's file writes are ephemeral per deploy — to keep history permanently across deploys, use the sidebar's "Download chat_history.csv" button periodically and commit the file locally.

## Notes

- Data cached for 30 min (`st.cache_data` TTL). Menu → "Clear cache" + "Rerun" to force refresh.
- `clean_bill_ids.csv` is regenerated on every Store Insight render — open it to spot any surveyor data-quality issues.
- Open items are listed in the footer of the dashboard.

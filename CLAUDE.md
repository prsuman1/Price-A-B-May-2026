# Price-Change A/B Dashboard — Project Context

## What this is

A Streamlit dashboard that measures customer reaction to a price increase rolled out on **2026-05-06** across **346 SKUs** at a pharmacy chain. A field team sits in 8 stores logging customer behaviour into a Google Form; their replies are joined to Redshift sales data to attribute each reaction to a specific bill, store, and SKU.

The dashboard now has **two pages**:
- **🏪 Store Insight** — fully built. RAG (red/amber/green) bill-health rollup, store cuts, sentiment cross-tabs, walk-aways, control comparison, drill-down.
- **📊 Quant Insight** — placeholder; quantitative/financial analysis to be built next.

## Run

```bash
cd /Users/suman/Documents/Price-ChangeA-B
.venv/bin/streamlit run app.py
```

Requires VPN to reach `redshift.internal`. `.env` is gitignored and holds Redshift creds.

## File layout

| File | Purpose |
|---|---|
| `app.py` | Thin entry. `st.navigation([...]).run()` registers the two pages. **Must hold the only `st.set_page_config` call.** |
| `store_insight.py` | All Store Insight logic — sidebar filters, top KPIs, 6 tabs. |
| `quant_insight.py` | Placeholder for the Quant Insight page. |
| `data.py` | Shared data layer: env loader, Redshift connection, CSV loaders (glob-discovered), normalization helpers, `build_joined()` returning the joined dict. All cached with `@st.cache_data(ttl=1800)`. |
| `requirements.txt` | streamlit / pandas / psycopg2-binary / python-dotenv / altair |
| `.env` | Redshift creds (gitignored). |
| `.gitignore` | `.env` only. Project is not yet a git repo. |

## Data sources

### 1. Survey CSV — auto-discovered via glob `Customer Behaviour Survey Replies*.csv`

Currently `Customer Behaviour Survey Replies - Sheet4 (1).csv`, ~2,186 rows. **Hinglish headers**, accessed by **column index** in `data.py`:

| Idx | Column | Notes |
|---:|---|---|
| 0 | Submission ID | Form response ID |
| 2 | Submitted at | ISO timestamp; filtered to ≥ `2026-05-06` |
| 3 | Store Name | Free-text from a dropdown (8 distinct values) |
| 4 | Staff informed? | Yes/No — staff proactively told customer about price change |
| 5 | Customer checked bill? | Yes/No |
| 6 | Customer spoke about price? | Yes/No |
| 7 | What customer said | Multi-select; comma-separated |
| 8 | Customer non-verbal reaction? | Yes/No — **conditionally shown only when col 6 = no** (so blanks here ≈ "spoke = yes") |
| 9 | Reaction type | Multi-select; comma-separated |
| 10 | `Bill ka kya hua?` (outcome) | 3 canonical values + 130+ free-text |
| 11 | Bill Number / Patient ID | Alphanumeric serial like `SANP-511929` |
| 12 | Notes | Free text |

### 2. SKU CSV — auto-discovered via glob `Final Price Change Proposal*.csv`

Currently `Final Price Change Proposal_Generic with Old Price.xlsx - Sheet1 (3).csv`, **346 rows**. Cols: `drug_id` (joins to `sales.drug-id`), `Drug name`, `Selling Price` (new), `Assortment Classification`, `Old Price`, `Drug Type` (GA/NGA — Goodaid vs non-Goodaid).

### 3. Redshift — `"prod2-generico"."sales"`

Connection via `psycopg2` + `.env`. Wire-compatible with Postgres. **Schema name has a hyphen → must be double-quoted.** Connection details live in `.env` (gitignored). Uses read-only user `ro_user_data_dbeaver`.

`fetch_sales(serials)` pulls only **12 columns** from the 98-column table, filtered to `created-at >= 2026-05-06` and `serial IN (survey serials)`. Chunks at 1000.

### 4. `unmatched_numeric_bills.csv` (defensive only)

Originally an export-and-correct workflow when surveyors entered short numeric bill IDs that didn't match `sales.bill-id`. The latest survey export has **zero numeric entries**, so this file is now dormant; the override-lookup code path is kept as a safety net.

## Business semantics — the rules that took multiple iterations to nail down

### Go-live and filtering

`GO_LIVE_DATE = 2026-05-06`. Pre-go-live rows in the survey are surveyor-training data; filtered out at load time.

### `bill_kind` taxonomy (per survey row)

- **`serial`** — bill-ID parsed cleanly (after Z-strip + uppercase + KEWT→KRWT typo fix) and matches `^[A-Z]+-\d+$`. Joinable to `sales.serial`.
- **`bill_created_no_id`** — bill was made (per outcome) but surveyor didn't write the bill ID. Cannot be joined to sales; sentiment columns still usable.
- **`walkaway`** — outcome contains "Bana hi nahin. Customer chale gaye." regardless of bill-ID state.

### `is_walkaway`

**Outcome-driven only.** `outcome.str.contains("Bana hi nahin", case=False)`. Ignores bill-ID state. (Earlier iterations conflated blank-bill with walkaway — corrected by user.)

### Bill-ID normalization

In `_normalize_alpha_serial()`:
1. Strip whitespace.
2. Strip leading `Z`/`z` (surveyors prepend a Z to some entries; `Zgwsv-67046` → `gwsv-67046`).
3. `.upper()`.
4. Apply `PREFIX_FIXES = {"KEWT": "KRWT"}` — confirmed surveyor typo at Khar West (E adjacent to R on keyboard); recovers ~120 bills.
5. Validate against `^[A-Z]+-\d+$`.

### Store name mapping (survey → sales)

```
Bandra Hill Road       → Bandra
Colaba                 → Colaba
Dhamankar Naka         → Bhiwandi Dhamankar Naka
Goregaon               → Goregaon
Goregaon W SV Road     → Goregaon West S.V. Road
Khanda Colony          → Khanda Colony   (only ~4 survey rows — unstable)
Khar                   → Khar West       (verified via KRWT serial prefix)
Sanpada                → Sanpada
```

### RAG classification

Implemented in `data.py::classify_rag(outcome, used_genrtn)`. Precedence:
1. **🔴 RED** if outcome (lowercased) contains any of: `"bana hi nahin"`, `"chale gaye"`, `"didn't buy"`, `"didnt buy"`, `"left without"`. Conservative — only explicit walkaway phrases.
2. **🟡 AMBER** if outcome contains `"partial bana"` or `"item kam"` (→ `is_amber_partial=True`) **OR** `used_genrtn=True` (→ `is_amber_genrtn=True`). Both flags can be True. **GENRTN trumps GREEN** — the till record beats surveyor's "no change" label.
3. **🟢 GREEN** otherwise. Free-text outcomes default here. Top-20 unmatched free-text outcomes shown in the audit expander on the RAG tab so the patterns can be extended later.

### GENRTN coupon

Price-match coupon. Detected via `UPPER(TRIM(promo-code)) LIKE 'GENRTN%'` on the sales side. **Currently 0 uses chain-wide post-go-live** — open item. The dedicated GENRTN tab was removed (no data) but the signal still surfaces as the `↳ GENRTN` sub-row inside Store cut and Control comparison.

## Store Insight page — tab summary

(File: `store_insight.py`. Sidebar filters: date range / store multiselect / drug-type multiselect.)

**Top of page (always visible):**
- 3 RAG metric chips (🟢 / 🟡 / 🔴) with %s.
- 6 functional KPIs (survey rows, bills created, bills w/ price-↑ SKU, bills joined, GENRTN bills, bill-no-ID).
- "Funnel" expander (math behind the KPIs).
- "Data freshness" expander (survey serials not yet in `sales`).

**6 tabs:**

1. **RAG by Store × SKU** *(headline)* — horizontal Altair bar chart, store names pinned on the left, two horizontal bars per store (with vs without price-↑ SKU), segments = 🟢 No change · 🟠 Partial bill. Cross-tab beneath. Audit expander shows `outcome_audit` counts and top-20 free-text outcomes that defaulted to GREEN. Below the chart: per-store table of "Customer interactions excluded from chart (no SKU info)" with `Total interactions`, `Bill made / ID not entered`, `Walk-away`, plus a TOTAL row.

2. **Store cut** — pinned `store` index column. Per-store: customer interactions, bills joined, bills w/ price-↑ SKU, RAG counts (green / amber / ↳partial / ↳GENRTN / red), and four sentiment percentages: `% spoke (all)` / `% spoke (price-↑ bills)` / `% non-verbal (all)` / `% non-verbal (price-↑ bills)`.

3. **Sentiment cross-tabs** — radio scope: `All bills` / `Bills WITH price-↑ SKU only` / `Side by side`. In side-by-side mode the layout is **organised by section first** (Yes/No row → What customer said → Reaction type), with **with/without stacked vertically** within each section. Yes/No cross-tabs use 4 fields (Staff informed, Spoke about price, Non-verbal, Checked bill); blanks labeled `(blank)`; columns ordered `yes / no / (blank) / All`. Multi-select bar charts show only Green and Amber (Red has no bills).

4. **Walk-aways** *(= 🔴 RED bucket)* — bar chart per store; table with verbal/non-verbal complaint splits.

5. **Control comparison** — single table: `WITH price-↑ SKU` vs `WITHOUT` × {🟢 / 🟡 / ↳partial / ↳GENRTN / 🔴}, plus `% spoke about price`, `% non-verbal reaction`, `Avg revenue`. Each colour cell shows raw count + % within row.

6. **Drill-down** — RAG filter chip, full bill-level table with `rag` / `is_amber_partial` / `is_amber_genrtn` columns, CSV download.

**Footer:** open-item disclaimers (Khar→Khar West, Goregaon vs Goregaon W SV Road, GENRTN code).

## Quant Insight page

(File: `quant_insight.py`. Sidebar: pre/post date inputs.)

A/B test of the 2026-05-06 price increase using **6 Mumbai store pairs**. **Pilot** = stores where the price went up. **Non-Pilot** = paired control. **DiD** = Pilot Δ% − Non-Pilot Δ%.

### Pair definitions (in `data.py::STORE_PAIRS`)

| # | Pilot (price ↑) | Non-Pilot (paired control) |
|---:|---|---|
| 1 | Colaba | Pantnagar |
| 2 | Khanda Colony | Virar West |
| 3 | Bhiwandi Dhamankar Naka | Dahisar |
| 4 | Khar West | Dombivali |
| 5 | Goregaon West S.V. Road | Mulund West-Sarvodaya Nagar |
| 6 | Sanpada | Airoli |

Helper constants: `PILOT_STORES`, `PAIRED_NON_PILOT_STORES`.

### Methodology

- **Symmetric pre/post windows.** Currently 12 days: Pre `2026-04-24 → 2026-05-05`, Post `2026-05-06 → 2026-05-17`. Constants in `data.py`.
- **Three comparators:** Pilot vs **Paired Non-Pilot** (6 stores) vs **All-Other** (~195 chain stores not in the experiment, from per-store-per-day aggregates).
- **Bill-set toggle.** Page-level radio: `Price-↑ bills` (default — bills with ≥1 price-↑ SKU) / `Non-price-↑ bills` (bills with ZERO price-↑ SKU lines) / `All bills`. Implemented as `bill_filter` kwarg through `compute_kpis`, `build_pair_table`, `compute_repeat_churn`.
- **DiD coloured by direction.** `KPI_DIRECTION` map in `quant_insight.py`. Positive-when-↑-is-good colours green; negative colours red.

### KPI catalogue (in `data.py::compute_kpis`)

Grouped for the dashboard:

**Volume** — `bill_count`, `unique_customers` (patients), `total_quantity`, `repeat_patient_count` (≥2 bills in window), `new_patient_count` (first-ever bill in chain falls in window — uses `patients_first_seen.parquet`).

**Revenue & Margin** — `revenue` (`Σ(revenue_value − promo_discount)`), `gm` (`revenue − cogs − promo_discount`; COGS = `Σ(net_qty × purchase_rate)`), `gm_pct`.

**Per bill / per patient** — `aov` (revenue/bill), `aom` (gm/bill), `acm` (gm/patient), `items_per_bill`, `revenue_per_patient`, `frequency` (bills/patient).

**Mix** — `generic_mix_units_pct`, `generic_mix_revenue_pct` (assortment IDs 3,4 = Generic; 1,2 = Ethical).

**Promo / Returns** — `promo_usage_rate` (GENRTN bills/total bills), `promo_share_revenue`, `return_rate`.

**Guardrails (pair view only)** — `repeat_rate` (customers in pre AND post), `churn_rate` (in pre, not in post). Computed by `compute_repeat_churn`.

**Elasticity** — `compute_elasticity` per drug_id; own-price + cross-price (Gen→Eth) at chain level.

### Page structure

4 tabs:
1. **🏪 Summary** — Hero metrics + per-group table: Pilot Pre/Post/Δ% vs Non-Pilot Δ% vs All-Other Δ%, with two DiD columns (vs Paired, vs All-Other). KPIs rendered by group.
2. **🔀 Store vs Store** — Per-pair pilot ↔ non-pilot for the same KPI groups.
3. **📈 Elasticity** — Top-20 most price-sensitive SKUs + chain-level cross-price elasticity (Gen→Eth).
4. **🔍 Drill-down** — Pair × side × window selector → bill-level table + CSV.

### Data layer (`data.py`)

- `_load_sales_snapshot()` — `data_snapshots/sales.parquet` (line-level, 14 pair-related stores).
- `_load_store_totals_snapshot()` — `data_snapshots/store_totals.parquet` (per-store-per-day-per-bill-set aggregates, all 201 stores).
- `_load_patients_snapshot()` — `data_snapshots/patients_first_seen.parquet` (patient_id → first-ever bill date).
- `aggregate_totals(...)` — sums the per-store-per-day totals for a store set + window; returns the same dict shape as `compute_kpis`. Used for the All-Other comparator.

### Caveats

- The All-Other column in Summary uses per-store-per-day aggregates → `unique_customers` may double-count cross-store visitors. Captioned in the UI.
- `new_patient_count` is 0 for the All-Other comparator (requires line-level patient data we don't snapshot for 195 stores).
- 3 pairs were re-labelled mid-project (pairs 2, 3, 5). Earlier "Test/Control" DiDs in chat history may not match current "Pilot/Non-Pilot" numbers — always trust the dashboard's current numbers.

## Deployment

Pushed to **https://github.com/prsuman1/Price-A-B-May-2026** (private). Streamlit Cloud reads `data_snapshots/sales.parquet`.

**Refresh data deployed:**
```bash
.venv/bin/python refresh_snapshots.py   # local, with VPN
git add data_snapshots/sales.parquet
git commit -m "refresh sales snapshot"
git push                                # Streamlit Cloud auto-redeploys
```

The `refresh_snapshots.py` script pulls ~498K rows for the 14 stores (8 survey ∪ 6 extra pair-control) from `QUANT_BASELINE_START` through today, writes Parquet (~16 MB). `data.py::_load_sales_snapshot()` is the snapshot-first reader; `fetch_sales` / `fetch_quant_sales` fall back to Redshift only when the file is missing.

## Open items the user should resolve

1. **GENRTN volume = 0** — if the field team is using a price-match coupon, confirm the actual code in `sales.promo-code`. Other `GEN%` codes that exist: `GEN50`, `GEN70`, `GENLAB`, `GENCV`.
2. **Goregaon vs Goregaon W SV Road** — survey treats them as separate stores; sales has both, with serial prefixes `GORE` and `GWSV`. Routed correctly today; confirm with field team if they think otherwise.
3. **Surveyor data-quality** — Khar West (18) and Bhiwandi Dhamankar Naka (16) have a relatively high rate of "bill made, ID not entered". Easy fix at the source.
4. **Free-text outcome patterns** — 144 outcomes default to GREEN by being unrecognized. Audit expander surfaces them; user can refine the RED/AMBER patterns when worth it.

## Dev gotchas

- **Restart Streamlit (don't rely on hot-reload) when adding new top-level constants/functions to `data.py`.** Hot-reload handles edits to page scripts (`store_insight.py`, `quant_insight.py`) but the imported `data.py` module's new symbols may not propagate, causing `ImportError: cannot import name 'X' from 'data'` even though X exists. Kill via `lsof -nP -iTCP:8501 -sTCP:LISTEN` then re-launch.
- **`set_page_config` lives only in `app.py`.** Calling it from a page script will error.
- **`st.dataframe(df.set_index("col"))` pins that column on the left during horizontal scroll.** Used in Store cut.
- **Free-text outcome blanks in non-verbal column are structural, not missing data.** The Google Form skips the non-verbal question when the customer spoke about price.
- **Caches**: `@st.cache_data(ttl=1800)` on `load_survey`, `load_skus`, `load_numeric_overrides`, `fetch_sales`, `build_joined`. Use the menu's "Clear cache" + "Rerun" to force a refresh.
- **Test runs**: `.venv/bin/python -c "from streamlit.testing.v1 import AppTest; at = AppTest.from_file('app.py', default_timeout=300); at.run(); print([str(e.value) for e in at.exception])"` — useful for catching import / render errors without booting a browser.

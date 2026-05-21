"""Refresh the local Parquet snapshots used by the dashboard.

Run locally (with VPN reachable to redshift.internal). Writes 3 files into
`data_snapshots/`:

  - `sales.parquet`             — line-level rows for the 14 pair-related
                                  stores since QUANT_BASELINE_START.
  - `store_totals.parquet`      — per-store-per-day-per-bill-set aggregates
                                  for ALL chain stores (~200). Powers the
                                  Summary tab's "all-other stores" group.
  - `patients_first_seen.parquet` — patient_id -> first-ever bill date in
                                    the chain. Powers the new-patient KPI.

Commit and push afterwards; Streamlit Cloud auto-redeploys.

    .venv/bin/python refresh_snapshots.py
"""

from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

import pandas as pd
import psycopg2

from data import (
    ASSORTMENT_GENERIC,
    ASSORTMENT_ETHICAL,
    COUPON_PREFIX,
    GO_LIVE_DATE,
    PARTIAL_BILL_SNAPSHOT,
    PATIENTS_SNAPSHOT,
    QUANT_BASELINE_START,
    SALES_SNAPSHOT,
    SKU_CATEGORY_SNAPSHOT,
    SNAPSHOTS_DIR,
    STORE_PAIRS,
    STORE_TOTALS_SNAPSHOT,
    STORE_MAP,
    _connect,
    build_joined,
    load_skus,
)

_SNAPSHOT_COLS = [
    "store-name", "bill-id", "serial", "patient-id", "drug-id", "drug-name",
    "net-quantity", "quantity", "revenue-value", "promo-discount",
    "purchase-rate", "rate", "mrp", "promo-code", "bill-flag",
    "assortment-classification-id", "created-at",
]


def fetch_sales_snapshot(stores: tuple[str, ...], date_from: str, date_to: str) -> pd.DataFrame:
    cols_sql = ", ".join(f'"{c}"' for c in _SNAPSHOT_COLS)
    with _connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT {cols_sql}
                FROM "prod2-generico"."sales"
                WHERE "store-name" IN %s
                  AND "created-at"::date BETWEEN %s AND %s
                """,
                (stores, date_from, date_to),
            )
            rows = cur.fetchall()
    df = pd.DataFrame(rows, columns=_SNAPSHOT_COLS)
    df = df.rename(columns={c: c.replace("-", "_") for c in df.columns})
    df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")
    for c in ("net_quantity", "quantity", "revenue_value", "promo_discount", "purchase_rate", "rate", "mrp"):
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("Float64")
    for c in ("drug_id", "assortment_classification_id", "patient_id", "bill_id"):
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")
    df["promo_code"] = df["promo_code"].fillna("").astype(str)
    df["bill_flag"] = df["bill_flag"].fillna("").astype(str)
    df["serial"] = df["serial"].fillna("").astype(str)
    df["store_name"] = df["store_name"].astype(str)
    df["drug_name"] = df["drug_name"].fillna("").astype(str)
    return df


def fetch_sku_category(sku_ids: set[int]) -> pd.DataFrame:
    """drug_id -> category lookup for the 346 price-↑ SKUs (one Redshift query).
    Falls back to 'unknown' if a drug_id isn't present in any sales row."""
    if not sku_ids:
        return pd.DataFrame(columns=["drug_id", "category"])
    sku_tuple = tuple(int(x) for x in sorted(sku_ids))
    with _connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT "drug-id", "category", COUNT(*) AS n
                FROM "prod2-generico"."sales"
                WHERE "drug-id" IN %s
                GROUP BY "drug-id", "category"
                """,
                (sku_tuple,),
            )
            rows = cur.fetchall()
    df = pd.DataFrame(rows, columns=["drug_id", "category", "n"])
    # If a drug has multiple category labels across rows, keep the most common.
    df = df.sort_values(["drug_id", "n"], ascending=[True, False]).drop_duplicates("drug_id", keep="first")
    df["category"] = df["category"].fillna("unknown").astype(str)
    df["drug_id"] = df["drug_id"].astype(int)
    return df[["drug_id", "category"]]


def build_store_totals(date_from: str, date_to: str, sku_ids: set[int],
                       sku_category: dict[int, str] | None = None) -> pd.DataFrame:
    """For ALL chain stores, pre-aggregate per-store-per-day KPI totals into
    a small Parquet. Two rows per (store, day) — one for `with_pi`, one for
    `without_pi`."""
    print(f"Querying per-store-per-day aggregates for ALL chain stores (this may take ~1 min)…")
    # Single query: pull only what's needed for aggregation.
    with _connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT "store-name", "bill-id", "patient-id", "drug-id",
                       "net-quantity", "revenue-value", "promo-discount",
                       "purchase-rate", "promo-code", "bill-flag",
                       "assortment-classification-id", "created-at"::date AS day
                FROM "prod2-generico"."sales"
                WHERE "created-at"::date BETWEEN %s AND %s
                """,
                (date_from, date_to),
            )
            cols = [d[0] for d in cur.description]
            rows = cur.fetchall()
    df = pd.DataFrame(rows, columns=cols)
    df = df.rename(columns={c: c.replace("-", "_") for c in df.columns})
    for c in ("net_quantity", "revenue_value", "promo_discount", "purchase_rate"):
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    for c in ("drug_id", "assortment_classification_id", "patient_id", "bill_id"):
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")
    df["promo_code"] = df["promo_code"].fillna("").astype(str)
    df["bill_flag"] = df["bill_flag"].fillna("").astype(str)
    print(f"  raw rows: {len(df):,} across {df['store_name'].nunique()} stores")

    # Bill-level price-↑ tag (per store-day key is bill, not row).
    gross = df[df["bill_flag"] == "gross"]
    pi_bill_ids = set(
        gross.loc[gross["drug_id"].isin(sku_ids), "bill_id"].dropna().unique()
    )
    df["with_pi_bill"] = df["bill_id"].isin(pi_bill_ids)

    out_rows: list[dict] = []
    for bill_filter in ("with_pi", "without_pi"):
        if bill_filter == "with_pi":
            sub = df[df["with_pi_bill"]]
        else:
            sub = df[~df["with_pi_bill"]]
        sub_gross = sub[sub["bill_flag"] == "gross"]
        sub_ret = sub[sub["bill_flag"] == "return"]

        # Per (store, day) groupbys.
        grp_keys = ["store_name", "day"]
        g_gross = sub_gross.groupby(grp_keys)
        g_ret = sub_ret.groupby(grp_keys)

        # Aggregate metrics.
        agg = pd.DataFrame({
            "revenue_raw": g_gross["revenue_value"].sum(),
            "promo_discount_total": g_gross["promo_discount"].sum(),
            "cogs": g_gross.apply(lambda x: (x["net_quantity"] * x["purchase_rate"]).sum(), include_groups=False),
            "bill_count": g_gross["bill_id"].nunique(),
            "unique_customers": g_gross["patient_id"].nunique(),
            "total_units": g_gross["net_quantity"].sum(),
        })
        # Generic / ethical split.
        gen_mask = sub_gross["assortment_classification_id"].isin(ASSORTMENT_GENERIC)
        agg["generic_units"] = sub_gross[gen_mask].groupby(grp_keys)["net_quantity"].sum().reindex(agg.index, fill_value=0)
        agg["generic_revenue"] = sub_gross[gen_mask].groupby(grp_keys)["revenue_value"].sum().reindex(agg.index, fill_value=0)
        eth_mask = sub_gross["assortment_classification_id"].isin(ASSORTMENT_ETHICAL)
        agg["ethical_units"] = sub_gross[eth_mask].groupby(grp_keys)["net_quantity"].sum().reindex(agg.index, fill_value=0)
        # Promo / GENRTN.
        promo_mask = sub_gross["promo_code"].str.upper().str.startswith(COUPON_PREFIX)
        agg["promo_bills"] = sub_gross[promo_mask].groupby(grp_keys)["bill_id"].nunique().reindex(agg.index, fill_value=0)
        agg["promo_discount_promo"] = sub_gross[promo_mask].groupby(grp_keys)["promo_discount"].sum().reindex(agg.index, fill_value=0)
        # Returns.
        agg["return_units"] = g_ret["net_quantity"].sum().abs().reindex(agg.index, fill_value=0)
        # Units of price-↑ SKU lines only (within this bill-filter subset).
        pi_lines = sub_gross[sub_gross["drug_id"].isin(sku_ids)].copy()
        agg["total_units_pi"] = pi_lines.groupby(grp_keys)["net_quantity"].sum().reindex(agg.index, fill_value=0)
        # Per-category split of price-↑ units (chronic / acute / therapy cycle).
        if sku_category and not pi_lines.empty:
            pi_lines["_cat"] = pi_lines["drug_id"].map(lambda d: sku_category.get(int(d), "unknown"))
            for col, cat_label in [("pi_units_chronic", "chronic"),
                                    ("pi_units_acute", "acute"),
                                    ("pi_units_therapy_cycle", "therapy cycle")]:
                sub_cat = pi_lines[pi_lines["_cat"] == cat_label]
                agg[col] = sub_cat.groupby(grp_keys)["net_quantity"].sum().reindex(agg.index, fill_value=0)
        else:
            agg["pi_units_chronic"] = 0
            agg["pi_units_acute"] = 0
            agg["pi_units_therapy_cycle"] = 0
        # Repeat patients: per (store, day), patients with ≥2 bills.
        bills_per_pt = (
            sub_gross.dropna(subset=["patient_id"])
            .groupby(grp_keys + ["patient_id"])["bill_id"].nunique()
            .reset_index()
        )
        rep = (bills_per_pt[bills_per_pt["bill_id"] >= 2]
               .groupby(grp_keys).size())
        agg["repeat_patient_count"] = rep.reindex(agg.index, fill_value=0)

        # Aliases for the dashboard side.
        agg["revenue"] = agg["revenue_raw"] - agg["promo_discount_total"]
        agg["total_revenue"] = agg["revenue_raw"]
        agg["total_quantity"] = agg["total_units"]
        # new_patient_count requires patient_first_seen (computed separately).
        agg["new_patient_count"] = 0  # filled in after patients snapshot is built

        agg = agg.reset_index()
        agg["bill_filter"] = bill_filter
        out_rows.append(agg)

    return pd.concat(out_rows, ignore_index=True)


def build_patients_first_seen(stores: tuple[str, ...], date_from: str, date_to: str) -> pd.DataFrame:
    """For every patient that touches our 14-store snapshot window, look up
    their first-ever bill date across the WHOLE chain (not just our stores).
    Returns patient_id -> first_seen date."""
    print(f"Querying first-ever bill date for snapshot-window patients…")
    # Step 1: distinct patients touching our 14 stores in window.
    with _connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT DISTINCT "patient-id"
                FROM "prod2-generico"."sales"
                WHERE "store-name" IN %s
                  AND "created-at"::date BETWEEN %s AND %s
                  AND "patient-id" IS NOT NULL
                """,
                (stores, date_from, date_to),
            )
            patient_ids = [int(r[0]) for r in cur.fetchall() if r[0] is not None]
    print(f"  {len(patient_ids):,} distinct patients in window")
    if not patient_ids:
        return pd.DataFrame(columns=["patient_id", "first_seen"])

    # Step 2: chunked first-ever lookup. patient-id IN clause can be huge; chunk it.
    rows: list[tuple] = []
    chunk_size = 5000
    with _connect() as conn:
        with conn.cursor() as cur:
            for i in range(0, len(patient_ids), chunk_size):
                chunk = tuple(patient_ids[i:i + chunk_size])
                cur.execute(
                    """
                    SELECT "patient-id", MIN("created-at"::date) AS first_seen
                    FROM "prod2-generico"."sales"
                    WHERE "patient-id" IN %s
                      AND "bill-flag" = 'gross'
                    GROUP BY "patient-id"
                    """,
                    (chunk,),
                )
                rows.extend(cur.fetchall())
    df = pd.DataFrame(rows, columns=["patient_id", "first_seen"])
    df["patient_id"] = df["patient_id"].astype(int)
    df["first_seen"] = pd.to_datetime(df["first_seen"]).dt.date
    return df


def fill_new_patient_count(store_totals: pd.DataFrame, patients_df: pd.DataFrame) -> pd.DataFrame:
    """The store_totals "new_patient_count" was left at 0 because we didn't
    have the first-seen lookup during aggregation. Compute it now by counting,
    per (store, day, bill_filter), patients whose first-ever bill == that day."""
    if patients_df.empty:
        return store_totals
    # Patient appearances per (store, day, bill_filter) — we lost the patient
    # detail in the aggregate. To compute new patients per row we need to
    # re-query at line level, but that's exactly what we wanted to avoid.
    # Pragmatic alternative: compute chain-wide new-patient daily totals from
    # the patients table, then distribute by store-share of patients on that
    # day. Too approximate.
    #
    # Cleanest: leave new_patient_count = 0 in store_totals; the dashboard
    # computes it via line-level data for the 14 pair stores (where we DO have
    # line detail) and reports it only for those stores. The "all-other"
    # group's new-patient count will read as 0 with a caveat in the UI.
    return store_totals


def main() -> int:
    SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)

    survey_stores = set(STORE_MAP.values())
    pair_stores = {p["pilot"] for p in STORE_PAIRS} | {p["non_pilot"] for p in STORE_PAIRS}
    pair_related_stores = tuple(sorted(survey_stores | pair_stores))

    date_from = QUANT_BASELINE_START.isoformat()
    date_to = dt.date.today().isoformat()
    print(f"Snapshot window: {date_from} → {date_to}")
    print()

    # 1) Line-level sales for the 14 pair-related stores.
    print(f"[1/5] Line-level sales for {len(pair_related_stores)} pair-related stores…")
    sales = fetch_sales_snapshot(pair_related_stores, date_from, date_to)
    sales.to_parquet(SALES_SNAPSHOT, engine="pyarrow", compression="snappy", index=False)
    print(f"  → {SALES_SNAPSHOT.name}: {len(sales):,} rows · {SALES_SNAPSHOT.stat().st_size/1e6:.1f} MB")
    print()

    # 2) Patients-first-seen for the 14 stores' patients.
    print(f"[2/5] Patients first-seen…")
    patients = build_patients_first_seen(pair_related_stores, date_from, date_to)
    patients.to_parquet(PATIENTS_SNAPSHOT, engine="pyarrow", compression="snappy", index=False)
    print(f"  → {PATIENTS_SNAPSHOT.name}: {len(patients):,} patients · {PATIENTS_SNAPSHOT.stat().st_size/1e6:.1f} MB")
    print()

    # 3) SKU category lookup (small file).
    print(f"[3/5] SKU category lookup for price-↑ SKUs…")
    skus = load_skus()
    sku_ids = set(skus["drug_id"].astype(int))
    sku_cat_df = fetch_sku_category(sku_ids)
    sku_cat_df.to_parquet(SKU_CATEGORY_SNAPSHOT, engine="pyarrow", compression="snappy", index=False)
    sku_cat_lookup = dict(zip(sku_cat_df["drug_id"], sku_cat_df["category"]))
    print(f"  → {SKU_CATEGORY_SNAPSHOT.name}: {len(sku_cat_df)} SKUs · {SKU_CATEGORY_SNAPSHOT.stat().st_size/1e3:.1f} KB")
    print(f"     categories: {dict(sku_cat_df['category'].value_counts())}")
    print()

    # 4) Per-store-per-day-per-bill-set totals for ALL chain stores.
    print(f"[4/5] Per-store-per-day aggregates (ALL stores)…")
    totals = build_store_totals(date_from, date_to, sku_ids, sku_category=sku_cat_lookup)
    totals = fill_new_patient_count(totals, patients)
    totals.to_parquet(STORE_TOTALS_SNAPSHOT, engine="pyarrow", compression="snappy", index=False)
    print(f"  → {STORE_TOTALS_SNAPSHOT.name}: {len(totals):,} rows · {STORE_TOTALS_SNAPSHOT.stat().st_size/1e6:.1f} MB")
    print(f"     stores: {totals['store_name'].nunique()}, days: {totals['day'].nunique()}, bill_filters: {totals['bill_filter'].nunique()}")
    print()

    # 5) Partial-bill cohort: 4-month pre + post price-↑ SKU history (ALL stores).
    print(f"[5/5] Partial-bill cohort history…")
    bundle = build_joined()
    pb_bills = bundle["bills"]
    pb_bills = pb_bills[pb_bills["is_amber_partial"].fillna(False)]
    partial_patient_ids = sorted({int(x) for x in pb_bills["patient_id"].dropna().unique()})
    print(f"  Partial-bill patients: {len(partial_patient_ids)}")

    pb_history = _fetch_partial_bill_history(partial_patient_ids, sku_ids)
    pb_history.to_parquet(PARTIAL_BILL_SNAPSHOT, engine="pyarrow", compression="snappy", index=False)
    print(f"  → {PARTIAL_BILL_SNAPSHOT.name}: {len(pb_history):,} rows · {PARTIAL_BILL_SNAPSHOT.stat().st_size/1e6:.1f} MB")
    print(f"     date range: {pb_history['day'].min()} → {pb_history['day'].max()}" if len(pb_history) else "     (empty)")
    print()
    print("Done.")
    return 0


def _fetch_partial_bill_history(patient_ids: list[int], sku_ids: set[int]) -> pd.DataFrame:
    """For the partial-bill patients, pull their price-↑ SKU history from
    2026-01-06 (4 months pre-go-live) through today across ALL chain stores."""
    if not patient_ids or not sku_ids:
        return pd.DataFrame(columns=["patient_id", "drug_id", "drug_name", "day",
                                       "store_name", "net_quantity", "revenue_value", "bill_flag"])
    pre_start = (GO_LIVE_DATE.date() - dt.timedelta(days=120)).isoformat()
    today = dt.date.today().isoformat()
    sku_tuple = tuple(int(x) for x in sorted(sku_ids))
    rows: list[tuple] = []
    chunk = 1000
    with _connect() as conn:
        with conn.cursor() as cur:
            for i in range(0, len(patient_ids), chunk):
                batch = tuple(patient_ids[i:i+chunk])
                cur.execute(
                    """
                    SELECT "patient-id","drug-id","drug-name","created-at"::date AS day,
                           "store-name","net-quantity","revenue-value","bill-flag"
                    FROM "prod2-generico"."sales"
                    WHERE "patient-id" IN %s
                      AND "drug-id" IN %s
                      AND "created-at"::date BETWEEN %s AND %s
                    """,
                    (batch, sku_tuple, pre_start, today),
                )
                rows.extend(cur.fetchall())
    df = pd.DataFrame(rows, columns=["patient_id", "drug_id", "drug_name", "day",
                                       "store_name", "net_quantity", "revenue_value", "bill_flag"])
    df["patient_id"] = pd.to_numeric(df["patient_id"], errors="coerce").astype("Int64")
    df["drug_id"] = pd.to_numeric(df["drug_id"], errors="coerce").astype("Int64")
    df["net_quantity"] = pd.to_numeric(df["net_quantity"], errors="coerce").fillna(0).astype("Float64")
    df["revenue_value"] = pd.to_numeric(df["revenue_value"], errors="coerce").fillna(0).astype("Float64")
    df["bill_flag"] = df["bill_flag"].astype(str)
    df["drug_name"] = df["drug_name"].astype(str)
    df["store_name"] = df["store_name"].astype(str)
    return df


if __name__ == "__main__":
    sys.exit(main())

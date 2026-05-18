"""Refresh the local Parquet snapshot of Redshift sales used by the dashboard.

Run this locally (with VPN reachable to redshift.internal) any time you want
the deployed Streamlit Cloud version to reflect newer data. Then commit
`data_snapshots/sales.parquet` and push.

    .venv/bin/python refresh_snapshots.py
"""

from __future__ import annotations

import datetime as dt
import os
import sys
from pathlib import Path

import pandas as pd
import psycopg2

# Reuse constants from data.py so we never drift.
from data import (
    PROJECT_DIR,
    QUANT_BASELINE_START,
    SALES_SNAPSHOT,
    SNAPSHOTS_DIR,
    STORE_MAP,
    STORE_PAIRS,
    _connect,
)

# Columns the dashboard needs (superset of fetch_sales + fetch_quant_sales cols).
_SNAPSHOT_COLS = [
    "store-name", "bill-id", "serial", "patient-id", "drug-id", "drug-name",
    "net-quantity", "quantity", "revenue-value", "promo-discount",
    "purchase-rate", "rate", "mrp", "promo-code", "bill-flag",
    "assortment-classification-id", "created-at",
]


def main() -> int:
    SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)

    survey_stores = set(STORE_MAP.values())
    pair_stores = {p["test"] for p in STORE_PAIRS} | {p["control"] for p in STORE_PAIRS}
    stores = tuple(sorted(survey_stores | pair_stores))

    date_from = QUANT_BASELINE_START.isoformat()
    date_to = dt.date.today().isoformat()

    print(f"Querying Redshift…")
    print(f"  stores ({len(stores)}): {', '.join(stores)}")
    print(f"  range : {date_from} → {date_to}")

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

    df.to_parquet(SALES_SNAPSHOT, engine="pyarrow", compression="snappy", index=False)

    size_mb = SALES_SNAPSHOT.stat().st_size / 1e6
    print()
    print(f"Wrote {SALES_SNAPSHOT}")
    print(f"  rows         : {len(df):,}")
    print(f"  unique stores: {df['store_name'].nunique()}")
    print(f"  date range   : {df['created_at'].min().date()} → {df['created_at'].max().date()}")
    print(f"  file size    : {size_mb:.1f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())

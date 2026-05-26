"""Data layer for the price-change A/B dashboard.

Loads the survey CSV, the SKU price-change CSV, and the user-corrected
unmatched-bills CSV; pulls sales rows from Redshift for the survey serials;
joins them into one bill-level DataFrame.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Iterable

import pandas as pd
import psycopg2
import streamlit as st
from dotenv import load_dotenv

# ----- constants -------------------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent


def _latest(pattern: str) -> Path:
    matches = sorted(PROJECT_DIR.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    if not matches:
        raise FileNotFoundError(f"No file matching {pattern!r} in {PROJECT_DIR}")
    return matches[0]


def _all_matching(pattern: str) -> list[Path]:
    """All matching files, oldest → newest by mtime."""
    return sorted(PROJECT_DIR.glob(pattern), key=lambda p: p.stat().st_mtime)


SURVEY_CSVS = _all_matching("Customer Behaviour Survey Replies*.csv")
SKU_CSV = _latest("Final Price Change Proposal*.csv")
OVERRIDES_CSV = PROJECT_DIR / "unmatched_numeric_bills.csv"

# Local Parquet snapshots — used for offline runs (Streamlit Cloud has no VPN
# access). Refresh via `python refresh_snapshots.py`.
SNAPSHOTS_DIR = PROJECT_DIR / "data_snapshots"
SALES_SNAPSHOT = SNAPSHOTS_DIR / "sales.parquet"           # line-level, 14 stores
STORE_TOTALS_SNAPSHOT = SNAPSHOTS_DIR / "store_totals.parquet"   # per-store-per-day-per-bill-set KPI totals, ALL chain stores
PATIENTS_SNAPSHOT = SNAPSHOTS_DIR / "patients_first_seen.parquet"  # patient_id -> first_seen_date
PARTIAL_BILL_SNAPSHOT = SNAPSHOTS_DIR / "partial_bill_history.parquet"  # partial-bill patients' price-↑ SKU history, all stores
SKU_CATEGORY_SNAPSHOT = SNAPSHOTS_DIR / "sku_category.parquet"  # drug_id → category lookup for price-↑ SKUs
COHORT_HISTORY_SNAPSHOT = SNAPSHOTS_DIR / "cohort_chronic_history.parquet"  # 6mo all-store history of chronic price-↑ SKUs for the pilot-post cohort

COHORT_HISTORY_START = pd.Timestamp("2025-11-06").date()  # 6 months pre-go-live
REFILL_GRACE_DAYS = 7  # grace before flagging a patient overdue
SAME_REFILL_WINDOW_DAYS = 7  # purchases of the same drug within this many days
                              # are treated as one refill event (avoids 1-2 day
                              # "cycles" from same-refill top-ups skewing the
                              # cycle estimate for chronic patients).
MIN_PLAUSIBLE_CYCLE_DAYS = 7  # cycle estimates shorter than this are flagged
                               # as no_estimate (chronic refills can't really be
                               # weekly or shorter — the data is noise).

# Known surveyor-typo prefixes — applied after Z-strip + uppercase in serial normalization.
PREFIX_FIXES = {"KEWT": "KRWT"}

GO_LIVE_DATE = pd.Timestamp("2026-05-06")
COUPON_PREFIX = "GENRTN"

# ----- Quant Insight: pilot/non-pilot A/B test ---------------------------------
# Pilot = stores where the price increase was applied. Non-pilot = paired control.
# (Earlier iterations used "test"/"control" labels; 3 pairs were re-labelled when
# the user clarified which side was actually piloted.)
STORE_PAIRS = [
    {"pair": 1, "city": "Mumbai", "pilot": "Colaba",                      "non_pilot": "Pantnagar"},
    {"pair": 2, "city": "Mumbai", "pilot": "Khanda Colony",               "non_pilot": "Virar West"},
    {"pair": 3, "city": "Mumbai", "pilot": "Bhiwandi Dhamankar Naka",     "non_pilot": "Dahisar"},
    {"pair": 4, "city": "Mumbai", "pilot": "Khar West",                   "non_pilot": "Dombivali"},
    {"pair": 5, "city": "Mumbai", "pilot": "Goregaon West S.V. Road",     "non_pilot": "Mulund West-Sarvodaya Nagar"},
    {"pair": 6, "city": "Mumbai", "pilot": "Sanpada",                     "non_pilot": "Airoli"},
]
PILOT_STORES = tuple(p["pilot"] for p in STORE_PAIRS)
PAIRED_NON_PILOT_STORES = tuple(p["non_pilot"] for p in STORE_PAIRS)
QUANT_PRE_START = pd.Timestamp("2026-04-20").date()
QUANT_PRE_END = pd.Timestamp("2026-05-05").date()
QUANT_POST_START = pd.Timestamp("2026-05-06").date()
QUANT_POST_END = pd.Timestamp("2026-05-26").date()
QUANT_BASELINE_START = pd.Timestamp("2026-01-01").date()
QUANT_BASELINE_END = pd.Timestamp("2026-04-19").date()
ASSORTMENT_GENERIC = (3, 4)   # Generic + Generic-Speciality
ASSORTMENT_ETHICAL = (1, 2)   # Ethical + Ethical-Speciality

# RAG classification — outcome-string patterns (case-insensitive substring match).
# Conservative: only explicit walkaway / partial phrases. Everything else defaults GREEN.
RED_PATTERNS = ["bana hi nahin", "chale gaye", "didn't buy", "didnt buy", "left without"]
PARTIAL_PATTERNS = ["partial bana", "item kam"]
RAG_COLORS = {"green": "#2ECC71", "amber": "#F39C12", "red": "#E74C3C"}
RAG_EMOJI = {"green": "🟢", "amber": "🟡", "red": "🔴"}


def classify_rag(outcome: str, used_genrtn: bool) -> tuple[str, bool, bool]:
    """Return (rag, is_amber_partial, is_amber_genrtn) for one bill.

    Precedence: RED > AMBER (partial OR genrtn, both flags can be True) > GREEN.
    GENRTN trumps GREEN — if the till applied the coupon, the customer pushed back
    enough that the bill is amber regardless of what the surveyor wrote.
    """
    o = (outcome or "").strip().lower()
    if any(p in o for p in RED_PATTERNS):
        return "red", False, False
    is_partial = any(p in o for p in PARTIAL_PATTERNS)
    if is_partial or used_genrtn:
        return "amber", is_partial, bool(used_genrtn)
    return "green", False, False

# survey "Store Name" -> sales "store-name"
STORE_MAP = {
    "Bandra Hill Road": "Bandra",
    "Colaba": "Colaba",
    "Dhamankar Naka": "Bhiwandi Dhamankar Naka",
    "Goregaon": "Goregaon",
    "Goregaon W SV Road": "Goregaon West S.V. Road",
    "Khanda Colony": "Khanda Colony",
    "Khar": "Khar West",  # assumed; flagged for confirmation in UI footer
    "Sanpada": "Sanpada",
}

# Survey column indices (verified from header inspection)
COL_SUBMISSION_ID = 0
COL_SUBMITTED_AT = 2
COL_STORE = 3
COL_STAFF_INFORMED = 4
COL_CHECKED_BILL = 5
COL_SPOKE_PRICE = 6
COL_WHAT_SAID = 7
COL_NONVERBAL = 8
COL_REACTION_TYPE = 9
COL_OUTCOME = 10
COL_BILL = 11
COL_NOTES = 12

SERIAL_RE = re.compile(r"^[A-Z]+-\d+$")

# ----- Redshift connection ---------------------------------------------------

def _env() -> dict:
    load_dotenv(PROJECT_DIR / ".env")
    return {
        "host": os.environ["REDSHIFT_HOST"],
        "port": int(os.environ["REDSHIFT_PORT"]),
        "dbname": os.environ["REDSHIFT_DB"],
        "user": os.environ["REDSHIFT_USER"],
        "password": os.environ["REDSHIFT_PASSWORD"],
    }


def _connect():
    return psycopg2.connect(connect_timeout=15, **_env())


# ----- numeric overrides -----------------------------------------------------

@st.cache_data(ttl=1800, show_spinner=False)
def load_numeric_overrides() -> pd.DataFrame:
    """User-corrected mapping from (Submission ID, numeric bill value) to a
    full serial like CLBA-153679, or `<STORE>-0` for walk-aways.
    """
    if not OVERRIDES_CSV.exists():
        return pd.DataFrame(columns=["submission_id", "bill_numeric", "original_bill_id"])
    df = pd.read_csv(OVERRIDES_CSV, dtype=str).fillna("")
    df = df.rename(columns={
        "Submission ID": "submission_id",
        "Bill Number (numeric)": "bill_numeric",
        "Original Bill ID": "original_bill_id",
    })
    return df[["submission_id", "bill_numeric", "original_bill_id"]].copy()


# ----- survey ----------------------------------------------------------------

def _normalize_alpha_serial(raw: str) -> str | None:
    """Strip whitespace, replace `_` with `-`, strip leading Z/z, uppercase,
    apply known-typo prefix fixes. If the raw value contains multiple bill IDs
    separated by `/`, `,` or `;`, return the first that normalizes successfully.
    Returns the normalized serial (e.g. `Zgwsv-67046` -> `GWSV-67046`,
    `ZBDHN_151188` -> `BDHN-151188`, `ZKEWT-195411` -> `KRWT-195411`) or None
    if it's not a STORE-NNN shape.
    """
    if not raw: return None
    # Split on common multi-bill separators ("X / Y", "X,Y", "X; Y") — take first valid.
    if any(sep in raw for sep in ("/", ",", ";")):
        import re as _re
        for tok in _re.split(r"[/,;]", raw):
            result = _normalize_alpha_serial(tok)
            if result is not None: return result
        return None

    s = raw.strip().replace("_", "-")  # surveyors sometimes type `STORE_N` instead of `STORE-N`
    if not s or s.isdigit():
        return None
    if s[:1] in ("Z", "z"):
        s = s[1:]
    s = s.strip().upper().replace(" ", "")
    if not SERIAL_RE.match(s):
        return None
    prefix, _, rest = s.partition("-")
    if prefix in PREFIX_FIXES:
        s = f"{PREFIX_FIXES[prefix]}-{rest}"
    return s


@st.cache_data(ttl=1800, show_spinner=False)
def load_survey() -> pd.DataFrame:
    """Concat ALL survey CSVs (incremental exports), dedup by Submission ID
    keeping the newest file's version, then filter to >= go-live, normalize
    stores, derive serial_norm + bill_kind."""
    if not SURVEY_CSVS:
        raise FileNotFoundError("No 'Customer Behaviour Survey Replies*.csv' files found")
    frames = []
    expected_cols = None
    for path in SURVEY_CSVS:
        r = pd.read_csv(path, header=0, dtype=str).fillna("")
        if expected_cols is None:
            expected_cols = len(r.columns)
        elif len(r.columns) != expected_cols:
            raise ValueError(
                f"Column count mismatch in {path.name}: got {len(r.columns)}, expected {expected_cols}"
            )
        # Rename to positional names so concat aligns even if headers differ slightly.
        r.columns = [f"c{i}" for i in range(len(r.columns))]
        frames.append(r)
    raw = pd.concat(frames, ignore_index=True)
    # Dedup by Submission ID (column 0). Files were read oldest → newest, so
    # `keep="last"` keeps the newest version in case of corrections.
    raw = raw.drop_duplicates(subset=["c0"], keep="last").reset_index(drop=True)
    cols = list(raw.columns)
    df = pd.DataFrame({
        "submission_id": raw[cols[COL_SUBMISSION_ID]],
        "submitted_at": pd.to_datetime(raw[cols[COL_SUBMITTED_AT]], errors="coerce"),
        "store_survey": raw[cols[COL_STORE]],
        "staff_informed": raw[cols[COL_STAFF_INFORMED]],
        "checked_bill": raw[cols[COL_CHECKED_BILL]],
        "spoke_price": raw[cols[COL_SPOKE_PRICE]],
        "what_said": raw[cols[COL_WHAT_SAID]],
        "nonverbal": raw[cols[COL_NONVERBAL]],
        "reaction_type": raw[cols[COL_REACTION_TYPE]],
        "outcome": raw[cols[COL_OUTCOME]],
        "bill_raw": raw[cols[COL_BILL]].fillna("").str.strip(),
        "notes": raw[cols[COL_NOTES]],
    })

    df["submitted_date"] = df["submitted_at"].dt.date
    df = df[df["submitted_at"] >= GO_LIVE_DATE].copy()

    df["store"] = df["store_survey"].map(STORE_MAP).fillna(df["store_survey"])

    overrides = load_numeric_overrides()
    ovr_map = dict(zip(
        list(zip(overrides["submission_id"], overrides["bill_numeric"])),
        overrides["original_bill_id"],
    ))

    is_walkaway_outcome = df["outcome"].str.contains("Bana hi nahin", case=False, na=False)

    serial_norm: list[str | None] = []
    bill_kind: list[str] = []
    for (_, r), is_walk in zip(df.iterrows(), is_walkaway_outcome):
        raw_bill = r["bill_raw"]
        sub_id = r["submission_id"]
        store_raw = r["store_survey"]

        # Try to resolve a serial first (works regardless of walkaway status).
        resolved: str | None = None
        if raw_bill and not raw_bill.isdigit():
            resolved = _normalize_alpha_serial(raw_bill)
        elif raw_bill and raw_bill.isdigit():
            # Goregaon-specific: surveyor sometimes drops the GORE- prefix
            # (e.g. "394502" instead of "GORE-394502"). len ≤ 7 keeps us safe
            # from 10-digit phone numbers.
            if store_raw == "Goregaon" and len(raw_bill) <= 7:
                cand = f"GORE-{raw_bill}"
                if SERIAL_RE.match(cand): resolved = cand
            # Defensive numeric branch via the override CSV.
            if resolved is None:
                ovr = ovr_map.get((sub_id, raw_bill))
                if ovr and not ovr.endswith("-0"):
                    cand = ovr.strip().upper()
                    if SERIAL_RE.match(cand):
                        pre, _, rest = cand.partition("-")
                        resolved = f"{PREFIX_FIXES.get(pre, pre)}-{rest}"

        if is_walk:
            # Walkaway classification trumps everything — bill never made it.
            serial_norm.append(None)
            bill_kind.append("walkaway")
        elif resolved is not None:
            serial_norm.append(resolved)
            bill_kind.append("serial")
        else:
            # Bill was made (per outcome) but surveyor didn't provide a usable ID.
            serial_norm.append(None)
            bill_kind.append("bill_created_no_id")

    df["serial_norm"] = serial_norm
    df["bill_kind"] = bill_kind
    df["is_walkaway"] = is_walkaway_outcome

    # Dedup duplicate survey entries on same serial (keep latest)
    serial_rows = df[df["bill_kind"] == "serial"].copy()
    other_rows = df[df["bill_kind"] != "serial"].copy()
    serial_rows = serial_rows.sort_values("submitted_at").drop_duplicates(
        subset=["serial_norm"], keep="last"
    )
    df = pd.concat([serial_rows, other_rows], ignore_index=True)
    return df


CLEAN_BILLS_CSV = PROJECT_DIR / "clean_bill_ids.csv"


def write_clean_bill_ids_csv(survey_df: pd.DataFrame) -> Path:
    """Write a human-readable reference sheet showing what each surveyor bill
    value got corrected to. Regenerated on every dashboard render."""
    corrected = survey_df.apply(
        lambda r: r["serial_norm"] if pd.notna(r["serial_norm"]) and r["serial_norm"]
        else {
            "walkaway": "(walkaway)",
            "bill_created_no_id": "(bill made, no ID)",
            "serial": "",  # shouldn't hit this branch
        }.get(r["bill_kind"], "(junk)"),
        axis=1,
    )
    out = pd.DataFrame({
        "Submission ID": survey_df["submission_id"],
        "Submitted at": survey_df["submitted_at"],
        "Store Name (raw)": survey_df["store_survey"],
        "Store Name (mapped)": survey_df["store"],
        "Bill ID (raw)": survey_df["bill_raw"],
        "Bill ID (corrected)": corrected,
        "bill_kind": survey_df["bill_kind"],
        "Outcome": survey_df["outcome"],
        "Notes": survey_df["notes"],
    }).sort_values("Submitted at", ascending=False)
    out.to_csv(CLEAN_BILLS_CSV, index=False)
    return CLEAN_BILLS_CSV


# ----- SKUs ------------------------------------------------------------------

@st.cache_data(ttl=1800, show_spinner=False)
def load_skus() -> pd.DataFrame:
    df = pd.read_csv(SKU_CSV)
    df = df.rename(columns={
        "drug_id": "drug_id",
        "Drug name": "drug_name",
        "Selling Price": "new_price",
        "Assortment Classification": "assortment",
        "Old Price": "old_price",
        "Drug Type": "drug_type",
    })
    df["drug_id"] = pd.to_numeric(df["drug_id"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["drug_id"]).copy()
    df["drug_id"] = df["drug_id"].astype(int)
    return df[["drug_id", "drug_name", "old_price", "new_price", "drug_type"]]


# ----- sales -----------------------------------------------------------------

_SALES_COLS = [
    "bill-id", "serial", "store-name", "drug-id", "drug-name",
    "created-at", "quantity", "rate", "mrp", "revenue-value",
    "promo-code", "patient-id",
]


def _fetch_sales_chunk(cur, serials_chunk: tuple[str, ...]) -> list[tuple]:
    cur.execute(
        f"""
        SELECT {", ".join(f'"{c}"' for c in _SALES_COLS)}
        FROM "prod2-generico"."sales"
        WHERE "created-at"::date >= DATE %s
          AND "serial" IN %s
        """,
        (GO_LIVE_DATE.date().isoformat(), serials_chunk),
    )
    return cur.fetchall()


@st.cache_data(ttl=1800, show_spinner=False)
def _load_sales_snapshot() -> pd.DataFrame | None:
    """Load the local Parquet snapshot, if present. Returns None when missing
    (caller then falls back to live Redshift)."""
    if not SALES_SNAPSHOT.exists():
        return None
    df = pd.read_parquet(SALES_SNAPSHOT)
    # Ensure dtypes match what the Redshift path would produce.
    if "created_at" in df.columns:
        df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")
    return df


@st.cache_data(ttl=1800, show_spinner=False)
def _load_store_totals_snapshot() -> pd.DataFrame | None:
    """Pre-aggregated per-store-per-day-per-bill-set KPI totals for ALL chain
    stores. Used by the Summary tab to compare pilot vs all-other stores
    without pulling line-level data for ~195 stores."""
    if not STORE_TOTALS_SNAPSHOT.exists():
        return None
    df = pd.read_parquet(STORE_TOTALS_SNAPSHOT)
    if "day" in df.columns:
        df["day"] = pd.to_datetime(df["day"], errors="coerce").dt.date
    return df


@st.cache_data(ttl=1800, show_spinner=False)
def _load_patients_snapshot() -> dict | None:
    """Load patient_id -> first-ever bill date lookup. Used for new-patient KPI."""
    if not PATIENTS_SNAPSHOT.exists():
        return None
    df = pd.read_parquet(PATIENTS_SNAPSHOT)
    return dict(zip(df["patient_id"].astype(int), pd.to_datetime(df["first_seen"]).dt.date))


def aggregate_totals(
    totals: pd.DataFrame,
    stores: tuple[str, ...],
    date_from,
    date_to,
    bill_filter: str = "with_pi",
) -> dict:
    """Sum pre-aggregated per-store-per-day totals for the given store set and
    date window. Returns the same KPI dict shape as `compute_kpis` (for the
    metrics that can be computed from aggregates — patient-related KPIs and
    mix ratios are approximated since aggregates don't preserve patient-level
    detail across stores)."""
    if totals is None or totals.empty:
        return {}
    d_from = pd.Timestamp(date_from).date()
    d_to = pd.Timestamp(date_to).date()
    sub = totals[
        totals["store_name"].isin(stores)
        & (totals["day"] >= d_from)
        & (totals["day"] <= d_to)
        & (totals["bill_filter"] == bill_filter)
    ]
    if sub.empty: return {}
    sum_cols = [
        "revenue", "cogs", "promo_discount_total", "bill_count", "unique_customers",
        "generic_units", "ethical_units", "total_units", "generic_revenue",
        "total_revenue", "promo_bills", "promo_discount_promo",
        "return_units", "total_quantity", "total_units_pi", "repeat_patient_count",
        "new_patient_count",
        "pi_units_chronic", "pi_units_acute", "pi_units_therapy_cycle",
    ]
    out = {c: float(sub[c].sum()) if c in sub.columns else 0.0 for c in sum_cols}
    out["gm"] = out["revenue"] - out["cogs"] - out["promo_discount_total"]
    out["gm_pct"] = (out["gm"] / out["revenue"]) if out["revenue"] else 0.0
    out["aov"] = (out["revenue"] / out["bill_count"]) if out["bill_count"] else 0.0
    out["aom"] = (out["gm"] / out["bill_count"]) if out["bill_count"] else 0.0
    out["acm"] = (out["gm"] / out["unique_customers"]) if out["unique_customers"] else 0.0
    out["frequency"] = (out["bill_count"] / out["unique_customers"]) if out["unique_customers"] else 0.0
    out["items_per_bill"] = (out["total_quantity"] / out["bill_count"]) if out["bill_count"] else 0.0
    out["revenue_per_patient"] = (out["revenue"] / out["unique_customers"]) if out["unique_customers"] else 0.0
    out["generic_mix_units_pct"] = (out["generic_units"] / out["total_units"]) if out["total_units"] else 0.0
    out["generic_mix_revenue_pct"] = (out["generic_revenue"] / out["total_revenue"]) if out["total_revenue"] else 0.0
    out["promo_usage_rate"] = (out["promo_bills"] / out["bill_count"]) if out["bill_count"] else 0.0
    out["promo_share_revenue"] = (out["promo_discount_promo"] / out["total_revenue"]) if out["total_revenue"] else 0.0
    out["return_rate"] = (out["return_units"] / out["total_units"]) if out["total_units"] else 0.0
    # Note: unique_customers is summed across stores, so a customer who shopped
    # at 2 stores is counted twice. That's a known limitation of per-store
    # pre-aggregation; documented in the dashboard caption.
    return out


@st.cache_data(ttl=1800, show_spinner="Querying Redshift…")
def fetch_sales(serials: tuple[str, ...]) -> pd.DataFrame:
    """Snapshot-first: filter local Parquet for the given serials.
    Falls back to live Redshift if snapshot missing."""
    if not serials:
        return pd.DataFrame(columns=[c.replace("-", "_") for c in _SALES_COLS])
    snap = _load_sales_snapshot()
    if snap is not None:
        needed = [c.replace("-", "_") for c in _SALES_COLS]
        cols = [c for c in needed if c in snap.columns]
        df = snap.loc[
            snap["serial"].isin(serials)
            & (snap["created_at"].dt.date >= GO_LIVE_DATE.date()),
            cols,
        ].copy()
        # Backfill any columns missing from the snapshot with NaN.
        for c in needed:
            if c not in df.columns: df[c] = pd.NA
        return df

    rows: list[tuple] = []
    chunk_size = 1000
    with _connect() as conn:
        with conn.cursor() as cur:
            for i in range(0, len(serials), chunk_size):
                chunk = serials[i:i + chunk_size]
                rows.extend(_fetch_sales_chunk(cur, chunk))
    df = pd.DataFrame(rows, columns=_SALES_COLS)
    df = df.rename(columns={c: c.replace("-", "_") for c in df.columns})
    df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")
    df["drug_id"] = pd.to_numeric(df["drug_id"], errors="coerce").astype("Int64")
    df["revenue_value"] = pd.to_numeric(df["revenue_value"], errors="coerce").fillna(0)
    df["promo_code"] = df["promo_code"].fillna("").astype(str)
    return df


# ----- joined view -----------------------------------------------------------

def _raw_prefix(raw: str) -> str:
    s = (raw or "").strip()
    if s[:1] in ("Z", "z"):
        s = s[1:]
    return s.split("-")[0].upper()


@st.cache_data(ttl=1800, show_spinner=False)
def build_joined() -> dict:
    """Returns a dict with:
       - survey: survey DataFrame (post-go-live)
       - skus:   SKU DataFrame
       - sales:  raw line-level sales for survey serials
       - bills:  bill-level DataFrame: one row per (serial) with survey + sales aggregates
       - missing_serials: list of serial_norms that didn't match any sales row
       - stats: dict of counters for the funnel widget and freshness expander
    """
    survey = load_survey()
    skus = load_skus()
    sku_ids = set(skus["drug_id"].astype(int))

    serial_set = sorted(survey.loc[survey["bill_kind"] == "serial", "serial_norm"].dropna().unique())
    sales = fetch_sales(tuple(serial_set))

    matched_serials = set(sales["serial"].unique())
    missing_serials = [s for s in serial_set if s not in matched_serials]

    typo_fixed = int(
        survey["bill_raw"].apply(lambda v: _raw_prefix(v) in PREFIX_FIXES).sum()
    )

    # Per-line: is the drug in the price-increased SKU list?
    sales["is_price_increased"] = sales["drug_id"].astype("Int64").isin(sku_ids)

    # Per-bill aggregates (one row per serial, since survey is one row per serial)
    if not sales.empty:
        bill_agg = sales.groupby("serial").agg(
            bill_id=("bill_id", "first"),
            store_sales=("store_name", "first"),
            bill_date=("created_at", "min"),
            n_lines=("drug_id", "size"),
            n_price_increase_lines=("is_price_increased", "sum"),
            revenue=("revenue_value", "sum"),
            promo_code_any=("promo_code", lambda s: ",".join(sorted({x for x in s if x}))),
            patient_id=("patient_id", "first"),
        ).reset_index()
        bill_agg["has_price_increased_sku"] = bill_agg["n_price_increase_lines"] > 0
        bill_agg["used_genrtn"] = bill_agg["promo_code_any"].str.upper().str.contains(
            COUPON_PREFIX, na=False
        )
    else:
        bill_agg = pd.DataFrame(columns=[
            "serial", "bill_id", "store_sales", "bill_date", "n_lines",
            "n_price_increase_lines", "revenue", "promo_code_any", "patient_id",
            "has_price_increased_sku", "used_genrtn",
        ])

    bills = survey[survey["bill_kind"] == "serial"].merge(
        bill_agg, left_on="serial_norm", right_on="serial", how="left"
    )

    # ----- RAG classification -------------------------------------------------
    # 1) Bring used_genrtn into the survey-level frame (False for non-serial rows).
    used_genrtn_by_serial = bill_agg.set_index("serial")["used_genrtn"].to_dict() if not bill_agg.empty else {}
    survey["used_genrtn_for_rag"] = survey["serial_norm"].map(
        lambda s: bool(used_genrtn_by_serial.get(s, False))
    )

    # 2) Classify each survey row (bill_kind=walkaway forces RED via outcome rule).
    rag_tuples = [
        classify_rag(o, g) for o, g in zip(survey["outcome"], survey["used_genrtn_for_rag"])
    ]
    survey["rag"] = [t[0] for t in rag_tuples]
    survey["is_amber_partial"] = [t[1] for t in rag_tuples]
    survey["is_amber_genrtn"] = [t[2] for t in rag_tuples]

    # 3) Mirror the same columns onto the per-bill view.
    bills["used_genrtn_for_rag"] = bills["used_genrtn"].fillna(False).astype(bool)
    rag_b = [
        classify_rag(o, g) for o, g in zip(bills["outcome"], bills["used_genrtn_for_rag"])
    ]
    bills["rag"] = [t[0] for t in rag_b]
    bills["is_amber_partial"] = [t[1] for t in rag_b]
    bills["is_amber_genrtn"] = [t[2] for t in rag_b]

    # ----- outcome audit ------------------------------------------------------
    canonical_green = "No change. Jaisa ban na tha bana."
    canonical_partial = "Partial bana. Customer nein item kam kiye"
    canonical_walkaway = "Bana hi nahin. Customer chale gaye."

    is_canon_green = survey["outcome"].eq(canonical_green)
    is_canon_partial = survey["outcome"].eq(canonical_partial)
    is_canon_walk = survey["outcome"].eq(canonical_walkaway)
    is_freetext = ~(is_canon_green | is_canon_partial | is_canon_walk)

    freetext_to_red = survey.loc[is_freetext & (survey["rag"] == "red"), "outcome"]
    freetext_to_partial = survey.loc[
        is_freetext & survey["is_amber_partial"], "outcome"
    ]
    freetext_to_green = survey.loc[is_freetext & (survey["rag"] == "green"), "outcome"]

    # GENRTN-vs-surveyor conflicts: surveyor wrote canonical "No change" but till applied GENRTN
    n_genrtn_overrides_no_change = int(
        (is_canon_green & survey["used_genrtn_for_rag"]).sum()
    )

    outcome_audit = {
        "n_canonical_green": int(is_canon_green.sum()),
        "n_canonical_partial": int(is_canon_partial.sum()),
        "n_canonical_walkaway": int(is_canon_walk.sum()),
        "n_freetext_total": int(is_freetext.sum()),
        "n_freetext_to_red": int(len(freetext_to_red)),
        "n_freetext_to_partial": int(len(freetext_to_partial)),
        "n_freetext_to_green": int(len(freetext_to_green)),
        "top_freetext_in_green": freetext_to_green.value_counts().head(20).to_dict(),
        "n_genrtn_overrides_no_change": n_genrtn_overrides_no_change,
    }

    # ----- stats --------------------------------------------------------------
    n_serial = int((survey["bill_kind"] == "serial").sum())
    n_no_id = int((survey["bill_kind"] == "bill_created_no_id").sum())
    n_walk = int((survey["bill_kind"] == "walkaway").sum())
    n_joined = int(bills["bill_id"].notna().sum()) if not bills.empty else 0
    stats = {
        "survey_rows": int(len(survey)),
        "serial": n_serial,
        "joined_to_sales": n_joined,
        "sync_lag": n_serial - n_joined,
        "bill_created_no_id": n_no_id,
        "walkaway": n_walk,
        "typo_fixed_kewt_to_krwt": typo_fixed,
        "rag_green": int((survey["rag"] == "green").sum()),
        "rag_amber": int((survey["rag"] == "amber").sum()),
        "rag_amber_partial": int(survey["is_amber_partial"].sum()),
        "rag_amber_genrtn": int(survey["is_amber_genrtn"].sum()),
        "rag_amber_both": int((survey["is_amber_partial"] & survey["is_amber_genrtn"]).sum()),
        "rag_red": int((survey["rag"] == "red").sum()),
        "outcome_audit": outcome_audit,
    }

    return {
        "survey": survey,
        "skus": skus,
        "sales": sales,
        "bills": bills,
        "missing_serials": missing_serials,
        "stats": stats,
    }


# ============================================================================
# Quant Insight — pair-based A/B test data layer
# ============================================================================

_QUANT_SALES_COLS = [
    "store-name", "bill-id", "patient-id", "drug-id",
    "net-quantity", "revenue-value", "promo-discount", "purchase-rate",
    "promo-code", "bill-flag", "assortment-classification-id", "created-at",
]


@st.cache_data(ttl=1800, show_spinner="Querying Redshift (Quant)…")
def fetch_quant_sales(stores: tuple[str, ...], date_from: str, date_to: str) -> pd.DataFrame:
    """Snapshot-first: filter local Parquet for the given stores & date range.
    Falls back to live Redshift if snapshot missing."""
    underscored = [c.replace("-", "_") for c in _QUANT_SALES_COLS]
    if not stores:
        return pd.DataFrame(columns=underscored)

    snap = _load_sales_snapshot()
    if snap is not None:
        d_from = pd.Timestamp(date_from).date()
        d_to = pd.Timestamp(date_to).date()
        cols = [c for c in underscored if c in snap.columns]
        df = snap.loc[
            snap["store_name"].isin(stores)
            & (snap["created_at"].dt.date >= d_from)
            & (snap["created_at"].dt.date <= d_to),
            cols,
        ].copy()
        for c in underscored:
            if c not in df.columns: df[c] = pd.NA
        return df

    cols_sql = ", ".join(f'"{c}"' for c in _QUANT_SALES_COLS)
    with _connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT {cols_sql}
                FROM "prod2-generico"."sales"
                WHERE "store-name" IN %s
                  AND "created-at"::date BETWEEN %s AND %s
                """,
                (tuple(stores), date_from, date_to),
            )
            rows = cur.fetchall()
    df = pd.DataFrame(rows, columns=_QUANT_SALES_COLS)
    df = df.rename(columns={c: c.replace("-", "_") for c in df.columns})
    df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")
    for col in ["net_quantity", "revenue_value", "promo_discount", "purchase_rate"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    df["drug_id"] = pd.to_numeric(df["drug_id"], errors="coerce").astype("Int64")
    df["promo_code"] = df["promo_code"].fillna("").astype(str)
    df["bill_flag"] = df["bill_flag"].fillna("").astype(str)
    df["assortment_classification_id"] = pd.to_numeric(
        df["assortment_classification_id"], errors="coerce"
    ).astype("Int64")
    return df


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_patient_monthly_activity(
    patient_ids: tuple[int, ...],
    date_from: str,
    date_to: str,
) -> pd.DataFrame:
    """For the given patients, return one row per (patient_id, calendar_month)
    they have ≥1 gross bill in the 14-store sales snapshot between
    date_from and date_to.

    Pure derivation from `sales.parquet` — no Redshift call, works on
    Streamlit Cloud. Limitation: only sees activity at the 14 pilot+paired
    stores; a patient who shopped exclusively at other chain stores in a
    given month would be missed. Pilot patients overwhelmingly transact at
    pilot stores so this is a fine proxy.

    Returns DataFrame with columns: patient_id (int), month (YYYY-MM string)."""
    empty = pd.DataFrame(columns=["patient_id", "month"])
    if not patient_ids:
        return empty

    snap = _load_sales_snapshot()
    if snap is None:
        return empty

    d_from = pd.Timestamp(date_from).date()
    d_to = pd.Timestamp(date_to).date()
    ids = set(int(x) for x in patient_ids)
    df = snap.loc[
        (snap["bill_flag"] == "gross")
        & snap["patient_id"].notna()
        & snap["patient_id"].astype("Int64").isin(ids)
        & (snap["created_at"].dt.date >= d_from)
        & (snap["created_at"].dt.date <= d_to),
        ["patient_id", "created_at"],
    ].copy()
    if df.empty:
        return empty
    df["patient_id"] = df["patient_id"].astype(int)
    df["month"] = df["created_at"].dt.to_period("M").astype(str)
    return df[["patient_id", "month"]].drop_duplicates().reset_index(drop=True)


def compute_kpis(
    df: pd.DataFrame,
    sku_ids: set[int] | None = None,
    patient_first_seen: dict | None = None,
    bill_filter: str = "with_pi",
) -> dict:
    """Pure compute: given a sales slice, return a dict of all KPIs.
    Caller filters by store + date window upfront.

    `sku_ids` is the price-↑ SKU set. `bill_filter`:
      - `with_pi`    : keep bills with ≥1 price-↑ SKU line (default)
      - `without_pi` : keep bills with ZERO price-↑ SKU lines (control bills)
      - `all`        : no bill-level filter
    If `sku_ids` is None, bill_filter is ignored.

    `patient_first_seen` is an optional `{patient_id: first_seen_date}` lookup
    used to compute `new_patient_count` (patients whose first-ever bill in the
    chain falls inside this slice's date range).
    """
    empty = {k: 0.0 for k in [
        "revenue", "cogs", "promo_discount_total", "gm", "gm_pct",
        "bill_count", "unique_customers", "aom", "acm", "aov", "frequency",
        "generic_units", "ethical_units", "total_units",
        "generic_mix_units_pct", "generic_revenue", "total_revenue",
        "generic_mix_revenue_pct", "promo_bills", "promo_usage_rate",
        "promo_share_revenue", "return_units", "return_rate",
        "total_quantity", "total_units_pi", "repeat_patient_count",
        "new_patient_count", "items_per_bill", "revenue_per_patient",
        "pi_units_chronic", "pi_units_acute", "pi_units_therapy_cycle",
    ]}
    if df.empty: return empty

    # Bill-level filter using gross bills as the source of truth.
    if sku_ids is not None and bill_filter != "all":
        gross_all = df[df["bill_flag"] == "gross"]
        pi_bill_ids = set(gross_all.loc[gross_all["drug_id"].isin(sku_ids), "bill_id"].dropna().unique())
        if bill_filter == "with_pi":
            if not pi_bill_ids: return empty
            df = df[df["bill_id"].isin(pi_bill_ids)]
        elif bill_filter == "without_pi":
            all_bill_ids = set(gross_all["bill_id"].dropna().unique())
            non_pi_bill_ids = all_bill_ids - pi_bill_ids
            if not non_pi_bill_ids: return empty
            df = df[df["bill_id"].isin(non_pi_bill_ids)]

    gross = df[df["bill_flag"] == "gross"]
    ret = df[df["bill_flag"] == "return"]

    revenue_raw = float(gross["revenue_value"].sum())
    promo_discount_total = float(gross["promo_discount"].sum())
    revenue = revenue_raw - promo_discount_total
    cogs = float((gross["net_quantity"] * gross["purchase_rate"]).sum())
    gm = revenue - cogs
    gm_pct = (gm / revenue) if revenue else 0.0

    bill_count = int(gross["bill_id"].nunique())
    unique_customers = int(gross["patient_id"].dropna().nunique())
    aom = (gm / bill_count) if bill_count else 0.0
    acm = (gm / unique_customers) if unique_customers else 0.0
    aov = (revenue / bill_count) if bill_count else 0.0
    frequency = (bill_count / unique_customers) if unique_customers else 0.0

    is_generic = gross["assortment_classification_id"].isin(ASSORTMENT_GENERIC)
    is_ethical = gross["assortment_classification_id"].isin(ASSORTMENT_ETHICAL)
    generic_units = float(gross.loc[is_generic, "net_quantity"].sum())
    ethical_units = float(gross.loc[is_ethical, "net_quantity"].sum())
    total_units = float(gross["net_quantity"].sum())
    generic_mix_units_pct = (generic_units / total_units) if total_units else 0.0
    generic_revenue = float(gross.loc[is_generic, "revenue_value"].sum())
    generic_mix_revenue_pct = (generic_revenue / revenue_raw) if revenue_raw else 0.0

    promo_mask = gross["promo_code"].str.upper().str.startswith(COUPON_PREFIX)
    promo_bills = int(gross.loc[promo_mask, "bill_id"].nunique())
    promo_usage_rate = (promo_bills / bill_count) if bill_count else 0.0
    promo_discount_promo = float(gross.loc[promo_mask, "promo_discount"].sum())
    promo_share_revenue = (promo_discount_promo / revenue_raw) if revenue_raw else 0.0

    return_units = float(ret["net_quantity"].abs().sum())  # returns stored as negative qty
    return_rate = (return_units / total_units) if total_units else 0.0

    # New KPIs (Phase 5)
    total_quantity = total_units  # alias — "total quantity" is just total_units
    items_per_bill = (total_quantity / bill_count) if bill_count else 0.0
    revenue_per_patient = (revenue / unique_customers) if unique_customers else 0.0
    # Phase 8: units of price-↑ SKUs only (subset of total_units, on gross bills)
    total_units_pi = (
        float(gross.loc[gross["drug_id"].isin(sku_ids), "net_quantity"].sum())
        if sku_ids is not None else 0.0
    )
    # Phase 10: split price-↑ units by chronic / acute / therapy_cycle
    pi_units_chronic = pi_units_acute = pi_units_therapy_cycle = 0.0
    if sku_ids is not None:
        cat_lookup = _load_sku_category()
        if cat_lookup:
            pi_lines = gross[gross["drug_id"].isin(sku_ids)].copy()
            pi_lines["_cat"] = pi_lines["drug_id"].map(lambda d: cat_lookup.get(int(d), "unknown"))
            by_cat = pi_lines.groupby("_cat")["net_quantity"].sum()
            pi_units_chronic = float(by_cat.get("chronic", 0))
            pi_units_acute = float(by_cat.get("acute", 0))
            pi_units_therapy_cycle = float(by_cat.get("therapy cycle", 0))

    # Repeat patient = patient with ≥2 distinct bills inside this slice.
    bills_per_patient = (
        gross.dropna(subset=["patient_id"])
        .groupby("patient_id")["bill_id"].nunique()
    )
    repeat_patient_count = int((bills_per_patient >= 2).sum())

    # New patient = patient whose first-ever bill (per the patients lookup) falls
    # within this slice's [min, max] date range.
    new_patient_count = 0
    if patient_first_seen is not None and not gross.empty:
        date_min = gross["created_at"].dt.date.min()
        date_max = gross["created_at"].dt.date.max()
        patient_ids = gross["patient_id"].dropna().unique().tolist()
        new_patient_count = sum(
            1 for pid in patient_ids
            if (fs := patient_first_seen.get(int(pid))) is not None
            and date_min <= fs <= date_max
        )

    return {
        "revenue": revenue, "cogs": cogs, "promo_discount_total": promo_discount_total,
        "gm": gm, "gm_pct": gm_pct,
        "bill_count": bill_count, "unique_customers": unique_customers,
        "aom": aom, "acm": acm, "aov": aov, "frequency": frequency,
        "generic_units": generic_units, "ethical_units": ethical_units,
        "total_units": total_units, "generic_mix_units_pct": generic_mix_units_pct,
        "generic_revenue": generic_revenue, "total_revenue": revenue_raw,
        "generic_mix_revenue_pct": generic_mix_revenue_pct,
        "promo_bills": promo_bills, "promo_usage_rate": promo_usage_rate,
        "promo_share_revenue": promo_share_revenue,
        "return_units": return_units, "return_rate": return_rate,
        "total_quantity": total_quantity,
        "total_units_pi": total_units_pi,
        "pi_units_chronic": pi_units_chronic,
        "pi_units_acute": pi_units_acute,
        "pi_units_therapy_cycle": pi_units_therapy_cycle,
        "repeat_patient_count": repeat_patient_count,
        "new_patient_count": new_patient_count,
        "items_per_bill": items_per_bill,
        "revenue_per_patient": revenue_per_patient,
    }


def build_pair_table(test_pre: pd.DataFrame, test_post: pd.DataFrame,
                     ctrl_pre: pd.DataFrame, ctrl_post: pd.DataFrame,
                     sku_ids: set[int] | None = None,
                     patient_first_seen: dict | None = None,
                     bill_filter: str = "with_pi") -> pd.DataFrame:
    """Produce a long-form DataFrame with columns:
       kpi · test_pre · test_post · test_delta_pct · ctrl_pre · ctrl_post · ctrl_delta_pct · did
    `test_pre` corresponds to pilot pre-window, `ctrl_pre` to non-pilot pre.
    (The legacy `test`/`ctrl` parameter names are kept for backwards compat;
    callers should pass pilot/non-pilot data.)
    """
    cells = {
        "test_pre": compute_kpis(test_pre, sku_ids, patient_first_seen, bill_filter),
        "test_post": compute_kpis(test_post, sku_ids, patient_first_seen, bill_filter),
        "ctrl_pre": compute_kpis(ctrl_pre, sku_ids, patient_first_seen, bill_filter),
        "ctrl_post": compute_kpis(ctrl_post, sku_ids, patient_first_seen, bill_filter),
    }
    rows = []
    keys = list(cells["test_pre"].keys())
    for k in keys:
        tpre, tpost = cells["test_pre"][k], cells["test_post"][k]
        cpre, cpost = cells["ctrl_pre"][k], cells["ctrl_post"][k]
        t_delta = ((tpost - tpre) / tpre * 100) if tpre else None
        c_delta = ((cpost - cpre) / cpre * 100) if cpre else None
        did = (t_delta - c_delta) if (t_delta is not None and c_delta is not None) else None
        rows.append({
            "kpi": k,
            "test_pre": tpre, "test_post": tpost, "test_delta_pct": t_delta,
            "ctrl_pre": cpre, "ctrl_post": cpost, "ctrl_delta_pct": c_delta,
            "did": did,
        })
    return pd.DataFrame(rows)


def compute_repeat_churn(pre_df: pd.DataFrame, post_df: pd.DataFrame,
                         sku_ids: set[int] | None = None) -> dict:
    """Given pre and post sales (both already filtered to one store), return
    8-day repeat rate and churn from patient-id sets.
    If sku_ids is provided, restrict pre customer set to those who bought
    ≥1 price-↑ SKU in pre; rate = % of them returning in post."""
    def patient_set(df: pd.DataFrame) -> set:
        gross = df[df["bill_flag"] == "gross"]
        if sku_ids is not None:
            pi_bill_ids = set(gross.loc[gross["drug_id"].isin(sku_ids), "bill_id"].dropna().unique())
            gross = gross[gross["bill_id"].isin(pi_bill_ids)]
        return set(gross["patient_id"].dropna().unique())
    pre_pts = patient_set(pre_df)
    post_pts = patient_set(post_df)
    if not pre_pts:
        return {"repeat_rate": 0.0, "churn_rate": 0.0,
                "n_pre": 0, "n_post": 0, "n_returned": 0, "n_churned": 0}
    returned = pre_pts & post_pts
    churned = pre_pts - post_pts
    return {
        "repeat_rate": len(returned) / len(pre_pts),
        "churn_rate": len(churned) / len(pre_pts),
        "n_pre": len(pre_pts), "n_post": len(post_pts),
        "n_returned": len(returned), "n_churned": len(churned),
    }


def compute_elasticity(pre_df: pd.DataFrame, post_df: pd.DataFrame,
                       sku_ids: set[int]) -> pd.DataFrame:
    """Per drug-id (within the price-↑ SKU list), compute pre/post units & avg
    realised price (revenue / units) and own-price elasticity."""
    def per_drug(df, mask=None):
        f = df[df["bill_flag"] == "gross"]
        if mask is not None: f = f[mask(f)]
        agg = f.groupby("drug_id").agg(
            units=("net_quantity", "sum"),
            revenue=("revenue_value", "sum"),
        )
        agg["avg_price"] = (agg["revenue"] / agg["units"]).where(agg["units"] > 0)
        return agg

    pre_g = per_drug(pre_df)
    post_g = per_drug(post_df)
    sku = pd.DataFrame(index=sorted(sku_ids))
    sku.index.name = "drug_id"
    sku = sku.join(pre_g.add_prefix("pre_"), how="left").join(post_g.add_prefix("post_"), how="left")
    sku = sku.fillna(0)

    sku["units_delta_pct"] = ((sku["post_units"] - sku["pre_units"])
                              / sku["pre_units"].replace(0, pd.NA)) * 100
    sku["price_delta_pct"] = ((sku["post_avg_price"] - sku["pre_avg_price"])
                              / sku["pre_avg_price"].replace(0, pd.NA)) * 100
    sku["own_elasticity"] = sku["units_delta_pct"] / sku["price_delta_pct"].replace(0, pd.NA)
    return sku.reset_index()


@st.cache_data(ttl=1800, show_spinner=False)
@st.cache_data(ttl=1800, show_spinner=False)
def _load_sku_category() -> dict[int, str]:
    """Load `drug_id -> category` lookup for price-↑ SKUs. Returns empty dict
    if the snapshot is missing (KPIs that depend on it gracefully degrade to 0)."""
    if not SKU_CATEGORY_SNAPSHOT.exists():
        return {}
    df = pd.read_parquet(SKU_CATEGORY_SNAPSHOT)
    return dict(zip(df["drug_id"].astype(int), df["category"].fillna("unknown").astype(str)))


@st.cache_data(ttl=1800, show_spinner=False)
def _load_partial_bill_snapshot() -> pd.DataFrame | None:
    """Load partial-bill cohort history: their 4-month pre + post purchases of
    price-↑ SKUs across all chain stores. Returns None if missing."""
    if not PARTIAL_BILL_SNAPSHOT.exists():
        return None
    df = pd.read_parquet(PARTIAL_BILL_SNAPSHOT)
    if "day" in df.columns:
        df["day"] = pd.to_datetime(df["day"], errors="coerce").dt.date
    return df


def compute_cohort_sku_pre_post(
    pre_df: pd.DataFrame, post_df: pd.DataFrame,
    cohort_patient_ids: set[int], sku_ids: set[int],
) -> pd.DataFrame:
    """For a fixed patient cohort, per-SKU pre vs post quantity and revenue.
    Both DataFrames should be pre-filtered to the relevant stores (e.g. pilot).
    Restricts to gross bills and price-↑ SKU drug_ids. Drug names are NOT
    included here — caller can merge with `load_skus()` for those."""
    def slice_for_cohort(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty: return df
        g = df[df["bill_flag"] == "gross"]
        g = g[g["patient_id"].astype("Int64").isin(cohort_patient_ids)]
        g = g[g["drug_id"].astype("Int64").isin(sku_ids)]
        return g

    pre = slice_for_cohort(pre_df)
    post = slice_for_cohort(post_df)

    pre_agg = pre.groupby("drug_id").agg(pre_qty=("net_quantity", "sum"),
                                          pre_revenue=("revenue_value", "sum"))
    post_agg = post.groupby("drug_id").agg(post_qty=("net_quantity", "sum"),
                                            post_revenue=("revenue_value", "sum"))
    out = pre_agg.join(post_agg, how="outer").fillna(0).reset_index()
    out["delta_qty"] = out["post_qty"] - out["pre_qty"]
    out["delta_qty_pct"] = (out["delta_qty"] / out["pre_qty"].replace(0, pd.NA)) * 100
    cols = ["drug_id", "pre_qty", "post_qty", "delta_qty",
            "delta_qty_pct", "pre_revenue", "post_revenue"]
    return out[cols].sort_values("post_qty", ascending=False)


def compute_cohort_monthly_history(
    sales_df: pd.DataFrame,
    cohort_ids: set[int],
    sku_ids: set[int],
    cat_lookup: dict[int, str],
    pilot_stores: tuple[str, ...],
) -> pd.DataFrame:
    """Per calendar month, KPI breakdown for the cohort patients' activity in
    pilot stores. Pure compute; caller passes in the already-loaded snapshot.

    Returns one row per calendar month present in the data, with columns:
      month, bills, patients, frequency,
      total_units, total_skus,
      total_units_chronic, total_units_acute,
      total_skus_chronic, total_skus_acute,
      price_hiked_units, price_hiked_skus,
      price_hiked_units_chronic, price_hiked_units_acute,
      price_hiked_skus_chronic, price_hiked_skus_acute,
    """
    empty_cols = [
        "month", "bills", "patients", "frequency",
        "total_units", "total_skus",
        "total_units_chronic", "total_units_acute",
        "total_skus_chronic", "total_skus_acute",
        "price_hiked_units", "price_hiked_skus",
        "price_hiked_units_chronic", "price_hiked_units_acute",
        "price_hiked_skus_chronic", "price_hiked_skus_acute",
    ]
    if sales_df is None or sales_df.empty or not cohort_ids:
        return pd.DataFrame(columns=empty_cols)

    df = sales_df[sales_df["bill_flag"] == "gross"].copy()
    if pilot_stores:
        df = df[df["store_name"].isin(pilot_stores)]
    df = df[df["patient_id"].astype("Int64").isin(cohort_ids)]
    if df.empty:
        return pd.DataFrame(columns=empty_cols)

    df["month"] = pd.to_datetime(df["created_at"]).dt.to_period("M").astype(str)
    df["_cat"] = df["drug_id"].map(lambda d: cat_lookup.get(int(d), "unknown") if pd.notna(d) else "unknown")
    df["_is_pi"] = df["drug_id"].astype("Int64").isin(sku_ids)

    def per_month(g: pd.DataFrame) -> pd.Series:
        bills = int(g["bill_id"].nunique())
        patients = int(g["patient_id"].dropna().nunique())
        pi = g[g["_is_pi"]]
        chronic = g[g["_cat"] == "chronic"]
        acute = g[g["_cat"] == "acute"]
        pi_chronic = pi[pi["_cat"] == "chronic"]
        pi_acute = pi[pi["_cat"] == "acute"]
        return pd.Series({
            "bills": bills,
            "patients": patients,
            "frequency": (bills / patients) if patients else 0.0,
            "total_units": float(g["net_quantity"].sum()),
            "total_skus": int(g["drug_id"].dropna().nunique()),
            "total_units_chronic": float(chronic["net_quantity"].sum()),
            "total_units_acute": float(acute["net_quantity"].sum()),
            "total_skus_chronic": int(chronic["drug_id"].dropna().nunique()),
            "total_skus_acute": int(acute["drug_id"].dropna().nunique()),
            "price_hiked_units": float(pi["net_quantity"].sum()),
            "price_hiked_skus": int(pi["drug_id"].dropna().nunique()),
            "price_hiked_units_chronic": float(pi_chronic["net_quantity"].sum()),
            "price_hiked_units_acute": float(pi_acute["net_quantity"].sum()),
            "price_hiked_skus_chronic": int(pi_chronic["drug_id"].dropna().nunique()),
            "price_hiked_skus_acute": int(pi_acute["drug_id"].dropna().nunique()),
        })

    out = df.groupby("month", sort=True).apply(per_month, include_groups=False).reset_index()
    return out[empty_cols]


# ============================================================================
# Phase 12 — Chronic-cohort deep dive (Refill Cycle + Bill Progression)
# ============================================================================

@st.cache_data(ttl=1800, show_spinner=False)
def _load_cohort_history_snapshot() -> pd.DataFrame | None:
    """6-month all-chain history of chronic price-↑ SKUs for the pilot-post cohort.
    Built by `refresh_snapshots.py` step [6/6]. Returns None if missing."""
    if not COHORT_HISTORY_SNAPSHOT.exists():
        return None
    df = pd.read_parquet(COHORT_HISTORY_SNAPSHOT)
    if "created_at" in df.columns:
        df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")
    return df


def chronic_pi_drug_ids() -> set[int]:
    """Drug IDs in the price-↑ universe whose category is `chronic`.
    Joins the SKU CSV against `sku_category.parquet`. Empty set if either is missing."""
    cat = _load_sku_category()
    if not cat:
        return set()
    skus = load_skus()
    sku_universe = set(int(x) for x in skus["drug_id"].astype(int))
    return {d for d in sku_universe if cat.get(int(d), "") == "chronic"}


def compute_refill_cycle(
    hist_df: pd.DataFrame,
    post_start,
    today,
) -> pd.DataFrame:
    """Per (patient_id, drug_id), derive refill cycle and flag overdue.

    Inputs:
      hist_df    — cohort-history rows (one row per gross-bill line), with
                   columns: patient_id, drug_id, drug_name, store_name,
                   created_at, net_quantity, bill_flag.
      post_start — date marking start of post window (anything before is `pre`).
      today      — date used to compute days_overdue.

    Output columns:
      patient_id, drug_id, drug_name, n_pre_purchases, last_pre_date,
      last_pre_qty, last_pre_store, cycle_days, expected_refill_date,
      any_post_purchase, days_overdue, status
        status ∈ {refilled, on_time, overdue, no_estimate}
    """
    if hist_df is None or hist_df.empty:
        return pd.DataFrame(columns=[
            "patient_id", "drug_id", "drug_name", "n_pre_purchases",
            "last_pre_date", "last_pre_qty", "last_pre_store",
            "cycle_days", "expected_refill_date",
            "any_post_purchase", "days_overdue", "status",
        ])

    post_start = pd.Timestamp(post_start).date()
    today = pd.Timestamp(today).date()

    df = hist_df[hist_df["bill_flag"].astype(str).str.lower() == "gross"].copy()
    df["day"] = pd.to_datetime(df["created_at"]).dt.date
    df["drug_id"] = df["drug_id"].astype("Int64")
    df["patient_id"] = df["patient_id"].astype("Int64")

    # Per (patient × drug × day): sum qty (handle multiple bills same day as one refill event).
    daily = (
        df.groupby(["patient_id", "drug_id", "day"], dropna=True)
        .agg(qty=("net_quantity", "sum"),
             store_name=("store_name", "first"),
             drug_name=("drug_name", "first"))
        .reset_index()
        .sort_values(["patient_id", "drug_id", "day"])
    )

    # Collapse consecutive purchases of the same drug within SAME_REFILL_WINDOW_DAYS
    # into a single refill event (anchored to the FIRST purchase, qty summed).
    # This prevents spurious 1-2 day "cycles" from same-refill top-ups (e.g., a
    # patient buying half a strip today and coming back tomorrow for the rest).
    from datetime import timedelta as _td
    collapsed_rows: list[dict] = []
    for (pid, did), g in daily.groupby(["patient_id", "drug_id"], dropna=True):
        events: list[dict] = []
        for _, r in g.sort_values("day").iterrows():
            if events and (r["day"] - events[-1]["day"]).days <= SAME_REFILL_WINDOW_DAYS:
                events[-1]["qty"] = events[-1]["qty"] + float(r["qty"])
            else:
                events.append({"patient_id": pid, "drug_id": did,
                                "day": r["day"], "qty": float(r["qty"]),
                                "store_name": r["store_name"],
                                "drug_name": r["drug_name"]})
        collapsed_rows.extend(events)
    daily = pd.DataFrame(collapsed_rows).sort_values(["patient_id", "drug_id", "day"])

    out_rows: list[dict] = []
    for (pid, did), g in daily.groupby(["patient_id", "drug_id"], dropna=True):
        if pd.isna(pid) or pd.isna(did):
            continue
        pre = g[g["day"] < post_start]
        post = g[g["day"] >= post_start]
        n_pre = len(pre)
        drug_name = g["drug_name"].iloc[-1] if len(g) else ""

        if n_pre == 0:
            # Patient bought this drug only post-go-live — irrelevant to refill-overdue.
            continue

        last_pre = pre.iloc[-1]
        last_pre_date = last_pre["day"]
        last_pre_qty = float(last_pre["qty"])
        last_pre_store = str(last_pre["store_name"])

        if n_pre >= 2:
            gaps = (pre["day"].diff().dt.days if hasattr(pre["day"], "dt")
                    else pd.Series([(b - a).days for a, b in zip(pre["day"][:-1], pre["day"][1:])]))
            # The lambda-safe way: compute via list.
            sorted_days = list(pre["day"])
            gap_list = [(sorted_days[i] - sorted_days[i - 1]).days for i in range(1, len(sorted_days))]
            if len(gap_list) >= 2:
                cycle_days = float(pd.Series(gap_list).median())
            else:
                cycle_days = float(pd.Series(gap_list).mean())
        else:
            cycle_days = float("nan")

        if pd.isna(cycle_days) or cycle_days < MIN_PLAUSIBLE_CYCLE_DAYS:
            status = "no_estimate"
            expected = pd.NaT
            days_overdue = None
            any_post = not post.empty
        else:
            from datetime import timedelta
            expected = last_pre_date + timedelta(days=int(round(cycle_days)))
            any_post = not post.empty
            days_overdue = (today - expected).days
            if any_post:
                status = "refilled"
            elif days_overdue <= REFILL_GRACE_DAYS:
                status = "on_time"
            else:
                status = "overdue"

        out_rows.append({
            "patient_id": int(pid),
            "drug_id": int(did),
            "drug_name": drug_name,
            "n_pre_purchases": int(n_pre),
            "last_pre_date": last_pre_date,
            "last_pre_qty": last_pre_qty,
            "last_pre_store": last_pre_store,
            "cycle_days": cycle_days if not pd.isna(cycle_days) else None,
            "expected_refill_date": expected if pd.notna(expected) else None,
            "any_post_purchase": bool(any_post),
            "days_overdue": days_overdue,
            "status": status,
        })

    return pd.DataFrame(out_rows)


def compute_bill_progression(
    hist_df: pd.DataFrame,
    post_start,
    post_end,
    chronic_pi_ids: set[int],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare each patient's latest post-window bill against their previous
    1st / 2nd / 3rd bills (anywhere in chain). Identify chronic price-↑ SKUs
    they used to buy but didn't this time.

    Returns:
      per_patient — one row per patient with comparisons + dropped drug_id lists.
      per_drug    — chronic price-↑ drug rollup: how many patients dropped each.
    """
    cols_pp = [
        "patient_id", "latest_bill_id", "latest_bill_date", "latest_store",
        "n_chronic_pi_latest",
        "prev1_date", "prev1_store", "dropped_vs_prev1", "dropped_vs_prev1_drugs",
        "prev2_date", "prev2_store", "dropped_vs_prev2", "dropped_vs_prev2_drugs",
        "prev3_date", "prev3_store", "dropped_vs_prev3", "dropped_vs_prev3_drugs",
    ]
    if hist_df is None or hist_df.empty or not chronic_pi_ids:
        return pd.DataFrame(columns=cols_pp), pd.DataFrame(columns=["drug_id", "drug_name", "vs_prev1", "vs_prev2", "vs_prev3", "total_dropped"])

    post_start = pd.Timestamp(post_start).date()
    post_end = pd.Timestamp(post_end).date()
    chronic_pi_ids = {int(x) for x in chronic_pi_ids}

    df = hist_df[hist_df["bill_flag"].astype(str).str.lower() == "gross"].copy()
    df["day"] = pd.to_datetime(df["created_at"]).dt.date
    df["drug_id"] = df["drug_id"].astype("Int64")
    df["patient_id"] = df["patient_id"].astype("Int64")
    # NB: cohort-history snapshot doesn't currently carry bill_id (kept lean),
    # so we use (patient, store, day) as the bill key — good enough for cohort
    # rollups where same-day same-store rows belong to the same visit.
    df["bill_key"] = df.apply(
        lambda r: (int(r["patient_id"]) if pd.notna(r["patient_id"]) else -1,
                   str(r["store_name"]),
                   r["day"]),
        axis=1,
    )

    drug_name_lookup = (
        df.dropna(subset=["drug_id"])
        .drop_duplicates("drug_id")
        .set_index("drug_id")["drug_name"]
        .astype(str)
        .to_dict()
    )

    pp_rows: list[dict] = []
    per_drug_counter = {k: {"vs_prev1": 0, "vs_prev2": 0, "vs_prev3": 0} for k in chronic_pi_ids}

    # Build per-bill aggregates
    bill_df = (
        df.dropna(subset=["patient_id"])
        .groupby("bill_key")
        .agg(patient_id=("patient_id", "first"),
             store_name=("store_name", "first"),
             day=("day", "first"),
             drug_ids=("drug_id", lambda s: set(int(x) for x in s if pd.notna(x))))
        .reset_index()
    )

    for pid, pg in bill_df.groupby("patient_id"):
        pg = pg.sort_values("day", ascending=False).reset_index(drop=True)
        latest_idx = None
        for i, r in pg.iterrows():
            if post_start <= r["day"] <= post_end:
                latest_idx = i
                break
        if latest_idx is None:
            continue

        latest = pg.iloc[latest_idx]
        latest_drugs = set(int(x) for x in latest["drug_ids"]) & chronic_pi_ids
        prev_bills = pg.iloc[latest_idx + 1: latest_idx + 4].reset_index(drop=True)

        row: dict = {
            "patient_id": int(pid),
            "latest_bill_id": f"{latest['store_name']}@{latest['day']}",
            "latest_bill_date": latest["day"],
            "latest_store": str(latest["store_name"]),
            "n_chronic_pi_latest": len(latest_drugs),
        }
        for n in (1, 2, 3):
            if n - 1 < len(prev_bills):
                pb = prev_bills.iloc[n - 1]
                pb_drugs = set(int(x) for x in pb["drug_ids"]) & chronic_pi_ids
                dropped = pb_drugs - latest_drugs
                row[f"prev{n}_date"] = pb["day"]
                row[f"prev{n}_store"] = str(pb["store_name"])
                row[f"dropped_vs_prev{n}"] = len(dropped)
                row[f"dropped_vs_prev{n}_drugs"] = sorted(dropped)
                for d in dropped:
                    per_drug_counter[d][f"vs_prev{n}"] += 1
            else:
                row[f"prev{n}_date"] = None
                row[f"prev{n}_store"] = None
                row[f"dropped_vs_prev{n}"] = 0
                row[f"dropped_vs_prev{n}_drugs"] = []
        pp_rows.append(row)

    per_patient = pd.DataFrame(pp_rows, columns=cols_pp)

    per_drug_rows = []
    for did, counts in per_drug_counter.items():
        if counts["vs_prev1"] + counts["vs_prev2"] + counts["vs_prev3"] == 0:
            continue
        per_drug_rows.append({
            "drug_id": did,
            "drug_name": drug_name_lookup.get(did, ""),
            "vs_prev1": counts["vs_prev1"],
            "vs_prev2": counts["vs_prev2"],
            "vs_prev3": counts["vs_prev3"],
            "total_dropped": counts["vs_prev1"] + counts["vs_prev2"] + counts["vs_prev3"],
        })
    per_drug = pd.DataFrame(per_drug_rows).sort_values("total_dropped", ascending=False)
    return per_patient, per_drug

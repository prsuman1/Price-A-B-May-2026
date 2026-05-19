"""AI chat data layer.

Three responsibilities:
  1. Build a comprehensive markdown context summary from survey + quant data.
  2. Stream chat completions from OpenRouter (anthropic/claude-sonnet-4.6).
  3. Persist conversation turns to `chat_history.csv`.

The chat page (`chat.py`) is a thin UI wrapper around these helpers.
"""

from __future__ import annotations

import csv
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Iterator

import pandas as pd
import requests
import streamlit as st
from dotenv import load_dotenv

from data import (
    PAIRED_NON_PILOT_STORES,
    PILOT_STORES,
    PROJECT_DIR,
    QUANT_POST_END,
    QUANT_POST_START,
    QUANT_PRE_END,
    QUANT_PRE_START,
    STORE_PAIRS,
    _load_patients_snapshot,
    _load_sales_snapshot,
    _load_store_totals_snapshot,
    aggregate_totals,
    build_joined,
    compute_kpis,
    load_skus,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL = "anthropic/claude-sonnet-4.6"
MAX_OUTPUT_TOKENS = 4096
TEMPERATURE = 0.2

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
CHAT_HISTORY_CSV = PROJECT_DIR / "chat_history.csv"

SYSTEM_PROMPT_TEMPLATE = """You are **Zeno**, the data analyst for a pharmacy chain's price-change A/B test.

Your job: answer questions about the experiment using the structured summary below.
- Be concise and numerical. Cite specific numbers from the summary.
- If a question can't be answered from the summary, say so explicitly — don't guess.
- Format responses with markdown (tables, bullets) when it helps.
- Use ₹ for currency. Use percentage points (pp) for differences in rates.
- If asked your name, you are Zeno.

# CONTEXT SUMMARY

{summary}
"""


def _api_key() -> str | None:
    load_dotenv(PROJECT_DIR / ".env")
    return os.environ.get("openrouterkey") or os.environ.get("OPENROUTER_API_KEY")


# ---------------------------------------------------------------------------
# Context summary builder
# ---------------------------------------------------------------------------

def _fmt_money(v) -> str:
    if v is None or pd.isna(v): return "—"
    return f"₹{v:,.0f}"


def _fmt_pct(v) -> str:
    if v is None or pd.isna(v): return "—"
    return f"{v*100:.1f}%"


def _fmt_delta(v) -> str:
    if v is None or pd.isna(v): return "—"
    return f"{v:+.1f}%"


def _fmt_int(v) -> str:
    if v is None or pd.isna(v): return "—"
    return f"{int(v):,}"


def _delta_pct(post, pre):
    return ((post - pre) / pre * 100) if pre else None


def _df_to_md(df: pd.DataFrame) -> str:
    """Tiny markdown-table renderer so we don't need the `tabulate` dep."""
    if df.empty: return "_(no rows)_"
    headers = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(headers) + " |",
             "|" + "|".join(["---"] * len(headers)) + "|"]
    for _, row in df.iterrows():
        cells = ["" if pd.isna(v) else str(v) for v in row.tolist()]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _summarise_survey(bundle: dict) -> str:
    """Markdown summary of the surveyor side."""
    survey = bundle["survey"]
    bills = bundle["bills"]
    stats = bundle["stats"]
    sales = bundle["sales"]

    out = []
    out.append("## Surveyor data")
    out.append(f"Survey rows post-go-live: **{stats['survey_rows']:,}**.")
    out.append(
        f"Breakdown by `bill_kind`: serial={stats['serial']:,}, "
        f"bill_created_no_id={stats['bill_created_no_id']:,}, walkaway={stats['walkaway']:,}."
    )
    out.append(
        f"Of the {stats['serial']:,} serial-bills, **{stats['joined_to_sales']:,}** joined to sales "
        f"(sync_lag={stats['sync_lag']:,})."
    )
    out.append(
        f"RAG: 🟢 green={stats['rag_green']:,}, "
        f"🟡 amber={stats['rag_amber']:,} "
        f"(partial={stats['rag_amber_partial']:,}, genrtn={stats['rag_amber_genrtn']:,}), "
        f"🔴 red={stats['rag_red']:,}."
    )

    # Per-store cut
    if not survey.empty:
        out.append("\n### Survey per store (post-go-live)")
        per_store = survey.groupby("store").agg(
            interactions=("submission_id", "size"),
            green=("rag", lambda s: (s == "green").sum()),
            amber=("rag", lambda s: (s == "amber").sum()),
            red=("rag", lambda s: (s == "red").sum()),
            spoke_yes=("spoke_price", lambda s: s.str.lower().eq("yes").sum()),
            nonverbal_yes=("nonverbal", lambda s: s.str.lower().eq("yes").sum()),
        ).reset_index()
        out.append(_df_to_md(per_store))

    # Bills w/ price-↑ SKU
    if not bills.empty:
        pi_bills = int(bills["has_price_increased_sku"].fillna(False).sum())
        joined = int(bills["bill_id"].notna().sum())
        out.append(
            f"\nBills joined to sales with **≥1 price-↑ SKU**: **{pi_bills:,}** "
            f"({100*pi_bills/joined:.1f}% of {joined:,} joined bills)."
        )
        genrtn = int(bills["used_genrtn"].fillna(False).sum())
        out.append(f"Bills with GENRTN coupon applied (per sales `promo-code`): **{genrtn}**.")

    # Top outcomes
    audit = stats.get("outcome_audit", {})
    if audit:
        out.append(
            f"\nOutcome distribution: canonical_green={audit.get('n_canonical_green',0):,}, "
            f"canonical_partial={audit.get('n_canonical_partial',0):,}, "
            f"canonical_walkaway={audit.get('n_canonical_walkaway',0):,}, "
            f"freetext={audit.get('n_freetext_total',0):,}."
        )
        top_ft = audit.get("top_freetext_in_green", {})
        if top_ft:
            top_str = ", ".join(f"\"{k}\" ({v})" for k, v in list(top_ft.items())[:8])
            out.append(f"Top free-text outcomes (defaulted to green): {top_str}.")

    return "\n".join(out)


def _summarise_quant(sku_ids: set, patients: dict | None) -> str:
    """Markdown summary of pilot vs non-pilot vs all-other for both bill-sets."""
    from data import fetch_quant_sales  # local import to avoid cycles

    out = ["## Quant (Pilot vs Non-Pilot vs All-Other)"]
    out.append(
        f"Pre window: **{QUANT_PRE_START} → {QUANT_PRE_END}** (12 days)\n"
        f"Post window: **{QUANT_POST_START} → {QUANT_POST_END}** (12 days)"
    )

    store_totals = _load_store_totals_snapshot()

    def load_group(stores: tuple[str, ...], d_from, d_to) -> pd.DataFrame:
        dfs = [fetch_quant_sales((s,), d_from.isoformat(), d_to.isoformat()) for s in stores]
        return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

    pilot_pre_df = load_group(PILOT_STORES, QUANT_PRE_START, QUANT_PRE_END)
    pilot_post_df = load_group(PILOT_STORES, QUANT_POST_START, QUANT_POST_END)
    npp_pre_df = load_group(PAIRED_NON_PILOT_STORES, QUANT_PRE_START, QUANT_PRE_END)
    npp_post_df = load_group(PAIRED_NON_PILOT_STORES, QUANT_POST_START, QUANT_POST_END)

    if store_totals is not None:
        all_chain = set(store_totals["store_name"].unique())
        other_stores = tuple(sorted(all_chain - set(PILOT_STORES)))
    else:
        other_stores = ()

    for bill_filter in ("with_pi", "without_pi"):
        label = "Price-↑ bills (treatment universe)" if bill_filter == "with_pi" else "Non-price-↑ bills"
        out.append(f"\n### {label} — DiD table")
        pilot_pre = compute_kpis(pilot_pre_df, sku_ids, patients, bill_filter)
        pilot_post = compute_kpis(pilot_post_df, sku_ids, patients, bill_filter)
        npp_pre = compute_kpis(npp_pre_df, sku_ids, patients, bill_filter)
        npp_post = compute_kpis(npp_post_df, sku_ids, patients, bill_filter)
        other_pre = aggregate_totals(store_totals, other_stores, QUANT_PRE_START, QUANT_PRE_END, bill_filter) if store_totals is not None else {}
        other_post = aggregate_totals(store_totals, other_stores, QUANT_POST_START, QUANT_POST_END, bill_filter) if store_totals is not None else {}

        rows = []
        for k in [
            "revenue", "gm", "gm_pct", "aov", "aom", "acm",
            "bill_count", "unique_customers", "total_quantity",
            "repeat_patient_count", "new_patient_count",
            "items_per_bill", "revenue_per_patient",
            "generic_mix_units_pct", "generic_mix_revenue_pct",
            "promo_usage_rate", "return_rate",
        ]:
            ppre, ppost = pilot_pre.get(k), pilot_post.get(k)
            npre, npost = npp_pre.get(k), npp_post.get(k)
            opre, opost = other_pre.get(k), other_post.get(k)
            p_d = _delta_pct(ppost, ppre)
            n_d = _delta_pct(npost, npre)
            o_d = _delta_pct(opost, opre) if opost is not None else None
            did_p = (p_d - n_d) if (p_d is not None and n_d is not None) else None
            did_o = (p_d - o_d) if (p_d is not None and o_d is not None) else None

            def fmt(v):
                if v is None or pd.isna(v): return "—"
                if k.endswith("_pct") or k.endswith("_rate"):
                    return f"{v*100:.1f}%"
                if k in ("revenue", "gm", "aov", "aom", "acm", "revenue_per_patient"):
                    return f"₹{v:,.0f}"
                if k in ("bill_count", "unique_customers", "total_quantity",
                          "repeat_patient_count", "new_patient_count"):
                    return f"{int(v):,}"
                return f"{v:.2f}"

            rows.append({
                "KPI": k,
                "Pilot Pre": fmt(ppre), "Pilot Post": fmt(ppost), "Pilot Δ%": _fmt_delta(p_d),
                "NPP Pre": fmt(npre), "NPP Post": fmt(npost), "NPP Δ%": _fmt_delta(n_d),
                "Other Δ%": _fmt_delta(o_d),
                "DiD vs Paired": _fmt_delta(did_p),
                "DiD vs Other": _fmt_delta(did_o),
            })
        out.append(_df_to_md(pd.DataFrame(rows)))

    # Per-pair quick view (post window, price-↑ bills only — most relevant)
    out.append("\n### Per pair (price-↑ bills, post window)")
    pair_rows = []
    for p in STORE_PAIRS:
        pi_df = load_group((p["pilot"],), QUANT_POST_START, QUANT_POST_END)
        np_df = load_group((p["non_pilot"],), QUANT_POST_START, QUANT_POST_END)
        pi_k = compute_kpis(pi_df, sku_ids, patients, "with_pi")
        np_k = compute_kpis(np_df, sku_ids, patients, "with_pi")
        pair_rows.append({
            "Pair": f"P{p['pair']}",
            "Pilot": p["pilot"], "Non-Pilot": p["non_pilot"],
            "Pilot Rev": _fmt_money(pi_k["revenue"]), "NPP Rev": _fmt_money(np_k["revenue"]),
            "Pilot GM": _fmt_money(pi_k["gm"]), "NPP GM": _fmt_money(np_k["gm"]),
            "Pilot AOV": _fmt_money(pi_k["aov"]), "NPP AOV": _fmt_money(np_k["aov"]),
            "Pilot Bills": _fmt_int(pi_k["bill_count"]), "NPP Bills": _fmt_int(np_k["bill_count"]),
        })
    out.append(_df_to_md(pd.DataFrame(pair_rows)))

    return "\n".join(out)


def _summarise_definitions() -> str:
    return """## Definitions & rules

**Experiment**: 346 SKU prices raised on 2026-05-06 at 6 Mumbai "pilot" stores. 6 paired "non-pilot" stores left unchanged for control. Survey team at 8 store-floor locations captures customer reactions.

**Pilot ↔ Non-Pilot pairs**:
1. Colaba ↔ Pantnagar
2. Khanda Colony ↔ Virar West
3. Bhiwandi Dhamankar Naka ↔ Dahisar
4. Khar West ↔ Dombivali
5. Goregaon West S.V. Road ↔ Mulund West-Sarvodaya Nagar
6. Sanpada ↔ Airoli

**RAG (survey)**:
- 🟢 GREEN — bill made normally (outcome="No change. Jaisa ban na tha bana." or free-text fall-through).
- 🟡 AMBER-partial — outcome contains "Partial bana" or "item kam".
- 🟡 AMBER-genrtn — bill's `promo-code` starts with `GENRTN` (trumps GREEN).
- 🔴 RED — outcome contains "Bana hi nahin" (walk-away). Outcome-driven, not bill-ID-driven.

**KPI formulas**:
- Revenue = `Σ(revenue_value − promo_discount)` on gross bills.
- Gross Margin = `Revenue − COGS − PromoDiscount`; COGS = `Σ(net_quantity × purchase_rate)`.
- AOV = Revenue / bill_count; AOM = GM / bill_count; ACM = GM / unique_customers.
- Generic mix % = units (or revenue) where `assortment_classification_id IN (3,4)`.
- Repeat patient = patient with ≥2 distinct bills in the window.
- New patient = patient whose first-ever bill in the chain falls inside the window.
- DiD = Pilot Δ% − Comparator Δ%.

**Bill-set filter**: `with_pi` = bills containing ≥1 price-↑ SKU; `without_pi` = bills with zero price-↑ SKUs (true control bills).
"""


@st.cache_data(ttl=1800, show_spinner=False)
def build_context_summary() -> str:
    """Build a markdown context summary from current data. Cached for 30 min."""
    bundle = build_joined()
    skus = load_skus()
    sku_ids = set(skus["drug_id"].astype(int))
    patients = _load_patients_snapshot()

    parts = [
        "# Project: Price-Change A/B Test (Pharmacy Chain, Mumbai)",
        f"Go-live: 2026-05-06. Today: {datetime.now().date().isoformat()}.",
        f"346 SKUs price-↑. 6 pilot stores ({', '.join(PILOT_STORES)}).",
        "",
        _summarise_definitions(),
        "",
        _summarise_survey(bundle),
        "",
        _summarise_quant(sku_ids, patients),
    ]
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Chat history (CSV)
# ---------------------------------------------------------------------------

_CHAT_COLS = ["timestamp", "session_id", "role", "content"]


def load_chat_history() -> pd.DataFrame:
    if not CHAT_HISTORY_CSV.exists():
        return pd.DataFrame(columns=_CHAT_COLS)
    try:
        return pd.read_csv(CHAT_HISTORY_CSV)
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=_CHAT_COLS)


def append_chat_turn(session_id: str, role: str, content: str) -> None:
    """Append a single chat turn to the CSV. Creates the file with headers if missing."""
    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "session_id": session_id,
        "role": role,
        "content": content,
    }
    write_header = not CHAT_HISTORY_CSV.exists()
    with open(CHAT_HISTORY_CSV, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_CHAT_COLS)
        if write_header: w.writeheader()
        w.writerow(row)


# ---------------------------------------------------------------------------
# OpenRouter streaming
# ---------------------------------------------------------------------------

def call_openrouter_stream(messages: list[dict]) -> Iterator[str]:
    """Stream chat completion chunks from OpenRouter. Yields text deltas."""
    key = _api_key()
    if not key:
        yield "**Error:** No API key found. Set `openrouterkey` in `.env`."
        return

    headers = {
        "Authorization": f"Bearer {key}",
        "HTTP-Referer": "https://github.com/prsuman1/Price-A-B-May-2026",
        "X-Title": "Price Change A/B Dashboard",
        "Content-Type": "application/json",
    }
    body = {
        "model": MODEL,
        "messages": messages,
        "stream": True,
        "max_tokens": MAX_OUTPUT_TOKENS,
        "temperature": TEMPERATURE,
    }
    try:
        with requests.post(OPENROUTER_URL, headers=headers, json=body, stream=True, timeout=120) as resp:
            if resp.status_code != 200:
                yield f"**API error** ({resp.status_code}): {resp.text[:500]}"
                return
            for raw in resp.iter_lines(decode_unicode=True):
                if not raw or not raw.startswith("data: "): continue
                payload = raw[len("data: "):]
                if payload.strip() == "[DONE]": break
                try:
                    chunk = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                delta = chunk.get("choices", [{}])[0].get("delta", {}).get("content")
                if delta: yield delta
    except requests.exceptions.RequestException as e:
        yield f"**Network error:** {e}"

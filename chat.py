"""Chat page — talk to Claude Sonnet 4.6 about the price-change experiment.

History persists to `chat_history.csv` in the project root. The chatbot answers
from a pre-computed structured summary (no live Redshift / no tool-use).
"""

from __future__ import annotations

import uuid

import streamlit as st

from ai_chat import (
    CHAT_HISTORY_CSV,
    MODEL,
    SYSTEM_PROMPT_TEMPLATE,
    _api_key,
    append_chat_turn,
    build_context_summary,
    call_openrouter_stream,
    load_chat_history,
)

st.title("💬 Ask Zeno")
st.caption(
    f"Powered by **{MODEL}** via OpenRouter. Answers from a pre-computed summary of survey + sales data. "
    f"History saved to `chat_history.csv` in the repo."
)

# --- Load full history first; choose default session ------------------------
history = load_chat_history()

# Order past sessions by latest activity (most recent first).
if not history.empty:
    sessions_ordered = (
        history.sort_values("timestamp")
        .groupby("session_id", sort=False)
        .agg(
            first_user_msg=("content", lambda s: next(
                (c for c, r in zip(s, history.loc[s.index, "role"]) if r == "user"), "(no user msg)"
            )),
            last_ts=("timestamp", "max"),
            n_turns=("content", "size"),
        )
        .sort_values("last_ts", ascending=False)
    )
else:
    sessions_ordered = None

# Default to the most recent session so a page refresh doesn't lose context.
if "chat_session_id" not in st.session_state:
    if sessions_ordered is not None and not sessions_ordered.empty:
        st.session_state.chat_session_id = sessions_ordered.index[0]
    else:
        st.session_state.chat_session_id = uuid.uuid4().hex[:8]
session_id = st.session_state.chat_session_id

# --- API key check -----------------------------------------------------------
if not _api_key():
    st.error(
        "OpenRouter API key not found. Add `openrouterkey=...` to `.env` (local) "
        "or set the `openrouterkey` secret in Streamlit Cloud settings."
    )
    st.stop()

# --- Sidebar ----------------------------------------------------------------
st.sidebar.header("Chat")

# 1) New chat
if st.sidebar.button("🆕 New chat", width='stretch', type="primary"):
    st.session_state.chat_session_id = uuid.uuid4().hex[:8]
    st.rerun()

st.sidebar.markdown("---")

# 2) Past chats list (like Claude.ai)
st.sidebar.markdown("**Past chats**")
if sessions_ordered is None or sessions_ordered.empty:
    st.sidebar.caption("_(none yet — say hi below to start)_")
else:
    for sid, row in sessions_ordered.iterrows():
        title = (row["first_user_msg"] or "(empty)").strip().replace("\n", " ")
        if len(title) > 40: title = title[:38] + "…"
        marker = "🟢 " if sid == session_id else ""
        if st.sidebar.button(
            f"{marker}{title}",
            key=f"sess_{sid}",
            width='stretch',
            help=f"{row['n_turns']} turn(s) · last {row['last_ts']}",
        ):
            st.session_state.chat_session_id = sid
            st.rerun()

st.sidebar.markdown("---")

# 3) Suggested prompts
st.sidebar.markdown("**Suggested questions**")
SUGGESTED = [
    "What's the headline DiD on revenue for price-↑ bills?",
    "Which store had the highest walk-away rate?",
    "How does Pilot compare to All-Other stores on GM%?",
    "How many new vs repeat patients in pilot stores during the post window?",
    "Top 5 price-sensitive SKUs by elasticity?",
    "Summarize the surveyor findings in 3 bullets.",
    "Did Goregaon West S.V. Road perform better than its non-pilot pair?",
    "How many bills used the GENRTN coupon?",
]
clicked = None
for q in SUGGESTED:
    if st.sidebar.button(q, key=f"suggest_{hash(q)}", width='stretch'):
        clicked = q

st.sidebar.markdown("---")

# 4) Download
if CHAT_HISTORY_CSV.exists():
    st.sidebar.download_button(
        "📥 Download chat_history.csv",
        data=CHAT_HISTORY_CSV.read_bytes(),
        file_name="chat_history.csv",
        mime="text/csv",
        width='stretch',
    )

st.sidebar.caption(
    "💡 On Streamlit Cloud, file writes are ephemeral per deploy — chat history "
    "persists only within the running app. For permanent persistence: pull the CSV "
    "locally via the download button and commit."
)

# --- Load context ------------------------------------------------------------
with st.spinner("Loading data summary…"):
    summary = build_context_summary()

# Current session's turns
session_history = history[history["session_id"] == session_id] if not history.empty else history

# --- Render past turns -------------------------------------------------------
for _, row in session_history.iterrows():
    with st.chat_message(row["role"]):
        st.markdown(row["content"])

# --- Handle new turn ---------------------------------------------------------
user_input = clicked or st.chat_input("Ask about the price-change experiment…")

if user_input:
    with st.chat_message("user"):
        st.markdown(user_input)
    append_chat_turn(session_id, "user", user_input)

    # Build messages: system + session history + new user turn.
    messages = [{"role": "system", "content": SYSTEM_PROMPT_TEMPLATE.format(summary=summary)}]
    for _, row in session_history.iterrows():
        messages.append({"role": row["role"], "content": row["content"]})
    messages.append({"role": "user", "content": user_input})

    with st.chat_message("assistant"):
        response = st.write_stream(call_openrouter_stream(messages))
    append_chat_turn(session_id, "assistant", response)
    st.rerun()

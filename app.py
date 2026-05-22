"""Multi-page entry for the price-change A/B dashboard."""

import streamlit as st

st.set_page_config(page_title="Price Change A/B Insights", layout="wide")

pages = [
    st.Page("store_insight.py", title="Store Insight", icon="🏪", default=True),
    st.Page("quant_insight.py", title="Quant Insight", icon="📊"),
    st.Page("refill_cycle.py", title="Refill Cycle", icon="🔁"),
    st.Page("bill_progression.py", title="Bill Progression", icon="📉"),
    st.Page("chat.py", title="Ask Zeno", icon="💬"),
]

st.navigation(pages).run()

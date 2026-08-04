import streamlit as st


def apply_theme() -> None:
    st.markdown(
        """
        <style>
        :root {--navy:#071d38; --blue:#2563eb; --ink:#0f172a; --muted:#64748b; --line:#e2e8f0;}
        .stApp {background:#f8fafc; color:var(--ink);}
        .block-container {padding-top:1.6rem; padding-bottom:3rem; max-width:1480px;}
        [data-testid="stSidebar"] {background:linear-gradient(180deg,#071d38 0%,#0b294c 100%);}
        [data-testid="stSidebar"] * {color:#f8fafc;}
        [data-testid="stSidebarNav"] {padding-top:.35rem;}
        [data-testid="stSidebarNavLink"] {border-radius:9px; margin:2px 8px;}
        [data-testid="stSidebarNavLink"][aria-current="page"] {background:#2257a5;}
        .hci-brand {display:flex; align-items:center; gap:9px; padding:6px 2px 18px;}
        .hci-brand b {font-size:2rem; color:#f6b73c; letter-spacing:-.06em;}
        .hci-brand span {font-size:.68rem; line-height:1.2; font-weight:700; color:#fff;}
        h1, h2, h3 {letter-spacing:-.025em; color:var(--ink);}
        h1 {font-size:1.85rem !important; margin-bottom:.3rem !important;}
        h3 {margin-top:.25rem !important;}
        .profile-card {margin-top:12px; padding:12px 14px; border:1px solid var(--line); border-radius:12px; background:#fff; text-align:right;}
        .profile-card span {color:var(--muted); font-size:.85rem;}
        .freshness {padding-top:2.15rem; text-align:right; color:var(--muted); font-size:.85rem;}
        [data-testid="stMetric"] {background:#fff; border:1px solid var(--line); padding:17px 18px; border-radius:13px; box-shadow:0 2px 8px rgba(15,23,42,.035); min-height:112px;}
        [data-testid="stMetricLabel"] {color:var(--muted); font-weight:600;}
        [data-testid="stMetricValue"] {color:var(--ink); font-weight:750;}
        [data-testid="stDataFrame"] {border:1px solid var(--line); border-radius:12px; overflow:hidden; background:#fff;}
        [data-testid="stVegaLiteChart"] {background:#fff; border:1px solid var(--line); padding:12px 14px; border-radius:13px;}
        [data-testid="stAlert"] * {color:#334155 !important;}
        .date-row {display:flex; justify-content:space-between; align-items:center; gap:12px; background:#fff; border:1px solid var(--line); border-radius:10px; padding:11px 12px; margin-bottom:8px;}
        .date-row span {color:var(--muted); font-size:.78rem;}
        .demand-badge {display:inline-block; border-radius:999px; padding:5px 9px; font-size:.74rem; font-weight:700; white-space:nowrap;}
        .demand-badge.very-high {background:#fee2e2; color:#b91c1c;}
        .demand-badge.high {background:#ffedd5; color:#c2410c;}
        .demand-badge.medium {background:#dbeafe; color:#1d4ed8;}
        .demand-badge.low {background:#dcfce7; color:#15803d;}
        .workflow-note {margin-top:14px; padding:13px 16px; background:#eff6ff; border:1px solid #bfdbfe; border-radius:11px; color:#1e3a8a;}
        .quality-banner {padding:14px 16px; border-radius:11px; margin:8px 0 16px; font-size:.94rem;}
        .quality-banner.warning {background:#fffbeb; border:1px solid #fde68a; color:#92400e;}
        .quality-banner.danger {background:#fef2f2; border:1px solid #fecaca; color:#991b1b;}
        .quality-banner.success {background:#f0fdf4; border:1px solid #bbf7d0; color:#166534;}
        .fact-list {background:#fff; border:1px solid var(--line); border-radius:12px; padding:15px 16px; min-height:190px;}
        .fact-list div {padding:11px 0; border-bottom:1px solid #eef2f7; color:#334155;}
        .fact-list div:last-child {border-bottom:none;}
        .tool-card {background:#fff; border:1px solid var(--line); border-radius:12px; padding:15px; min-height:145px; box-shadow:0 2px 8px rgba(15,23,42,.03);}
        .tool-title {font-weight:750; color:var(--ink); margin-bottom:10px;}
        .tool-status {display:inline-block; padding:4px 9px; border-radius:999px; font-size:.75rem; font-weight:700; margin-bottom:12px;}
        .tool-status.success {background:#dcfce7; color:#15803d;}
        .tool-status.info {background:#dbeafe; color:#1d4ed8;}
        .tool-status.warning {background:#ffedd5; color:#c2410c;}
        .tool-status.danger {background:#fee2e2; color:#b91c1c;}
        .tool-status.neutral {background:#f1f5f9; color:#475569;}
        .tool-detail {font-size:.86rem; line-height:1.45; color:var(--muted);}
        div[data-testid="stExpander"] {background:#fff; border-color:var(--line);}
        @media (max-width:900px) {.freshness{text-align:left;padding-top:.5rem}.profile-card{text-align:left}}
        </style>
        """,
        unsafe_allow_html=True,
    )


def status_label(value: str) -> str:
    icons = {"Mapped": "🟢", "No Room": "🟠", "Not Live": "🟡", "Not Mapped": "🔴", "Unknown": "⚪"}
    return f"{icons.get(str(value), '⚪')} {value}"

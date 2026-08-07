from __future__ import annotations

import pandas as pd
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
        [data-testid="stMetricValue"], [data-testid="stMetricValue"] > div {
            color:var(--ink); font-weight:750; font-size:2rem !important; line-height:1.12;
            white-space:normal !important; overflow:visible !important; text-overflow:clip !important;
        }
        [data-testid="stMetricValue"] p {
            font-size:1.65rem !important; line-height:1.08 !important; white-space:normal !important;
            overflow:visible !important; text-overflow:clip !important; overflow-wrap:anywhere;
        }
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


def style_table(frame: pd.DataFrame) -> pd.io.formats.style.Styler:
    """Apply HCI's shared, restrained status colors to a dataframe."""

    def semantic_color(value: object) -> str:
        if pd.isna(value):
            return ""
        label = str(value).strip().lower()
        colors = {
            "critical surge": ("#fee2e2", "#b91c1c"),
            "very high": ("#fee2e2", "#b91c1c"),
            "failed": ("#fee2e2", "#b91c1c"),
            "not mapped": ("#fee2e2", "#b91c1c"),
            "high increase": ("#ffedd5", "#c2410c"),
            "high": ("#ffedd5", "#c2410c"),
            "warning": ("#ffedd5", "#c2410c"),
            "no room": ("#ffedd5", "#c2410c"),
            "new demand": ("#dbeafe", "#1d4ed8"),
            "medium": ("#dbeafe", "#1d4ed8"),
            "stable": ("#dcfce7", "#15803d"),
            "low": ("#dcfce7", "#15803d"),
            "passed": ("#dcfce7", "#15803d"),
            "mapped": ("#dcfce7", "#15803d"),
            "available": ("#dcfce7", "#15803d"),
            "declining": ("#f1f5f9", "#64748b"),
            "no baseline": ("#f1f5f9", "#64748b"),
            "missing": ("#f1f5f9", "#64748b"),
            "unknown": ("#f1f5f9", "#64748b"),
            "not live": ("#fef9c3", "#854d0e"),
        }.get(label)
        if colors is None:
            if label.startswith("check inventory") or label.startswith("validate demand"):
                colors = ("#ffedd5", "#c2410c")
            elif label in {"monitor", "none"}:
                colors = ("#dcfce7", "#15803d")
            elif label == "lower priority":
                colors = ("#f1f5f9", "#64748b")
        if colors is None:
            return ""
        background, foreground = colors
        return f"background-color:{background};color:{foreground};font-weight:650"

    def change_color(value: object) -> str:
        number = pd.to_numeric(value, errors="coerce")
        if pd.isna(number):
            return "color:#94a3b8"
        if number >= 25:
            return "background-color:#fee2e2;color:#b91c1c;font-weight:700"
        if number >= 10:
            return "background-color:#ffedd5;color:#c2410c;font-weight:700"
        if number <= -10:
            return "background-color:#f1f5f9;color:#64748b;font-weight:650"
        return "background-color:#dcfce7;color:#15803d;font-weight:650"

    styler = frame.style
    semantic_columns = [
        column
        for column in frame.columns
        if column in {
            "Signal",
            "Destination Signal",
            "Hotel Signal",
            "Demand Level",
            "Status",
            "Agoda Status",
            "Ctrip Status",
            "Next step",
            "Next Action",
        }
    ]
    if semantic_columns:
        styler = styler.map(semantic_color, subset=semantic_columns)
    for column in [name for name in frame.columns if "Change %" in str(name)]:
        styler = styler.map(change_color, subset=[column])
    return styler

import streamlit as st


def kpi_card(
    title,
    value,
    subtitle,
    icon,
    color="#2563EB"
):

    st.markdown(
        f"""
<div style="
background:white;
padding:22px;
border-radius:18px;
border:1px solid #E5E7EB;
box-shadow:0 2px 8px rgba(0,0,0,.05);
height:165px;
">

<div style="
font-size:34px;
">
{icon}
</div>

<div style="
font-size:16px;
color:#6B7280;
margin-top:12px;
">
{title}
</div>

<div style="
font-size:38px;
font-weight:700;
color:{color};
">
{value}
</div>

<div style="
font-size:14px;
color:#9CA3AF;
">
{subtitle}
</div>

</div>
""",
        unsafe_allow_html=True
    )
import streamlit as st


def show_kpis(df):

    c1,c2,c3,c4,c5 = st.columns(5)

    c1.metric(
        "Hotels",
        len(df)
    )

    c2.metric(
        "Critical",
        len(df[df.Priority=="Critical"])
    )

    c3.metric(
        "High",
        len(df[df.Priority=="High"])
    )

    c4.metric(
        "Revenue",
        f"฿{df['Revenue'].sum():,.0f}"
    )

    c5.metric(
        "Avg Opportunity",
        round(df["Opportunity Index"].mean(),1)
    )
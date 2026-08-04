import streamlit as st


def hotel_card(row):

    with st.container(border=True):

        left,right = st.columns([5,1])

        with left:

            st.markdown(
                f"### 🏨 {row['ProductName']}"
            )

            st.caption(
                f"{row['Region']} • Rank #{int(row['Commercial Rank'])}"
            )

            c1,c2,c3 = st.columns(3)

            c1.metric(
                "Demand",
                round(row["Demand_Score"],1)
            )

            c2.metric(
                "Opportunity",
                round(row["Opportunity Index"],1)
            )

            c3.metric(
                "Revenue",
                f"฿{row['Revenue']:,.0f}"
            )

            st.write(
                f"**Priority:** {row['Priority']}"
            )

            st.write(
                f"**Action:** {row['Action']}"
            )

        with right:

            st.button(
                "Open",
                key=f"hotel_{row.name}"
            )
def ai_summary(row):

    text = []

    if row["Priority"]=="Critical":

        text.append(
            "This hotel requires immediate attention."
        )

    if row["Demand_Score"]>70:

        text.append(
            "Customer demand is currently strong."
        )

    if row["Agoda Status"]!="Mapped":

        text.append(
            "Agoda mapping is the highest priority."
        )

    if row["Revenue"]>500000:

        text.append(
            "Revenue contribution is significant."
        )

    text.append(
        f"Recommended action: {row['Action']}."
    )

    return " ".join(text)
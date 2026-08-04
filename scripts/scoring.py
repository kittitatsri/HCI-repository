import pandas as pd


def calculate_revenue_score(engine):

    max_revenue = engine["Revenue"].max()

    engine["Revenue Score v2"] = (
        engine["Revenue"] / max_revenue
    ) * 100

    engine["Revenue Score v2"] = (
        engine["Revenue Score v2"]
        .round(2)
    )

    return engine


def calculate_rn_score(engine):

    max_rn = engine["B2B2C RN"].max()

    engine["RN Score v2"] = (
        engine["B2B2C RN"] / max_rn
    ) * 100

    engine["RN Score v2"] = (
        engine["RN Score v2"]
        .round(2)
    )

    return engine


def calculate_mapping_score(engine):

    def mapping(agoda, ctrip):

        score = 0

        if agoda == "Mapped":
            score += 50

        if ctrip == "Mapped":
            score += 50

        return score

    engine["Mapping Score v2"] = engine.apply(
        lambda row: mapping(
            row["Agoda Status"],
            row["Ctrip Status"]
        ),
        axis=1
    )

    return engine


def calculate_business_score(engine):

    engine["Business Score v2"] = (

        engine["Revenue Score v2"] * 0.5

        +

        engine["RN Score v2"] * 0.3

        +

        engine["Demand_Score"] * 0.2

    )

    engine["Business Score v2"] = (
        engine["Business Score v2"]
        .round(2)
    )

    return engine


def calculate_opportunity_score(engine):

    engine["Opportunity Score v2"] = (

        engine["Business Score v2"]

        *

        (100 - engine["Mapping Score v2"])

        / 100

    )

    engine["Opportunity Score v2"] = (
        engine["Opportunity Score v2"]
        .round(2)
    )

    return engine
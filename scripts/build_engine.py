import pandas as pd


def build_engine(summary, master, performance):

    # -----------------------------
    # Remove duplicate ProductIDs
    # -----------------------------

    master = master.drop_duplicates(
        subset=["ProductID", "ProductName"],
        keep="first"
    )

    performance = performance.drop_duplicates(
        subset=["ProductID", "ProductName"],
        keep="first"
    )

    # -----------------------------
    # Merge Demand Summary + Master
    # -----------------------------

    engine = summary.merge(
        master,
        on=["ProductID", "ProductName"],
        how="left"
    )

    # -----------------------------
    # Merge Performance
    # -----------------------------

    engine = engine.merge(
        performance,
        on=["ProductID", "ProductName"],
        how="left"
    )

    print("=" * 60)
    print("ENGINE VALIDATION")
    print("=" * 60)

    print("Rows:", len(engine))
    print("Unique ProductIDs:", engine["ProductID"].nunique())

    if len(engine) == engine["ProductID"].nunique():
       print("✅ No duplicate ProductIDs")
    else:
       print("❌ Duplicate ProductIDs detected")

    return engine
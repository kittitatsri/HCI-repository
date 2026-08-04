from scripts.pipeline import run_pipeline


if __name__ == "__main__":
    engine, demand, source = run_pipeline()
    print(f"Demand source: {source.name}")
    print(f"Demand rows: {len(demand):,}")
    print(f"Engine hotels: {len(engine):,}")
    print(engine["Priority"].value_counts().to_string())

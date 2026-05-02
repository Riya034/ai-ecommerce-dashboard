import pandas as pd
import os

def run_feature_engineering():
    print("Running feature engineering...")

    df = pd.read_csv("data/silver/data.csv")

    # Clean country properly
    df["country"] = df["country"].str.strip()

    df["country"] = df["country"].replace({
        "UK": "United Kingdom",
        "EIRE": "Ireland"
    })

    # Features
    df["total"] = df["price"] * df["quantity"]
    df["avg_price_per_user"] = df.groupby("user_id")["price"].transform("mean")

    os.makedirs("data/gold", exist_ok=True)
    df.to_csv("data/gold/data.csv", index=False)

    print("Feature engineering completed")
   
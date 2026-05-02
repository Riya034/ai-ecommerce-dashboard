import pandas as pd
import os

def run_ingestion():
    print("Running ingestion...")

    df = pd.read_csv(
        "data/raw.csv",
        encoding="ISO-8859-1",
        engine="python",
        on_bad_lines="skip"
    )

    print("Original Columns:", df.columns)

    # Clean column names
    df.columns = df.columns.str.strip()

    # 🔥 UPDATED MAPPING (THIS IS THE FIX)
    rename_map = {
        "Customer ID": "user_id",
        "StockCode": "product_id",
        "Price": "price",
        "Quantity": "quantity",
        "Country": "country"
    }

    df = df.rename(columns=rename_map)

    print("Columns after rename:", df.columns)

    # Required columns
    required_cols = ["user_id", "product_id", "price", "quantity", "country"]

    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"❌ Missing column: {col}")

    df = df[required_cols]

    # Cleaning
    df = df.dropna(subset=["user_id"])
    df = df[df["price"] > 0]
    df = df[df["quantity"] > 0]

    df["user_id"] = df["user_id"].astype(int)

    os.makedirs("data/bronze", exist_ok=True)
    df.to_csv("data/bronze/data.csv", index=False)

    print("Ingestion completed")
   
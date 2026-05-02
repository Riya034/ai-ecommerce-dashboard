import pandas as pd
import os

def run_preprocessing():
    print("Running preprocessing...")

    df = pd.read_csv("data/bronze/data.csv")

    df = df.dropna()

    os.makedirs("data/silver", exist_ok=True)
    df.to_csv("data/silver/data.csv", index=False)

    print("Preprocessing completed")
   
    
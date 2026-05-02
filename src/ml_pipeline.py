import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

def run_training():
    print("Running training...")

    df = pd.read_csv("data/gold/data.csv")

    X = df[["price", "quantity", "avg_price_per_user"]]
    y = df["total"]

    model = LinearRegression()
    model.fit(X, y)

    joblib.dump(model, "model.pkl")

    print("Model trained & saved")
   
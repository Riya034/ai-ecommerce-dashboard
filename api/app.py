from fastapi import FastAPI
import pandas as pd

app = FastAPI()

@app.get("/")
def home():
    return {"message": "API is running"}

@app.get("/data")
def get_data():
    df = pd.read_csv("data/gold/data.csv")
    return df.to_dict(orient="records")
    
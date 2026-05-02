from src.ingestion import run_ingestion
from src.preprocessing import run_preprocessing
from src.feature_engineering import run_feature_engineering
from src.ml_pipeline import run_training
from src.evaluation import run_evaluation

def run_pipeline():
    print("Starting pipeline...")

    run_ingestion()
    run_preprocessing()
    run_feature_engineering()
    run_training()
    run_evaluation()

    print("Pipeline completed!")

if __name__ == "__main__":
    run_pipeline()
   
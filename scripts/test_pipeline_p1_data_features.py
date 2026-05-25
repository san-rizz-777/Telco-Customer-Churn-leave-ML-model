import os
import sys

# Now python can find the src package
sys.path.append(os.path.abspath("src"))

from data.load_data import load_data
from data.preprocess import preprocess
from features.build_features import build_features

"""
Testing the first phase of pipeline -> data loading, processing and feature engineering.
"""

DATA_PATH = "data/raw_data/telco_customer_churn.csv"
TARGET_COL = "Churn"
OUTPUT_DIR = "data/processed_data"

def main():
    print("Testing the phase 1:- Load -> Process -> Build features.")

    # Load the data
    print("Loading the data.....")
    df = load_data(DATA_PATH)
    print(f"Loaded the data :- Shape - {df.shape}")
    print(f"{df.sample(3)}")

    # Pre-process the data
    print("Preprocessing the data.....")
    df_clean = preprocess(df, target_col=TARGET_COL)
    print(f"Data preprocessed :- Shape - {df_clean.shape}")
    print(f"{df_clean.sample(3)}")

    # Build the features
    print("Feature engineering the data.....")
    df_features = build_features(df_clean, target_col=TARGET_COL)
    print(f"Feature engineered :- Shape - {df_features.shape}")
    print(f"{df_features.sample(3)}")

    print("Phase 1 testing of the pipeline completed successfully!!!!")

if __name__ == "__main__":
    main()







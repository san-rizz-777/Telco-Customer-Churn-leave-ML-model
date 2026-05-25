import os
import sys
import time
import argparse
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
recall_score, precision_score, f1_score, roc_auc_score, classification_report
)
from xgboost import XGBClassifier
import json
import joblib

# Fix the import path
sys.path.append(os.path.abspath("src"))

# Local modules
from data.load_data import load_data
from data.preprocess import preprocess
from features.build_features import build_features
from utils.validate import validate_data


def main(args):
    """
    Main training pipeline function that orchestrates the total workflow.
    """

    # MLflow Setup - ESSENTIAL for experiment tracking (logging the artifacts, metrics.)
    # Configure MLflow to use local file-based tracking (not a tracking server)
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    mlruns_path = args.mlflow_uri or f"file://{project_root}/mlruns"  # Local file-based tracking
    mlflow.set_tracking_uri(mlruns_path)
    mlflow.set_experiment(args.experiment)  # Creates experiment if doesn't exist

    # Start MLflow run - all subsequent logging will be tracked under this run
    with mlflow.start_run():
        # Log hyperparameters and configuration
        #  These parameters are essential for model reproducibility
        mlflow.log_param("model", "xgboost")  # Model type for comparison
        mlflow.log_param("threshold", args.threshold)  # Classification threshold (default: 0.3)
        mlflow.log_param("test_size", args.test_size)  # Train/test split ratio

        # Step 1 -> Data loading and validation.

        # i) Loading
        print("Loading data...")
        df = load_data(args.input)   # raw csv data
        print(f"Data looks like -> {df.shape[0]} rows and {df.shape[1]} columns.")

        # ii) Validation
        # This step is ESSENTIAL for production ML - validates data quality before training
        print("Validating the data......")
        is_valid, failed = validate_data(df)
        mlflow.log_metric("data_quality_pass", int(is_valid))   # Tracking thee data

        if not is_valid:
            # Log the validation failures for the debugging purpose
            mlflow.log_text(json.dumps(failed, indent=2), "failed_expectations.json")
            raise ValueError(f"Data quality check failed. Issues: {failed}")
        else:
            print("Data quality check passed and logged to mlflow.")

        # Step 2 :-> Data preprocessing.
        print("Preprocessing data...")
        df = preprocess(df)

        # Save the processed data for reproducibility and debuggging
        processed_path = os.path.join(project_root, "data", "processed_data", "telco_churn_processed.csv")
        os.makedirs(os.path.dirname(processed_path), exist_ok=True)
        df.to_csv(processed_path, index=False)
        print(f"Data preprocessed and saved to {processed_path} with shape {df.shape}.")

        # Step 3 - Feature engineering
        print("Building features.....")
        target = args.target
        if target not in df.columns:
            raise ValueError(f"Target column -> {target} not found in dataframe.")

        # Apply feature engineering
        df_features = build_features(df)

        # convert the boolean columns to int for XGBoost compatibility
        for c in df_features.select_dtypes(include=["object"]).columns:
            df_features[c] = df_features[c].astype(int)

        # save feature metadata for serving consistency
        artifacts_dir = os.path.join(project_root, "artifacts")
        os.makedirs(artifacts_dir, exist_ok=True)

        # Get the feature columns
        features_columns = list(df_features.drop(columns=[target]).columns)

        # Save locally for development serving
        with open(os.path.join(artifacts_dir, "features_columns.json"), "w") as f:
            json.dump(features_columns, f)

        # Log to mlflow for production serving
        mlflow.log_text('\n'.join(features_columns), artifact_file='features_columns.txt')

        # Save the preprocessing artifacts for serving purpose(insures exact feature order)
        preprocessing_artifact = {"features": features_columns, "target": target}

        # For serving consistency
        joblib.dump(preprocessing_artifact, os.path.join(artifacts_dir, "preprocessing.pkl"))
        mlflow.log_artifact(os.path.join(artifacts_dir, "preprocessing.pkl"))
        print(f"Saved the {len(features_columns)} feature columns.")

        # Step 4 - Train and test split
        print("Splitting data...")

        X = df_features.drop(columns=[target])
        y = df_features[target]

        # stratified train-test split to maintain class distribution in sets
        X_train, X_test , y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

        print(f"Train:- {X_train.shape[0]} samples and Test:- {X_test.shape[0]} samples.")

        # Calculate scale_pos_weight to handle imbalanced dataset
        # This tells XGBoost to give more weight to the minority class (churners)
        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
        print(f"Class imbalance ratio:- {scale_pos_weight} applied to positive class.")

        ##### Step 5 :- Training and tuning the model with XGBoost and optuna ###
        print("Training XGBoost model...")

        model = XGBClassifier(
            # Tree structure params
            n_estimators=600, learning_rate=0.0123, max_depth=3,

            # Regularisation params
            subsample=0.999, colsample_bytree=0.791, gamma=4.982385648061852,

            # Performance params
            n_jobs=-1, random_state=42, eval_metric="logloss",

            # imbalance in tree
            scale_pos_weight=scale_pos_weight
            )

        # Train the model and track the time
        start_train = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - start_train
        mlflow.log_metric("train_time", train_time)  # Track the training performance
        print(f"Model trained in {train_time: .2f} seconds.")


        # Step 6 :- Model Evaluation
        print("Evaluating model...")

        # Apply classification threshold (default: 0.35, optimized for churn detection)
        # Lower threshold = more sensitive to churn (higher recall, lower precision)
        start_test = time.time()
        proba = model.predict_proba(X_test)[:, 1]
        y_pred = (proba >= args.threshold).astype(int)
        test_time = time.time() - start_test

        mlflow.log_metric("test_time", test_time)  # Track the testing time

        # These metrics are essential for model comparison and monitoring
        precision = precision_score(y_test, y_pred)  # of predicted churners, how many actually churned???
        recall = recall_score(y_test, y_pred)   # of the actual churners, how many did we catch????
        f1 = f1_score(y_test, y_pred)    # balance of precision and recall
        roc_auc = roc_auc_score(y_test, proba)   # threshold independent

        # Log all the metrics for experiment tracking
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1", f1)
        mlflow.log_metric("roc_auc", roc_auc)

        print(f" Model Performance:- ")
        print(f"   Precision: {precision:.3f} and  Recall: {recall:.3f}")
        print(f"   F1 Score: {f1:.3f} and  ROC AUC: {roc_auc:.3f}")

        # Step 7 :- Model Serialization and logging
        print("Saving the model to MLflow....")

        # logging model in mlflow's standard format
        mlflow.sklearn.log_model(model, artifact_path="model")

        print("Model saved to MLflow for serving pipeline.")

        # === Final Performance Summary ===
        print(f"\n  Performance Summary:")
        print(f"   Training time: {train_time:.2f}s")
        print(f"   Inference time: {test_time:.4f}s")
        print(f"   Samples per second: {len(X_test) / test_time:.0f}")

        print(f"\n Detailed Classification Report:")
        print(classification_report(y_test, y_pred, digits=3))

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Run churn pipeline with XGBoost + MLflow")
    p.add_argument("--input", type=str, required=True,
                   help="path to CSV (e.g., data/raw/Telco-Customer-Churn.csv)")
    p.add_argument("--target", type=str, default="Churn")
    p.add_argument("--threshold", type=float, default=0.35)
    p.add_argument("--test_size", type=float, default=0.2)
    p.add_argument("--experiment", type=str, default="Telco Churn")
    p.add_argument("--mlflow_uri", type=str, default=None,
                   help="override MLflow tracking URI, else uses project_root/mlruns")

    args = p.parse_args()
    main(args)






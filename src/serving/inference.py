"""
This module provides the core inference functionality for the Telco Churn prediction model.
It ensures that serving-time feature transformations exactly match training-time transformations,
which is CRITICAL for model accuracy in production.

Key Responsibilities:
1. Load MLflow-logged model and feature metadata from training
2. Apply identical feature transformations as used during training
3. Ensure correct feature ordering for model input
4. Convert model predictions to user-friendly output

CRITICAL PATTERN: Training/Serving Consistency
- Uses fixed BINARY_MAP for deterministic binary encoding
- Applies same one-hot encoding with drop_first=True
- Maintains exact feature column order from training
- Handles missing/new categorical values gracefully

Production Deployment:
- MODEL_DIR points to containerized model artifacts
- Feature schema loaded from training-time artifacts
- Optimized for single-row inference (real-time serving)
"""

import os
import pandas as pd
import mlflow
import glob

##### MODEL LOADING CONFIGURATION
# Important: This path is set during Docker container build
# In development: uses local MLflow artifacts
# In production: uses model copied to container at build time
MODEL_DIR = "/app/model"

try:
    # Load the trained XGBoost model in MLflow pyfunc format
    # This ensures compatibility regardless of the underlying ML library
    model = mlflow.pyfunc.load_model(MODEL_DIR)
    print(f"Model loaded successfully from {MODEL_DIR}")
except Exception as e:
    print(f"Failed to load model from {MODEL_DIR}: {e}")

    # Fallback for local development
    try:
        # Try loading from local MLflow tracking
        local_model_paths = glob.glob("./mlruns/*/*/artifacts/model")
        if local_model_paths:
            latest_model = max(local_model_paths, key=os.path.getmtime)
            model = mlflow.pyfunc.load_model(latest_model)
            MODEL_DIR = latest_model
            print(f"Fallback: Loaded model from {latest_model}")
        else:
            raise Exception("No model found in local mlruns")
    except Exception as fallback_error:
        raise Exception(f"Failed to load model: {e}. Fallback failed: {fallback_error}")

#####  FEATURE SCHEMA LOADING
#Load the exact feature column order used during training (could be fatal....)
# This ensures the model receives features in the expected order
try:
    feature_file = os.path.join(MODEL_DIR, "feature_columns.txt")
    with open(feature_file) as f:
        FEATURE_COLS = [ln.strip() for ln in f if ln.strip()]
    print(f"Loaded {len(FEATURE_COLS)} feature columns from training")
except Exception as e:
    raise Exception(f"Failed to load feature columns: {e}")


##### FEATURE TRANSFORMATION CONSTANTS
#These mappings must exactly match those used in training (imp part.....)
# Any changes here will cause train/serve skew and degrade model performance

# Deterministic binary feature mappings (consistent with training)
BINARY_MAP = {
    "gender": {"Female": 0, "Male": 1},  # Demographics
    "Partner": {"No": 0, "Yes": 1},  # Has partner
    "Dependents": {"No": 0, "Yes": 1},  # Has dependents
    "PhoneService": {"No": 0, "Yes": 1},  # Phone service
    "PaperlessBilling": {"No": 0, "Yes": 1},  # Billing preference
}

# Numeric columns that need type coercion
NUMERIC_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]


def _serve_transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply identical feature transformations as used during model training.

    This function is CRITICAL for production ML - it ensures that features are
    transformed exactly as they were during training to prevent train/serve skew.

    Transformation Pipeline:
    1. Clean column names and handle data types
    2. Apply deterministic binary encoding (using BINARY_MAP)
    3. One-hot encode remaining categorical features
    4. Convert boolean columns to integers
    5. Align features with training schema and order

    Args:
        df: Single-row DataFrame with raw customer data

    Returns:
        DataFrame with features transformed and ordered for model input

    Note for me[Sanskar] :-  Any changes to this function must be reflected in training
    feature engineering to maintain consistency.
    """
    df = df.copy()

    # Clean column names (remove any whitespace)
    df.columns = df.columns.str.strip()

    #  STEP 1: Numeric Type Coercion
    # Ensure numeric columns are properly typed (handle string inputs)
    for c in NUMERIC_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
            df[c] = df[c].fillna(0)

    # STEP 2: Binary Feature Encoding
    # Apply deterministic mappings for binary features
    # Must use exact same mappings as training [[[important]]]
    for c, mapping in BINARY_MAP.items():
        if c in df.columns:
            df[c] = (
                df[c]
                .astype(str)
                .str.strip()
                .map(mapping)  # Apply binary mapping
                .astype("Int64")  # Handle NaN values
                .fillna(0)
                .astype(int)
            )

    #  STEP 3: One-Hot Encoding for Remaining Categorical Features
    # Find remaining object/categorical columns [ > 2 categories]
    obj_cols = [c for c in df.select_dtypes(include=["object"]).columns]
    if obj_cols:
        # Apply one-hot encoding with drop_first=True (same as training)
        # Prevents multicollinearity by dropping the first category
        df = pd.get_dummies(df, columns=obj_cols, drop_first=True)

    # STEP 4: Boolean to Integer Conversion
    # Convert any boolean columns to integers (XGBoost compatibility)
    bool_cols = df.select_dtypes(include=["bool"]).columns
    if len(bool_cols) > 0:
        df[bool_cols] = df[bool_cols].astype(int)

    #  STEP 5: Feature Alignment with Training Schema
    # Ensure features are in exact same order as training
    # Missing features get filled with 0, extra features are dropped
    df = df.reindex(columns=FEATURE_COLS, fill_value=0)

    return df


def predict(input_dict: dict) -> str:
    """
    Main prediction function for customer churn inference.

    -> This function provides the complete inference pipeline from raw customer data
    to business-friendly prediction output.
   -> It's called by both the FastAPI endpoint
    and the Gradio interface to ensure consistent predictions.

    Pipeline:
    1. Convert input dictionary to DataFrame
    2. Apply feature transformations (identical to training)
    3. Generate model prediction using loaded XGBoost model
    4. Convert prediction to user-friendly string

    Args:
        input_dict: Dictionary containing raw customer data with keys matching
                   the CustomerData schema (18 features total)

    Returns:
        Human-readable prediction string:
        - "Likely to churn" for high-risk customers (model prediction = 1)
        - "Not likely to churn" for low-risk customers (model prediction = 0)
    """

    # STEP 1: Convert Input to DataFrame
    df = pd.DataFrame([input_dict])

    # STEP 2: Apply Feature Transformations
    df_enc = _serve_transform(df)

    # STEP 3: Generate Model Prediction
    # Call the loaded MLflow model for inference
    # The model returns predictions in various formats depending on the ML library
    try:
        preds = model.predict(df_enc)

        # Normalize prediction output to consistent format
        if hasattr(preds, "tolist"):
            preds = preds.tolist()

        # Extract single prediction value (for single-row input)
        if isinstance(preds, (list, tuple)) and len(preds) == 1:
            result = preds[0]
        else:
            result = preds

    except Exception as e:
        raise Exception(f"Model prediction failed: {e}")

    #  STEP 4: Convert to Business-Friendly Output
    # Convert binary prediction (0/1) to actionable business language
    if result == 1:
        return "Likely to churn"  # High risk - needs intervention
    else:
        return "Not likely to churn"  # Low risk - maintain normal service
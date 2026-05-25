import pandas as pd


def map_binary_series(s: pd.Series) -> pd.Series:
    """
    Apply deterministic binary encoding to 2 category features.

    Implements binary encoding converting the features with only 2 categories to 0/1.
    Deterministic mapping i.e consistent while training and serving.
    """

    vals = list(pd.Series(s.dropna().unique()).astype(str))
    valset = set(vals)

    # Deterministic binary mappings
    # Same mappings are hard-coded in the serving pipeline

    # Yes/No mapping
    if valset == {"Yes", "No"}:
        return s.map({"No":0, "Yes":1}).astype("Int64")

    # Gender mapping[demographic feature]
    if valset == {"Male", "Female"}:
        return s.map({"Female":0, "Male":1}).astype("Int64")

    # Generic binary mapping
    # use alphabetical ordering for any other 2 category feature
    if len(vals) == 2:
        # Sort the values to ensure the consistent mapping across runs
        sorted_values = sorted(vals)
        mapping = {sorted_values[0]:0, sorted_values[1]:1}
        return s.map(mapping).astype("Int64")

    # One hot encoding will handle other features
    return s


def one_hot_encoding(s: list[str], df: pd.DataFrame) -> pd.DataFrame:
    """
   Apply one-hot encoding to more than 2 category features.
    """
    print("Applying one-hot encoding....")

    df = pd.get_dummies(df, columns=s, drop_first=True)

    return df


def build_features(df: pd.DataFrame, target_col:str = "Churn") -> pd.DataFrame:
    """
    Apply complete feature engineering pipeline for training data.

    Main feature engineering function transforms the raw data into ML-ready features.
    """

    df = df.copy()
    print(f"Starting feature engineering on {df.shape[1]} columns.....")

    # Separate the numeric and categorical columns
    num_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = [c for c in df.select_dtypes(include=["object"]).columns if c!=target_col] # Excluding the "Churn" column
    print(f"Found {len(cat_cols)} categorical features and {len(num_cols)} numerical features....")

    # Again separating the binary and multi-categorical columns and then
    binary_cols = [c for c in cat_cols if df[c].dropna().nunique()==2]
    multi_cols = [c for c in cat_cols if df[c].dropna().nunique() > 2]
    print(f"Found {len(binary_cols)} binary features and {len(multi_cols)} multi-label features....")

    if binary_cols:
        print(f"Binary: {binary_cols}")
    if multi_cols:
        print(f"Multi-label: {multi_cols}")

    # Apply binary encoding to binary label features and one hot encoding to multi-label features
    for c in binary_cols:
        ori_type = df[c].dtypes
        df[c] = map_binary_series(df[c].astype("str"))
        print(f"{c}: {ori_type} to binary[0/1]")

    # Apply OHE for multi-label columns
    ori_shape = df.shape

    df = one_hot_encoding(multi_cols, df)
    new_features = df.shape[1] - ori_shape[1] + len(multi_cols)

    print(f"Created {new_features} new features from {len(multi_cols)} categorical features.")


    # Convert the nullable integers(Int64) to standard integers(int) for XGBoost
    for c in binary_cols:
        if pd.api.types.is_integer_dtype(df[c]):
            # Fill any NaN values with 0 and convert to int
            df[c] = df[c].fillna(0).astype(int)

    print(f"Feature Engineering Completed!!!!:- {df.shape[1]} new features created.")

    return df



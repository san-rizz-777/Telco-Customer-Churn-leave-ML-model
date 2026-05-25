import pandas as pd


def preprocess(df: pd.DataFrame, target_col: str = "Churn") -> pd.DataFrame:
    """
    Basic cleaning for the Telco Churn
    - trim column names(strip off the whitespaces)
    - drop the ID cols
    - fix the TotalCharges to numeric
    - map the target Churn to 0/1 if needed
    - simple data cleaning -> handling the NA values
    """

    # neat and tidy headers
    df.columns = df.columns.str.strip()

    # drop the customer ids if present
    for col in ["customerID, CustomerID, customer_id"]:
        if col in df.columns:
            df = df.drop(columns=[col])

    # target to 0/1 if it's No/Yes
    if target_col in df.columns and df[target_col].dtype == "object":
        df[target_col] = df[target_col].str.strip().map({"No":0, "Yes":1})

    # Total Charges often blank in dataset coerce to float
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # Senior Citizen should be 0/1 int if present
    if "SeniorCitizen" in df.columns:
        df["SeniorCitizen"] = df["SeniorCitizen"].fillna(0).astype(int)

    # Strategy to fill NA values
    # for numerical columns fill with 0
    # for categorical the encoders will handle (get_dummies ignores the NaN)
    num_cols = df.select_dtypes(include=["number"]).columns
    df[num_cols] = df[num_cols].fillna(0)

    return df

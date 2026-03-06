import pandas as pd

def validate_transactions(df):

    required = ["client_id", "amount", "date"]

    for col in required:

        if col not in df.columns:
            raise ValueError(f"{col} column missing")

    df["date"] = pd.to_datetime(df["date"])

    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")

    df = df.dropna(subset=["amount"])

    if "type" not in df.columns:
        df["type"] = "income"

    if "category" not in df.columns:
        df["category"] = "Service"

    return df

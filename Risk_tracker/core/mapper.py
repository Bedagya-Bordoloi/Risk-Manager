import pandas as pd


def auto_map_columns(df):

    mapping = {}

    for col in df.columns:

        c = col.lower()

        # ------------------------------
        # CLIENT COLUMN
        # ------------------------------

        if "client" in c or "customer" in c or "company" in c:
            mapping[col] = "client_id"

        # ------------------------------
        # AMOUNT COLUMN
        # ------------------------------

        elif "amount" in c or "payment" in c or "value" in c or "revenue" in c:
            mapping[col] = "amount"

        # ------------------------------
        # DATE COLUMN
        # ------------------------------

        elif "date" in c or "paid" in c or "time" in c:
            mapping[col] = "date"

        # ------------------------------
        # TYPE COLUMN
        # ------------------------------

        elif "type" in c:
            mapping[col] = "type"

        # ------------------------------
        # CATEGORY COLUMN
        # ------------------------------

        elif "category" in c:
            mapping[col] = "category"

    df = df.rename(columns=mapping)

    # ------------------------------
    # SAFETY DEFAULTS
    # ------------------------------

    if "type" not in df.columns:
        df["type"] = "income"

    if "category" not in df.columns:
        df["category"] = "Service"

    return df

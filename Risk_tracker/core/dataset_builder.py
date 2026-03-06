import pandas as pd


def build_datasets_from_transactions(df_tx):

    # -------------------------------------
    # CLIENTS DATASET
    # -------------------------------------

    clients = df_tx["client_id"].dropna().unique()

    df_clients = pd.DataFrame({
        "client_id": clients,
        "industry": "Unknown",
        "contract_type": "Project",
        "base_value": 0
    })


    # -------------------------------------
    # INVOICES DATASET
    # -------------------------------------

    df_inv = df_tx.copy()

    df_inv["invoice_id"] = [
        f"INV_{i}" for i in range(len(df_inv))
    ]

    df_inv["due_date"] = pd.to_datetime(df_inv["date"]) - pd.Timedelta(days=30)

    df_inv["paid_date"] = pd.to_datetime(df_inv["date"])

    df_inv = df_inv[[
        "invoice_id",
        "client_id",
        "due_date",
        "paid_date",
        "amount"
    ]]

    return df_clients, df_inv

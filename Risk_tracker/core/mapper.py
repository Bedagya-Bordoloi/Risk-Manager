def auto_map_columns(df):

    mapping = {}

    for col in df.columns:

        c = col.lower()

        if "client" in c or "customer" in c:
            mapping[col] = "client_id"

        elif "payment" in c or "amount" in c or "revenue" in c:
            mapping[col] = "amount"

        elif "date" in c or "time" in c:
            mapping[col] = "date"

        elif "type" in c:
            mapping[col] = "type"

        elif "category" in c:
            mapping[col] = "category"

    df = df.rename(columns=mapping)

    return df

import pandas as pd
import numpy as np


def generate_churn_label(df):

    df["churn_label"] = (
        (df["recency"] > 90) |
        (df["revenue_drop_ratio"] < 0.5) |
        ((df["payment_count"] < 3) & (df["recency"] > 45))
    ).astype(int)

    return df


class AdaptiveDataPipeline:

    def __init__(self, tx_path, client_path, invoice_path):

        self.tx_path = tx_path
        self.client_path = client_path
        self.invoice_path = invoice_path


    def process(self):

        df_tx = pd.read_csv(self.tx_path)

        df_tx = df_tx.drop_duplicates()
        df_tx = df_tx.dropna(subset=["client_id", "amount"])

        df_tx["date"] = pd.to_datetime(df_tx["date"])


        # ------------------------------------------------
        # INVOICE DELAYS
        # ------------------------------------------------

        df_inv = pd.read_csv(self.invoice_path)

        df_inv["paid_date"] = pd.to_datetime(df_inv["paid_date"])
        df_inv["due_date"] = pd.to_datetime(df_inv["due_date"])

        df_inv["delay"] = (df_inv["paid_date"] - df_inv["due_date"]).dt.days

        avg_delays = df_inv.groupby("client_id")["delay"].mean().fillna(0)


        # ------------------------------------------------
        # FEATURE ENGINEERING
        # ------------------------------------------------

        income_df = df_tx[df_tx["type"] == "income"].sort_values(
            ["client_id", "date"]
        )

        total_biz_revenue = income_df["amount"].sum()

        today = df_tx["date"].max()


        def compute_features(group):

            n = len(group)

            total_rev = group["amount"].sum()

            rev_share = (total_rev / (total_biz_revenue + 1e-9)) * 100

            gaps = group["date"].diff().dt.days

            avg_gap = gaps.mean() if n > 1 else 30

            recency = (today - group["date"].max()).days


            # ---- volatility

            volatility = gaps.std()

            if pd.isna(volatility):
                volatility = 0

            volatility = volatility / (avg_gap + 1)


            # ---- revenue trend

            revenue_trend = 0

            if n >= 3:

                try:

                    revenue_trend = np.polyfit(
                        np.arange(n),
                        group["amount"],
                        1
                    )[0]

                except:
                    revenue_trend = 0


            # ---- revenue drop

            drop_ratio = 1

            if n >= 4:

                recent = group["amount"].tail(2).mean()
                older = group["amount"].iloc[:-2].mean()

                drop_ratio = recent / (older + 1e-9)


            return pd.Series({

                "total_revenue": total_rev,
                "revenue_share_%": rev_share,
                "payment_count": n,
                "avg_gap": avg_gap,
                "recency": recency,
                "volatility": volatility,
                "revenue_trend": revenue_trend,
                "revenue_drop_ratio": drop_ratio

            })


        client_stats = income_df.groupby(
            "client_id",
            group_keys=False
        ).apply(compute_features)


        client_stats["avg_payment_delay"] = client_stats.index.map(
            avg_delays
        ).fillna(0)


        client_stats = generate_churn_label(client_stats)


        print("\nChurn Distribution")
        print(client_stats["churn_label"].value_counts())


        # ------------------------------------------------
        # MONTHLY AGGREGATION
        # ------------------------------------------------

        monthly_stats = df_tx.resample(
            "MS",
            on="date"
        ).agg({"amount": "sum"})


        monthly_stats["revenue"] = df_tx[
            df_tx["type"] == "income"
        ].resample("MS", on="date")["amount"].sum()


        monthly_stats["expenses"] = df_tx[
            df_tx["type"] == "expense"
        ].resample("MS", on="date")["amount"].sum()


        monthly_stats = monthly_stats.fillna(0)


        monthly_stats["net_cash_flow"] = (
            monthly_stats["revenue"]
            - monthly_stats["expenses"]
        )


        return monthly_stats, client_stats

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from statsmodels.tsa.arima.model import ARIMA

class ChurnPredictorML:
    def __init__(self):
        # We use Random Forest because it handles non-linear business patterns well
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.is_trained = False

    def train(self, feature_df, labels):
        """Trains the model on historical client behavior."""
        if len(feature_df) < 10: return # Not enough data to learn
        self.model.fit(feature_df, labels)
        self.is_trained = True

    def predict_probs(self, feature_df):
        """Returns the raw ML churn probability (0 to 100)."""
        if not self.is_trained:
            return np.zeros(len(feature_df))
        return self.model.predict_proba(feature_df)[:, 1] * 100

class AdaptiveIntelligenceEngine:
    def __init__(self):
        self.ml_layer = ChurnPredictorML()

    def predict_lifecycle(self, row):
        # Lifecycle stages remain heuristic (Business Rules)
        if row['payment_count'] <= 2: return "NEW"
        if row['recency'] > 60: return "CHURNED"
        if row['revenue_trend'] < -10: return "DECLINING"
        if row['revenue_trend'] > 10: return "GROWING"
        if row['volatility'] < 0.2: return "STABLE"
        return "ACTIVE"

    def calculate_hybrid_risk(self, row, ml_prob):
        """
        The Hybrid Layer: Blends ML patterns with Domain Rules.
        ML_Prob: What the data says.
        Rules: What the business owner knows (Domain Knowledge).
        """
        # 1. Domain Heuristics (Safety Layer)
        rule_risk = 0
        if row['recency'] > (row['avg_gap'] * 1.5): rule_risk += 50
        if row['volatility'] > 0.5: rule_risk += 30
        rule_risk = min(rule_risk, 100)

        # 2. Adaptive Weighting
        # If we have lots of payments, trust the ML more (up to 80%)
        ml_weight = min(row['payment_count'] / 15, 0.8)
        
        final_score = (ml_prob * ml_weight) + (rule_risk * (1 - ml_weight))
        return round(final_score, 1)

class BusinessHealthEngine:
    def calculate_burnout_risk(self, client_stats):
        """Calculate overall business burnout risk based on client health."""
        high_risk_clients = (client_stats['RISK_%'] > 70).sum()
        total_clients = len(client_stats)
        burnout_percentage = (high_risk_clients / total_clients) * 100
        return round(burnout_percentage, 1)

class RevenueForecaster:
    def predict_net_cash_flow(self, monthly_stats):
        """Simple linear trend forecast for next 3 months."""
        if len(monthly_stats) < 3:
            return [0, 0, 0]
        
        recent = monthly_stats['net_cash_flow'].tail(3).values
        trend = np.polyfit(range(3), recent, 1)[0]
        
        forecast = []
        for i in range(1, 4):
            forecast.append(recent[-1] + (trend * i))
        
        return [round(x, 2) for x in forecast]

class HeuristicPricingOptimizer:
    def suggest_adjustment(self, row):
        """Suggest pricing strategy based on client metrics."""
        if row['STAGE'] == 'DECLINING':
            return "Reduce Rate 10%"
        elif row['STAGE'] == 'GROWING':
            return "Increase Rate 15%"
        elif row['RISK_%'] > 50:
            return "Maintain Rate"
        else:
            return "Increase Rate 5%"

class CLVEngine:
    def estimate_predictive_clv(self, row):
        """Estimate Customer Lifetime Value based on current metrics."""
        base_value = row['total_revenue'] / max(row['payment_count'], 1)
        months_active = row['payment_count'] * row['avg_gap'] / 30
        
        # Simple CLV estimation
        if row['STAGE'] == 'CHURNED':
            return round(row['total_revenue'] * 0.8, 2)
        elif row['STAGE'] == 'DECLINING':
            return round(row['total_revenue'] * 1.2, 2)
        elif row['STAGE'] == 'GROWING':
            return round(row['total_revenue'] * 2.5, 2)
        else:
            return round(row['total_revenue'] * 1.8, 2)
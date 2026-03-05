from core.data_pipelining import AdaptiveDataPipeline
from core.models import (AdaptiveIntelligenceEngine, ChurnPredictorML, BusinessHealthEngine, 
                         RevenueForecaster, HeuristicPricingOptimizer, CLVEngine)
from core.visualizer import plot_advanced_health  # Integrated Visualizer

def run_analysis_service():
    """
    Service Layer: Orchestrates Data Pipeline -> ML Training -> Decision Engines.
    Returns a unified intelligence report.
    """
    # 1. Pipeline: Chaos-Proof Data Ingestion
    pipeline = AdaptiveDataPipeline('data/transactions.csv', 'data/clients.csv', 'data/invoices.csv')
    monthly_stats, client_stats = pipeline.process()

    # 2. ML Training: Learning Churn Patterns from Feature Space
    feature_cols = ['payment_count', 'avg_gap', 'recency', 'volatility', 
                    'revenue_trend', 'revenue_drop_ratio', 'revenue_share_%']
    X = client_stats[feature_cols].fillna(0)
    
    # Label Generation: Clients who haven't paid in 45 days are 'Churned' in training
    y = (client_stats['recency'] > 45).astype(int)

    ml_model = ChurnPredictorML()
    ml_model.train(X, y)
    ml_probs = ml_model.predict_probs(X)

    # 3. Decision Engines: Hybrid Intelligence
    intel = AdaptiveIntelligenceEngine()
    health = BusinessHealthEngine()
    forecaster = RevenueForecaster()
    pricing = HeuristicPricingOptimizer()
    clv = CLVEngine()

    # Calculate Lifecycle & Hybrid Risk Score
    client_stats['STAGE'] = client_stats.apply(intel.predict_lifecycle, axis=1)
    
    risks = []
    for i in range(len(client_stats)):
        # ML + Heuristic Blending
        risk = intel.calculate_hybrid_risk(client_stats.iloc[i], ml_probs[i])
        risks.append(risk)
    client_stats['RISK_%'] = risks

    # Prescriptive Metrics (CLV & Pricing)
    client_stats['PREDICTIVE_CLV'] = client_stats.apply(clv.estimate_predictive_clv, axis=1)
    client_stats['PRICE_STRATEGY'] = client_stats.apply(pricing.suggest_adjustment, axis=1)
    
    # Business-Wide Health
    burnout = health.calculate_burnout_risk(client_stats)
    profit_forecast = forecaster.predict_net_cash_flow(monthly_stats)

    return {
        "burnout": burnout,
        "forecast": profit_forecast,
        "clients": client_stats,
        "monthly": monthly_stats,
        "ml_confidence": "High" if ml_model.is_trained else "Low"
    }

def display_dashboard():
    """
    Presentation Layer: Outputs the Priority Inbox and Visual Dashboard.
    """
    data = run_analysis_service()
    
    print("\n" + "═"*110)
    print(f" 🛡️  RISK INTELLIGENCE REPORT | 🔥 BURNOUT: {data['burnout']}% | 🧠 ML CONFIDENCE: {data['ml_confidence']}")
    print(f" 📈 90-DAY PROFIT FORECAST: ${data['forecast'][0]:,.0f} -> ${data['forecast'][1]:,.0f} -> ${data['forecast'][2]:,.0f}")
    print("═"*110)

    # --- THE PRIORITY ACTION INBOX ---
    print("\n 📬 PRIORITY ACTION INBOX (Top High-Impact Tasks):")
    
    # Find high-value clients who are declining or high-risk
    critical_clients = data['clients'][data['clients']['RISK_%'] > 40].sort_values('PREDICTIVE_CLV', ascending=False)
    
    if not critical_clients.empty:
        top_c = critical_clients.index[0]
        clv_val = critical_clients.loc[top_c, 'PREDICTIVE_CLV']
        stage = critical_clients.loc[top_c, 'STAGE']
        print(f" [!] RECOVER: Client {top_c} has high CLV (${clv_val:,.0f}) but is {stage}. Immediate retention needed.")
    
    if data['burnout'] > 65:
        print(f" [!] CAPACITY: Burnout Risk is Critical ({data['burnout']}%). Do not accept new projects for 14 days.")
    
    # Find upsell opportunities
    upsell_target = data['clients'][(data['clients']['STAGE'] == 'STABLE') & (data['clients']['RISK_%'] < 15)]
    if not upsell_target.empty:
        print(f" [!] REVENUE: Client {upsell_target.index[0]} is highly stable. Ideal for a standard rate adjustment.")

    # --- CLIENT PULSE TABLE ---
    print(f"\n {'CLIENT ID':<10} | {'STAGE':<12} | {'RISK':<6} | {'PRED. CLV':<12} | {'PRICING STRATEGY'}")
    print("-" * 110)
    
    # Sort by CLV to show the most important clients first
    for cid, row in data['clients'].sort_values('PREDICTIVE_CLV', ascending=False).head(10).iterrows():
        print(f" {cid:<10} | {row['STAGE']:<12} | {row['RISK_%']:>3.0f}% | $ {row['PREDICTIVE_CLV']:>10,.0f} | {row['PRICE_STRATEGY']}")

    print("\n" + "═"*110)

    # --- THE VISUAL TRACKER ---
    # Passing the real stats to the visualizer for the dark-mode dashboard
    plot_advanced_health(data['monthly'], data['burnout'])

if __name__ == "__main__":
    display_dashboard()

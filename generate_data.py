"""
Synthetic e-commerce dataset generator for the Customer LTV Prediction project.

Simulates 24 months of transaction history for ~3,000 customers across
5 acquisition channels and 6 product categories. Customer behavior is
generated from latent "value tiers" so that RFM features actually carry
predictive signal for future spend (mirrors real-world LTV patterns).
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

np.random.seed(42)

N_CUSTOMERS = 3000
START_DATE = datetime(2023, 7, 1)
END_DATE = datetime(2025, 6, 30)          # 24 months total
CHANNELS = ["Organic Search", "Paid Ads", "Email", "Social Media", "Referral"]
CATEGORIES = ["Electronics", "Apparel", "Home & Kitchen", "Beauty", "Sports", "Books"]

# ---------------------------------------------------------------
# 1. Customers table
# ---------------------------------------------------------------
customer_ids = [f"CUST{str(i).zfill(5)}" for i in range(1, N_CUSTOMERS + 1)]

# Latent value tier drives purchase frequency & spend (not visible to model directly)
value_tier = np.random.choice(
    ["Low", "Mid", "High", "VIP"], size=N_CUSTOMERS, p=[0.40, 0.35, 0.20, 0.05]
)
tier_lambda = {"Low": 0.4, "Mid": 1.1, "High": 2.3, "VIP": 4.5}      # avg orders/month
tier_aov_mean = {"Low": 550, "Mid": 950, "High": 1650, "VIP": 3200}  # INR

signup_days_offset = np.random.randint(0, 200, size=N_CUSTOMERS)
signup_dates = [START_DATE + timedelta(days=int(d)) for d in signup_days_offset]

customers = pd.DataFrame({
    "customer_id": customer_ids,
    "signup_date": signup_dates,
    "acquisition_channel": np.random.choice(CHANNELS, size=N_CUSTOMERS, p=[0.30, 0.25, 0.15, 0.20, 0.10]),
    "age": np.random.randint(19, 60, size=N_CUSTOMERS),
    "city_tier": np.random.choice(["Tier 1", "Tier 2", "Tier 3"], size=N_CUSTOMERS, p=[0.45, 0.35, 0.20]),
    "_value_tier": value_tier,  # kept for generation only, dropped before saving raw file
})

# ---------------------------------------------------------------
# 2. Transactions table
# ---------------------------------------------------------------
rows = []
txn_counter = 1

for idx, cust in customers.iterrows():
    cid = cust["customer_id"]
    tier = cust["_value_tier"]
    lam_month = tier_lambda[tier]
    aov_mean = tier_aov_mean[tier]

    active_from = cust["signup_date"]
    active_days = (END_DATE - active_from).days
    if active_days <= 0:
        continue

    # Simulate month-by-month purchases with mild churn risk over time
    n_months_active = max(1, active_days // 30)
    churn_point = np.random.randint(int(n_months_active * 0.3), n_months_active + 1) \
        if np.random.rand() < 0.25 else n_months_active  # 25% of customers churn early

    for m in range(min(n_months_active, churn_point)):
        n_orders = np.random.poisson(lam_month)
        for _ in range(n_orders):
            day_offset = m * 30 + np.random.randint(0, 30)
            txn_date = active_from + timedelta(days=int(day_offset))
            if txn_date > END_DATE:
                continue
            order_value = max(150, np.random.normal(aov_mean, aov_mean * 0.35))
            rows.append({
                "transaction_id": f"TXN{str(txn_counter).zfill(6)}",
                "customer_id": cid,
                "transaction_date": txn_date,
                "category": np.random.choice(CATEGORIES),
                "order_value": round(order_value, 2),
                "quantity": np.random.randint(1, 5),
            })
            txn_counter += 1

transactions = pd.DataFrame(rows)

customers_out = customers.drop(columns=["_value_tier"])

customers_out.to_csv("/home/claude/ltv-project/data/customers.csv", index=False)
transactions.to_csv("/home/claude/ltv-project/data/transactions.csv", index=False)

print(f"Customers: {len(customers_out)}")
print(f"Transactions: {len(transactions)}")
print(f"Date range: {transactions['transaction_date'].min()} -> {transactions['transaction_date'].max()}")

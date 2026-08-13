"""
Customer Lifetime Value (LTV) Prediction Pipeline
===================================================
Objective : Predict 12-month forward customer LTV from behavioral history,
            to support targeted marketing and retention prioritization.

Approach  : Time-based split to avoid data leakage.
              - Feature window : first 12 months of each customer's history
              - Label window   : following 12 months (actual LTV to predict)
            This mirrors how LTV models are built in production -- you
            can only use data you'd actually have *before* the prediction
            date.

Models    : Random Forest Regressor vs XGBoost Regressor (compared)
Metrics   : MAE, RMSE, R^2
Output    : Trained model, prediction CSV, evaluation charts, customer segments
"""

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import timedelta

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 110

DATA_DIR = "/home/claude/ltv-project/data"
OUT_DIR = "/home/claude/ltv-project/outputs"
VIZ_DIR = f"{OUT_DIR}/visualizations"
MODEL_DIR = "/home/claude/ltv-project/models"

# ---------------------------------------------------------------------
# 1. LOAD DATA
# ---------------------------------------------------------------------
customers = pd.read_csv(f"{DATA_DIR}/customers.csv", parse_dates=["signup_date"])
transactions = pd.read_csv(f"{DATA_DIR}/transactions.csv", parse_dates=["transaction_date"])

print(f"Loaded {len(customers):,} customers and {len(transactions):,} transactions.")

# ---------------------------------------------------------------------
# 2. TIME-BASED WINDOW SPLIT (prevents leakage)
# ---------------------------------------------------------------------
FEATURE_WINDOW_MONTHS = 12
CUTOFF_DATE = customers["signup_date"].min() + pd.DateOffset(months=FEATURE_WINDOW_MONTHS)
LABEL_END_DATE = CUTOFF_DATE + pd.DateOffset(months=FEATURE_WINDOW_MONTHS)

# Only keep customers who signed up early enough to have a full feature + label window
eligible_customers = customers[customers["signup_date"] <= CUTOFF_DATE - pd.DateOffset(months=1)].copy()

feature_txns = transactions[
    (transactions["transaction_date"] <= CUTOFF_DATE) &
    (transactions["customer_id"].isin(eligible_customers["customer_id"]))
]
label_txns = transactions[
    (transactions["transaction_date"] > CUTOFF_DATE) &
    (transactions["transaction_date"] <= LABEL_END_DATE) &
    (transactions["customer_id"].isin(eligible_customers["customer_id"]))
]

print(f"Feature window ends : {CUTOFF_DATE.date()}")
print(f"Label window ends   : {LABEL_END_DATE.date()}")
print(f"Eligible customers  : {len(eligible_customers):,}")

# ---------------------------------------------------------------------
# 3. FEATURE ENGINEERING (RFM + extras)
# ---------------------------------------------------------------------
snapshot_date = CUTOFF_DATE

agg = feature_txns.groupby("customer_id").agg(
    frequency=("transaction_id", "count"),
    monetary_total=("order_value", "sum"),
    aov=("order_value", "mean"),
    last_purchase=("transaction_date", "max"),
    first_purchase=("transaction_date", "min"),
    total_quantity=("quantity", "sum"),
    n_categories=("category", "nunique"),
).reset_index()

agg["recency_days"] = (snapshot_date - agg["last_purchase"]).dt.days
agg["tenure_days"] = (snapshot_date - agg["first_purchase"]).dt.days
agg["purchase_rate"] = agg["frequency"] / agg["tenure_days"].replace(0, 1) * 30  # orders/month

features = eligible_customers.merge(agg, on="customer_id", how="left")

# Customers with zero purchases in feature window -> fill with zeros / max recency
features["frequency"] = features["frequency"].fillna(0)
features["monetary_total"] = features["monetary_total"].fillna(0)
features["aov"] = features["aov"].fillna(0)
features["total_quantity"] = features["total_quantity"].fillna(0)
features["n_categories"] = features["n_categories"].fillna(0)
features["recency_days"] = features["recency_days"].fillna(FEATURE_WINDOW_MONTHS * 30)
features["tenure_days"] = (snapshot_date - features["signup_date"]).dt.days
features["purchase_rate"] = features["purchase_rate"].fillna(0)
features["account_age_days"] = (snapshot_date - features["signup_date"]).dt.days

# ---------------------------------------------------------------------
# 4. LABEL: actual LTV in the following 12-month window
# ---------------------------------------------------------------------
label_agg = label_txns.groupby("customer_id")["order_value"].sum().reset_index()
label_agg.columns = ["customer_id", "future_ltv"]

dataset = features.merge(label_agg, on="customer_id", how="left")
dataset["future_ltv"] = dataset["future_ltv"].fillna(0)

dataset.to_csv(f"{OUT_DIR}/feature_engineered_dataset.csv", index=False)
print(f"Final modeling dataset shape: {dataset.shape}")

# ---------------------------------------------------------------------
# 5. TRAIN / TEST SPLIT
# ---------------------------------------------------------------------
feature_cols = [
    "frequency", "monetary_total", "aov", "recency_days", "tenure_days",
    "purchase_rate", "total_quantity", "n_categories", "account_age_days", "age",
]
categorical_cols = ["acquisition_channel", "city_tier"]

X = pd.get_dummies(dataset[feature_cols + categorical_cols], columns=categorical_cols, drop_first=True)
y = dataset["future_ltv"]

X_train, X_test, y_train, y_test, cust_train, cust_test = train_test_split(
    X, y, dataset["customer_id"], test_size=0.2, random_state=42
)

print(f"Train size: {len(X_train)}  |  Test size: {len(X_test)}")

# ---------------------------------------------------------------------
# 6. MODEL TRAINING — Random Forest vs XGBoost
# ---------------------------------------------------------------------
rf = RandomForestRegressor(
    n_estimators=300, max_depth=8, min_samples_leaf=5, random_state=42, n_jobs=-1
)
rf.fit(X_train, y_train)
rf_preds = rf.predict(X_test)

xgb = XGBRegressor(
    n_estimators=400, max_depth=5, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1
)
xgb.fit(X_train, y_train)
xgb_preds = xgb.predict(X_test)

def evaluate(y_true, y_pred, name):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(f"{name:15s} | MAE: {mae:8.2f} | RMSE: {rmse:8.2f} | R2: {r2:.4f}")
    return {"model": name, "MAE": mae, "RMSE": rmse, "R2": r2}

print("\n--- Model Evaluation (Test Set) ---")
results = [
    evaluate(y_test, rf_preds, "Random Forest"),
    evaluate(y_test, xgb_preds, "XGBoost"),
]
results_df = pd.DataFrame(results)
results_df.to_csv(f"{OUT_DIR}/model_comparison.csv", index=False)

# Pick best model by RMSE
best_row = results_df.loc[results_df["RMSE"].idxmin()]
best_model = xgb if best_row["model"] == "XGBoost" else rf
best_preds_test = xgb_preds if best_row["model"] == "XGBoost" else rf_preds
print(f"\nBest model: {best_row['model']}")

joblib.dump(best_model, f"{MODEL_DIR}/ltv_model.pkl")

# ---------------------------------------------------------------------
# 7. FULL-DATASET PREDICTIONS + SEGMENTATION
# ---------------------------------------------------------------------
dataset["predicted_ltv"] = best_model.predict(X)
dataset["predicted_ltv"] = dataset["predicted_ltv"].clip(lower=0)

def segment(ltv, q):
    if ltv >= q[0.9]:
        return "VIP"
    elif ltv >= q[0.7]:
        return "High"
    elif ltv >= q[0.4]:
        return "Medium"
    else:
        return "Low"

quantiles = dataset["predicted_ltv"].quantile([0.4, 0.7, 0.9]).to_dict()
dataset["ltv_segment"] = dataset["predicted_ltv"].apply(lambda x: segment(x, quantiles))

final_output = dataset[[
    "customer_id", "acquisition_channel", "city_tier", "frequency", "monetary_total",
    "aov", "recency_days", "purchase_rate", "future_ltv", "predicted_ltv", "ltv_segment"
]].sort_values("predicted_ltv", ascending=False)

final_output.to_csv(f"{OUT_DIR}/ltv_predictions.csv", index=False)
print(f"\nSaved predictions for {len(final_output):,} customers -> ltv_predictions.csv")
print(final_output["ltv_segment"].value_counts())

# ---------------------------------------------------------------------
# 8. VISUALIZATIONS
# ---------------------------------------------------------------------

# 8a. Predicted vs Actual scatter (test set, best model)
plt.figure(figsize=(7, 6))
plt.scatter(y_test, best_preds_test, alpha=0.35, s=18, color="#2563EB")
lims = [0, max(y_test.max(), best_preds_test.max())]
plt.plot(lims, lims, "--", color="gray", linewidth=1.5, label="Perfect Prediction")
plt.xlabel("Actual 12-Month LTV (₹)")
plt.ylabel("Predicted 12-Month LTV (₹)")
plt.title(f"Predicted vs Actual LTV — {best_row['model']} (Test Set)")
plt.legend()
plt.tight_layout()
plt.savefig(f"{VIZ_DIR}/predicted_vs_actual.png")
plt.close()

# 8b. Feature importance
importances = pd.Series(best_model.feature_importances_, index=X.columns).sort_values(ascending=False).head(10)
plt.figure(figsize=(8, 6))
sns.barplot(x=importances.values, y=importances.index, color="#2563EB")
plt.title(f"Top 10 Feature Importances — {best_row['model']}")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig(f"{VIZ_DIR}/feature_importance.png")
plt.close()

# 8c. LTV distribution by segment
plt.figure(figsize=(8, 5))
order = ["Low", "Medium", "High", "VIP"]
sns.boxplot(data=dataset, x="ltv_segment", y="predicted_ltv", order=order, palette="Blues")
plt.title("Predicted LTV Distribution by Customer Segment")
plt.xlabel("Segment")
plt.ylabel("Predicted 12-Month LTV (₹)")
plt.tight_layout()
plt.savefig(f"{VIZ_DIR}/ltv_by_segment.png")
plt.close()

# 8d. Segment counts + revenue share
seg_summary = dataset.groupby("ltv_segment").agg(
    customers=("customer_id", "count"),
    total_predicted_ltv=("predicted_ltv", "sum")
).reindex(order)
seg_summary["revenue_share_pct"] = (seg_summary["total_predicted_ltv"] / seg_summary["total_predicted_ltv"].sum() * 100).round(1)
seg_summary.to_csv(f"{OUT_DIR}/segment_summary.csv")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
sns.barplot(x=seg_summary.index, y=seg_summary["customers"], order=order, color="#3B82F6", ax=axes[0])
axes[0].set_title("Customer Count by Segment")
axes[0].set_ylabel("Number of Customers")
axes[1].pie(seg_summary["total_predicted_ltv"], labels=seg_summary.index, autopct="%1.1f%%",
            colors=["#BFDBFE", "#60A5FA", "#2563EB", "#1E3A8A"])
axes[1].set_title("Share of Total Predicted LTV")
plt.tight_layout()
plt.savefig(f"{VIZ_DIR}/segment_summary.png")
plt.close()

# 8e. Model comparison bar chart
plt.figure(figsize=(7, 5))
x = np.arange(len(results_df))
width = 0.35
fig, ax1 = plt.subplots(figsize=(7, 5))
ax1.bar(x - width/2, results_df["MAE"], width, label="MAE", color="#60A5FA")
ax1.bar(x + width/2, results_df["RMSE"], width, label="RMSE", color="#1E3A8A")
ax1.set_xticks(x)
ax1.set_xticklabels(results_df["model"])
ax1.set_ylabel("Error (₹)")
ax1.set_title("Model Comparison — MAE & RMSE (lower is better)")
ax1.legend()
plt.tight_layout()
plt.savefig(f"{VIZ_DIR}/model_comparison.png")
plt.close()

print("\nAll visualizations saved to:", VIZ_DIR)
print("Pipeline complete.")

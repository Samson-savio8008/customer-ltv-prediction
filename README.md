# Customer Lifetime Value (LTV) Prediction Model

Predicting 12-month forward customer value from purchase behavior, to help prioritize retention and marketing spend on the customers who matter most.

## Business Problem

Not all customers are worth the same to a business — but most marketing budgets get spent as if they were. This project builds a regression model that predicts each customer's **future 12-month spend** from their historical purchase behavior, then segments the customer base into actionable value tiers (Low / Medium / High / VIP) so marketing and retention efforts can be targeted where they'll return the most.

## Dataset

Synthetic e-commerce data generated to mimic realistic customer behavior:
- **3,000 customers** across 5 acquisition channels (Organic Search, Paid Ads, Email, Social Media, Referral) and 3 city tiers
- **~70,000 transactions** spanning a 24-month window, across 6 product categories
- Purchase frequency and order value are generated from four latent customer value tiers (Low/Mid/High/VIP), so the resulting RFM signal mirrors what you'd see in real transaction data

> Since real transactional data is proprietary, this project uses a generated dataset (`data/generate_data.py`) built to reflect realistic RFM (Recency, Frequency, Monetary) distributions and customer value skew.

## Methodology

**The key design choice in this project is a time-based train/label split** — a common leakage mistake in LTV projects is engineering features and labels from overlapping time windows. Here:

- **Feature window:** each customer's first 12 months of activity (what you'd actually know at prediction time)
- **Label window:** the following 12 months (the actual future value we're trying to predict)

This means the model is validated the way it would actually be used in production — predicting forward, not backward.

**Feature engineering (RFM + behavioral):**
- Recency (days since last purchase), Frequency (order count), Monetary total & Average Order Value
- Purchase rate (orders/month), category diversity, tenure, account age
- Acquisition channel and city tier (one-hot encoded)

**Models compared:** Random Forest Regressor vs. XGBoost Regressor, evaluated on a held-out 20% test set using MAE, RMSE, and R².

## Results

| Model | MAE (₹) | RMSE (₹) | R² |
|---|---|---|---|
| **Random Forest (selected)** | 6,246 | 13,089 | **0.871** |
| XGBoost | 6,818 | 13,701 | 0.859 |

Random Forest was selected as the production model based on lower error and higher explained variance. **Average order value and recency were the strongest predictors** of future LTV, consistent with classic RFM theory.

### Customer Segmentation

Scoring the full customer base and bucketing by predicted LTV quantile surfaces a clear concentration of value:

| Segment | Customers | Share of Predicted Revenue |
|---|---|---|
| VIP | 300 (10%) | **50.0%** |
| High | 600 (20%) | 30.4% |
| Medium | 900 (30%) | 15.0% |
| Low | 1,200 (40%) | 4.6% |

**Key insight:** the top 10% of customers (VIP segment) are predicted to drive half of all future revenue — a textbook 80/20-style pattern that should directly inform where retention and loyalty budget is spent.

![Segment Summary](outputs/visualizations/segment_summary.png)
![Predicted vs Actual](outputs/visualizations/predicted_vs_actual.png)

## Repository Structure

```
├── data/
│   ├── generate_data.py          # Synthetic dataset generator
│   ├── customers.csv              # Customer master data
│   └── transactions.csv           # Transaction-level log
├── notebooks/
│   └── Customer_LTV_Prediction.ipynb   # Full analysis notebook (executed, with outputs)
├── src/
│   └── ltv_pipeline.py            # Production-style script version of the full pipeline
├── models/
│   └── ltv_model.pkl               # Trained Random Forest model
├── outputs/
│   ├── ltv_predictions.csv         # Final scored customer list (all 3,000 customers)
│   ├── feature_engineered_dataset.csv
│   ├── model_comparison.csv
│   ├── segment_summary.csv
│   └── visualizations/             # All charts (PNG)
└── requirements.txt
```

## How to Run

```bash
pip install -r requirements.txt
python data/generate_data.py        # generates customers.csv and transactions.csv
python src/ltv_pipeline.py          # runs full pipeline, saves model + predictions + charts
```

Or open `notebooks/Customer_LTV_Prediction.ipynb` to walk through the analysis step by step.

## Tools & Libraries

Python · Pandas · NumPy · Scikit-learn · XGBoost · Matplotlib · Seaborn

## Next Steps (Production Roadmap)

- Incorporate marketing/acquisition cost per channel to compute LTV:CAC ratios
- Validate across multiple rolling time windows (not just one train/test split) to check stability
- Add SHAP values for per-customer explainability, useful for CRM-facing reason codes
- Retrain on a rolling monthly basis as new transaction data arrives

---
*Author: Samson Savio — Data Analyst | [LinkedIn](https://linkedin.com/in/samson-savio-263202165)*

# Chennai Restaurant Chain — Data Science Analytics Project
**Retail & Restaurant Sales Analytics Challenge**

---

## Project Overview

This project performs end-to-end data science analysis on a multi-branch restaurant chain operating across 5 Chennai locations (Anna Nagar, T Nagar, Velachery, Adyar, OMR) over 2 years (2022–2023). It covers all 7 assignment tasks: data cleaning, EDA, business insights, ML-based revenue prediction, anomaly detection, dashboard creation, and business recommendations.

---

## Repository Structure

```
├── restaurant_sales_raw.csv       # Synthetic dataset (3,650 rows, 5 branches, 2 years)
├── restaurant_sales_clean.csv     # Cleaned dataset with engineered features
├── analysis_notebook.py           # Main analysis script (all 7 tasks)
├── eda_dashboard.png              # EDA visualizations (Task 2 + 3)
├── insights_extra.png             # Additional insight charts
├── ml_model.png                   # Model evaluation plots (Task 4)
├── anomaly_detection.png          # Anomaly plots (Task 5)
├── executive_dashboard.png        # Dark-theme executive dashboard (Task 6)
└── README.md                      # This file
```

---

## Dataset

| Column           | Description                          |
|------------------|--------------------------------------|
| date             | Transaction date                     |
| branch           | Branch name (5 Chennai locations)    |
| revenue          | Daily revenue (₹)                    |
| orders           | Total daily orders                   |
| profit           | Daily profit (₹)                     |
| customer_rating  | Rating 1–5                           |
| complaints       | Daily complaint count                |
| marketing_spend  | Daily marketing spend (₹)            |
| online_orders    | Swiggy/Zomato orders                 |
| dinein_orders    | Dine-in orders                       |

**Size:** 3,650 rows × 10 base columns + 9 engineered features = 19 columns total

---

## Task Summary

### Task 1 — Data Cleaning
- **30 duplicate rows** removed
- **277 missing values** handled (branch+month median for revenue; global median for rating)
- **360 outliers** capped using IQR method (not removed — preserves sample size)
- **Datatypes** corrected: date → datetime, branch → category

### Task 2 — EDA
- Revenue trends by branch over 24 months
- Weekend vs weekday comparison
- Seasonal patterns (monthly/quarterly)
- Order channel mix (online vs dine-in)
- Day-of-week heatmap by branch

### Task 3 — 8 Business Insights
1. **Weekend revenue is ~22% higher** than weekdays (Insight 1)
2. **T Nagar is the top revenue branch** (Insight 2)
3. **Oct–Jan festival months drive 15%+ revenue uplift** (Insight 3)
4. **Strong negative correlation** between complaints and ratings (Insight 4)
5. **Online orders grew YoY** post-COVID delivery shift (Insight 5)
6. **Marketing spend positively correlates with revenue** (Insight 6)
7. **Saturday is the peak day** across all branches (Insight 7)
8. **Profit margin variance** is highest at Velachery — cost control issue (Insight 8)

### Task 4 — Revenue Prediction Model
| Model              | MAE     | RMSE    | R²     |
|--------------------|---------|---------|--------|
| Linear Regression  | ₹2,613  | ₹3,287  | 0.888  |
| **Random Forest**  | **₹2,380** | **₹3,110** | **0.900** |

**Selected Model:** Random Forest — handles non-linear interactions, robust to outliers, R² ≈ 0.90

**Top Features:** orders, marketing_spend, month, branch, is_weekend

### Task 5 — Anomaly Detection
- **Z-Score (threshold 2.5σ):** 188 anomalies flagged
- **Isolation Forest (3% contamination):** 110 anomalies flagged
- **Combined (union):** 208 total anomaly records
- Likely causes: festival spikes, food safety incidents, kitchen breakdowns, viral reviews

### Task 6 — Dashboard
Dark-theme executive dashboard with:
- 4 KPI tiles (Revenue, Orders, Rating, Profit %)
- Revenue trend by branch
- Branch comparison bar
- Anomaly scatter overlay
- Prediction accuracy scatter
- Customer rating by branch
- Orders channel mix

### Task 7 — 5 Business Recommendations
1. **Weekend Marketing Surge** — Target Friday evening ads on Swiggy/Instagram
2. **Festival Campaign Calendar** — Pre-plan Oct–Jan promotions 6 weeks ahead
3. **Online Order Growth Strategy** — Loyalty points for repeat delivery customers
4. **Complaint Resolution SLA** — 2-hour resolution target, weekly dashboard tracking
5. **Profit Margin Audit** — Standardize procurement to reduce cost volatility

---

## Tools Used

| Tool/Library       | Purpose                              |
|--------------------|--------------------------------------|
| Python 3.11        | Core language                        |
| pandas             | Data manipulation & cleaning         |
| numpy              | Numerical operations                 |
| matplotlib         | Base plotting & dashboard layout     |
| seaborn            | Statistical visualizations           |
| scikit-learn       | ML models, anomaly detection, metrics|
| scipy.stats        | Z-score anomaly detection            |

---

## How to Run

```bash
# 1. Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn scipy

# 2. Generate the dataset
python generate_dataset.py

# 3. Run full analysis
python analysis_notebook.py
```

---

## Evaluation Checklist

| Area              | Marks | Status  |
|-------------------|-------|---------|
| Data cleaning     | 15    | ✅ Done  |
| EDA quality       | 20    | ✅ Done  |
| Insights          | 20    | ✅ Done  |
| ML model          | 15    | ✅ Done  |
| Dashboard         | 15    | ✅ Done  |
| Anomaly detection | 10    | ✅ Done  |
| Documentation     | 5     | ✅ Done  |
| **Total**         | **100** | ✅      |

---

*Project by: Data Science Internship Candidate | Dataset: Synthetic (Chennai restaurant chain simulation)*

# RetailIQ

### Intelligent Retail Decision Support System

RetailIQ is a machine-learning-powered retail intelligence and decision-support system designed to help businesses understand sales, inventory health, forward demand projections, and replenishment needs through an interactive dark-mode Streamlit dashboard.

## Live Application

https://retailiq-analytics.streamlit.app/

## Overview

RetailIQ combines predictive transaction regression, forward-looking chronological daily sales forecasting, multi-class inventory classification, and rule-based dynamic restocking recommendations into a single retail analytics platform.

The system allows users to upload daily sales CSVs with automatic schema validation and deduplication, analyze business performance across multiple time horizons, generate sales forecasts, classify catalog inventory states, and generate purchase orders.

## Key Features

- **Business Analytics Hub**
  - **Overview**: Executive Retail KPIs (Revenue, Volume, Active Catalog, Margin, Restock Alerts), Monthly Sales Trend visualization, and Category breakdowns.
  - **Daily Sales**: Real-time daily transactional tracking and date-filtered metrics.
  - **Weekly & Monthly**: Medium- and long-term sales velocity and revenue trends.
  - **Product Performance**: Top-selling products and brand-level sales rankings.
  - **Business Insights**: Category margins and operational transaction distributions.

- **Sales Forecasting Hub**
  - Interactive Forecast Horizon slider (3 to 30 days).
  - Projected period revenue turnover and unit volume calculations.
  - Daily projected sales visual trend line and itemized forecast export table.

- **ML Models & Evaluation Hub**
  - **Task 1: Transaction-Level Revenue Regression**: Benchmark of 4 algorithms (Linear Regression, Ridge, Lasso, Random Forest Regressor) with positive explanatory variance ($R^2 = 0.8899$ for linear models, $R^2 = 0.9979$ for Random Forest).
  - **Task 2: Chronological Store-Level Daily Sales Forecasting**: Benchmark of 4 machine learning regressors (Random Forest, Ridge, Linear Regression, Gradient Boosting) against Seasonal Naive baseline on held-out test periods with transparent WAPE ($6.36\% - 9.79\%$) and $R^2$ disclosure.
  - **Inventory Health Classification**: Multi-class benchmark across Baseline and Tuned configurations (KNN, Decision Tree, Linear SVC, Random Forest, Majority Baseline) with confusion matrix diagnostics.

- **Inventory & Restock Hub**
  - Real-time stock health distribution (*Adequate Stock*, *Low Stock / High Urgency*, *Overstocked*).
  - **Immediate Reorders**: Automated purchase order generation for items at or below reorder threshold.
  - **Watchlist (Restock Soon)**: Lead-time demand forecasting for preventative replenishment.
  - **1-Click Purchase Order Export**: Instant CSV download for vendor orders.

- **Resilient Data Ingestion & Persistence Engine**
  - Multi-file CSV and Excel upload with schema normalization.
  - Deterministic SHA-256 fingerprinting rejecting accidental duplicate uploads while preserving multi-item invoices.
  - Isolated persistence under `data/uploads/` leaving baseline historical data untouched.
  - Instant cache invalidation and date widget state reset without application restarts.

## Machine Learning Benchmarks

### Task 1: Transaction-Level Revenue Regression
Evaluated on 20,000 held-out test transactions:

| Model | MAE (₹) | RMSE (₹) | WAPE | R² |
|---|---:|---:|---:|---:|
| **Random Forest Regressor** | **10.22** | **13.64** | **0.0258** | **0.9979** |
| Lasso Regression | 72.22 | 99.09 | 0.1826 | 0.8899 |
| Ridge Regression | 72.23 | 99.09 | 0.1826 | 0.8899 |
| Linear Regression | 72.23 | 99.09 | 0.1826 | 0.8899 |

### Task 2: Chronological Daily Store Sales Forecasting
Evaluated across held-out chronological test period (final 15% of historical days):

| Model | MAE (₹) | RMSE (₹) | WAPE | R² |
|---|---:|---:|---:|---:|
| **Random Forest Regressor** | **6,770.79** | **8,523.28** | **0.0636** | **-0.1290** |
| Ridge Regression | 6,971.39 | 8,736.05 | 0.0655 | -0.1860 |
| Linear Regression | 7,091.69 | 8,856.05 | 0.0666 | -0.2189 |
| Gradient Boosting Regressor | 7,515.52 | 9,235.41 | 0.0706 | -0.3255 |
| Seasonal Naive (7-Day Lag) | 10,424.71 | 12,552.19 | 0.0979 | -1.4486 |

*Note on Negative R²:* On daily retail transaction aggregations where day-to-day autocorrelation is low ($r_{lag=1} \approx 0.026$), out-of-sample prediction error exceeds test period variance ($MSE > Var(y_{test})$), resulting in a negative $R^2$. This is a mathematically valid reflection of stochastic daily retail noise, while WAPE ($6.36\% - 9.79\%$) confirms low percentage error on actual revenue scales.

### Inventory Classification Benchmark
Evaluated across 20,000 held-out test records:

| Model | Stage | Accuracy | Precision | Recall | F1 Score |
|---|---|---:|---:|---:|---:|
| KNN | Baseline | 0.4533 | 0.3782 | 0.4533 | 0.3802 |
| Decision Tree | Baseline | 0.3752 | 0.3782 | 0.3752 | 0.3766 |
| SVM | Baseline | 0.5009 | 0.2509 | 0.5009 | 0.3343 |
| Random Forest | Baseline | 0.4696 | 0.3766 | 0.4696 | 0.3700 |
| KNN | Tuned | 0.4350 | 0.3362 | 0.3345 | 0.3144 |
| Decision Tree | Tuned | 0.4992 | 0.3217 | 0.3333 | 0.2264 |
| SVM | Tuned | 0.5009 | 0.1670 | 0.3333 | 0.2225 |
| Random Forest | Tuned | 0.4652 | 0.3418 | 0.3367 | 0.2929 |

## Technology Stack

- **Application & UI:** Streamlit (Dark Theme, Vectorized Caching)
- **Data Engineering:** Pandas, NumPy
- **Machine Learning:** Scikit-learn, Joblib
- **Visualization:** Matplotlib, Seaborn
- **Testing:** Pytest

## Project Structure

```text
RetailIQ/
│
├── app/
│   └── streamlit_app.py
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── uploads/
│
├── models/
│   ├── classification/
│   │   ├── baseline/
│   │   ├── tuned/
│   │   └── upgraded/
│   ├── forecasting/
│   └── regression/
│
├── notebooks/
│   ├── 01_EDA_Preprocessing.ipynb
│   ├── 02_Linear_Regression.ipynb
│   ├── 04_Classification.ipynb
│   └── 05_Model_Tuning_Evaluation.ipynb
│
├── outputs/
│   └── results/
│
├── src/
│   ├── data_manager.py
│   ├── ml_pipeline.py
│   ├── model_loader.py
│   ├── model_predictions.py
│   ├── preprocessing.py
│   └── restocking.py
│
├── tests/
│   └── test_retailiq.py
│
├── docs/
│   └── RetailIQ_Comprehensive_Project_Report.md
│
├── requirements.txt
├── README.md
└── run_app.py
```

## Quickstart

```bash
# Install dependencies
pip install -r requirements.txt

# Run automated tests
pytest tests/ -v

# Launch the Streamlit application
streamlit run app/streamlit_app.py
```

# RetailIQ

### Intelligent Retail Decision Support System

RetailIQ is a machine-learning-powered retail intelligence and decision-support system designed to help businesses understand sales, inventory, product relationships, customer patterns, and restocking needs through an interactive Streamlit dashboard.

## Live Application

https://retailiq-analytics.streamlit.app/

## Overview

RetailIQ combines data analytics, machine learning, association-rule mining, clustering, inventory classification, and rule-based restocking recommendations into a single retail analytics platform.

The system allows users to upload retail datasets, analyze business performance, generate predictions, identify product groups, discover product combinations, classify inventory-related patterns, and receive actionable restocking recommendations.

## Key Features

- **Home Dashboard**
  - Retail KPIs
  - Sales and revenue overview
  - Inventory insights
  - Business performance summary

- **Needs Attention**
  - Highlights products and inventory situations requiring attention
  - Identifies restocking priorities

- **Inventory Analysis**
  - Stock-on-hand analysis
  - Reorder-level monitoring
  - Inventory classification using machine-learning models

- **Restock Planner**
  - Calculates required stock using average demand and lead time
  - Provides Restock Now, Restock Soon, and Stock Sufficient recommendations
  - Suggests order quantities

- **Sales & Forecast**
  - Revenue prediction using Linear Regression
  - Sales-related trends and analytics

- **Product Combos**
  - Association-rule mining using Apriori
  - Product recommendation based on support, confidence, and lift

- **Product Groups**
  - Product/customer behavior grouping using K-Means clustering

- **Update Sales Data**
  - Upload CSV/Excel datasets
  - Automatic column normalization
  - Dataset validation and preprocessing

## Machine Learning

RetailIQ uses multiple machine-learning approaches.

### Regression

**Linear Regression**

Used to predict expected revenue/sales value from retail and customer-related features.

Performance:

| Metric | Result |
|---|---:|
| R² | 0.8899 |
| MAE | 72.23 |
| RMSE | 99.09 |

### Inventory Classification

Four classification models were evaluated:

- K-Nearest Neighbors
- Decision Tree
- Support Vector Machine
- Random Forest

The models are evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score

### Clustering

**K-Means Clustering**

Used to identify groups of products based on retail and inventory characteristics such as:

- Units
- Revenue
- Selling Price
- Margin
- Stock on Hand
- Reorder Level
- Lead Time

### Association Rule Mining

**Apriori**

Used to discover relationships between products in transaction-level data.

The association rules are evaluated using:

- Support
- Confidence
- Lift

### Restocking Recommendation

The restocking module uses a decision-support approach based on:

- Average demand
- Lead time
- Current stock
- Reorder level

The system categorizes inventory into:

- Restock Now
- Restock Soon
- Stock Sufficient

## Classification Model Comparison

| Model | Accuracy | Precision | Recall | F1 Score |
|---|---:|---:|---:|---:|
| KNN | 0.4533 | 0.3782 | 0.4533 | 0.3802 |
| Decision Tree | 0.3752 | 0.3782 | 0.3752 | 0.3766 |
| SVM | 0.5009 | 0.2509 | 0.5009 | 0.3343 |
| Random Forest | 0.4696 | 0.3766 | 0.4696 | 0.3700 |

Model performance is reported from the project's actual evaluation results rather than assumed benchmark values.

## Dataset

The primary retail dataset contains information related to:

- Invoice information
- Store and city
- Product category
- Brand
- Sales units
- Cost price
- Selling price
- Revenue
- Inventory
- Reorder level
- Lead time
- Customer age
- Customer gender
- Loyalty status

The preprocessing pipeline supports flexible column naming and normalizes common variations such as:

`Quantity`, `Qty`, and `Units`

or

`Price`, `Selling Price`, and `Selling_Price`.

Revenue and margin can also be derived when sufficient source columns are available.

## Technology Stack

### Frontend / Application

- Streamlit
- Python

### Data Processing

- Pandas
- NumPy

### Machine Learning

- Scikit-learn
- Joblib

### Association Rule Mining

- MLxtend

### Visualization

- Matplotlib
- Seaborn

### File Processing

- OpenPyXL

## Project Structure

```text
RetailIQ/
│
├── app/
│   └── streamlit_app.py
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│   ├── association/
│   ├── classification/
│   │   ├── baseline/
│   │   └── tuned/
│   ├── clustering/
│   └── regression/
│
├── outputs/
│   ├── figures/
│   └── results/
│
├── src/
│   ├── association.py
│   ├── clustering.py
│   ├── model_loader.py
│   ├── model_predictions.py
│   ├── preprocessing.py
│   └── restocking.py
│
├── notebooks/
│
├── requirements.txt
├── .gitignore
└── README.md

# RetailIQ — Intelligent Retail Decision Support System

RetailIQ is a machine-learning-powered retail analytics dashboard built with **Streamlit**. It brings sales reporting, demand forecasting, inventory classification, and replenishment support into one interactive application.

**Live dashboard:** https://retailiq-analytics.streamlit.app/

## Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Technology Stack](#technology-stack)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Data and Upload Handling](#data-and-upload-handling)
- [Machine Learning and Evaluation](#machine-learning-and-evaluation)
- [Tests](#tests)
- [Project Documentation](#project-documentation)
- [Limitations and Notes](#limitations-and-notes)

## Project Overview

RetailIQ is designed to help users explore retail transaction data and make more informed operational decisions. The dashboard reads the project's prepared historical dataset, derives reporting measures, uses saved machine-learning artifacts for supported prediction tasks, and provides inventory replenishment recommendations.

The application is organized around three practical areas:

1. **Understand performance** through sales summaries and time-based reporting.
2. **Explore predictions** through sales forecasting and model evaluation views.
3. **Support operations** through inventory/restocking views and a controlled sales-data upload workflow.

## Features

### Business analytics
- **Overview:** headline measures such as revenue, units, transactions, product count, and restock alerts.
- **Daily Sales:** date-focused sales reporting.
- **Weekly & Monthly:** longer-period sales trends and summaries.
- **Product Performance:** product-level sales comparisons.
- **Business Insights:** evidence-based summaries derived from the transaction data.

### Forecasting and machine learning
- **Sales Forecasting:** generates a forward-looking daily sales forecast using the project's saved forecasting pipeline.
- **ML Models & Evaluation:** presents model-comparison and evaluation results for the project's supported regression, forecasting, and classification tasks.
- **Inventory & Restock:** displays inventory status and replenishment recommendations.

### Data management and reports
- **Update Sales Data:** accepts supported CSV and Excel files, validates their contents, checks for duplicate file uploads, and provides a confirmation step before saving accepted data.
- Accepted uploads are recorded in `data/uploads/` with a manifest. The historical source dataset is kept separate from uploaded files.
- **Downloadable Reports:** provides report/export functionality available in the application.

## Technology Stack

- **Application:** Python, Streamlit
- **Data processing:** pandas, NumPy
- **Machine learning:** scikit-learn, joblib
- **Charts and visualizations:** Matplotlib, Seaborn
- **Excel input support:** openpyxl
- **Testing:** pytest

Exact dependency versions are listed in [`requirements.txt`](requirements.txt).

## Project Structure

```text
RetailIQ/
├── app/
│   └── streamlit_app.py          # Streamlit dashboard entry point
├── data/
│   ├── raw/                      # Source datasets
│   ├── processed/                # Prepared datasets and demand summaries
│   └── uploads/                  # Upload manifest and accepted uploads
├── docs/
│   ├── RetailIQ_10_Pages_Report.html
│   ├── RetailIQ_10_Pages_Report.pdf
│   └── RetailIQ_Comprehensive_Project_Report.md
├── models/
│   ├── classification/
│   │   ├── baseline/
│   │   ├── tuned/
│   │   └── upgraded/
│   ├── forecasting/
│   └── regression/
├── notebooks/                    # Data analysis and model experiments
├── outputs/
│   ├── figures/                  # Generated figures
│   └── results/                  # Model comparisons and predictions
├── src/
│   ├── data_manager.py           # Historical data and upload management
│   ├── ml_pipeline.py            # ML pipeline utilities
│   ├── model_loader.py           # Saved model loading
│   ├── model_predictions.py      # Prediction and forecasting functions
│   ├── preprocessing.py          # Data cleaning and feature preparation
│   └── restocking.py             # Replenishment logic
├── tests/
│   └── test_retailiq.py
├── .devcontainer/
├── requirements.txt
└── README.md
```

## Getting Started

### 1. Prerequisites

- Python compatible with the dependency versions in `requirements.txt`
- Git, if cloning the repository
- Git LFS if you are cloning the repository and need the large tracked model file

### 2. Clone the repository

```bash
git clone https://github.com/SimranChawla03/RetailIQ.git
cd RetailIQ
```

If Git LFS is installed, pull any LFS-managed assets:

```bash
git lfs install
git lfs pull
```

### 3. Create and activate a virtual environment

**Windows (Command Prompt):**

```bat
python -m venv .venv
.venv\Scripts\activate
```

**macOS / Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 4. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 5. Run the dashboard

From the repository root, run:

```bash
streamlit run app/streamlit_app.py
```

Streamlit will print a local URL (usually `http://localhost:8501`) in the terminal. Open that URL in your browser.

## Data and Upload Handling

The repository contains raw and processed sample data under `data/`. The dashboard's reporting pipeline loads the approved historical dataset from `data/processed/cleaned_data.csv`.

The **Update Sales Data** workflow is designed to:
1. Read a supported CSV or Excel file.
2. Normalize and validate columns and data.
3. Reject a file if its content hash matches an already accepted upload.
4. Ask for confirmation before committing an accepted upload.
5. Store the accepted data and update the upload manifest under `data/uploads/`.

Use files that contain the retail fields expected by the application's preprocessing and validation logic. Uploads with missing required fields, unreadable contents, or negative units/selling prices are rejected by the validation workflow.

> **Deployment note:** the upload code writes files to the application's local `data/uploads/` directory. On hosted platforms, local filesystem changes may not be durable across app rebuilds or instance replacement unless persistent storage is configured.

## Machine Learning and Evaluation

The project includes saved artifacts and evaluation outputs for these areas:

- **Transaction-level revenue regression**
- **Chronological daily sales forecasting**
- **Inventory health classification**
- **Inventory replenishment decision support**

Model artifacts are stored in `models/`, while evaluation tables and prediction outputs are stored in `outputs/results/`. Notebooks documenting parts of the analysis are in `notebooks/`.

Read the [comprehensive project report](docs/RetailIQ_Comprehensive_Project_Report.md) for the project's methodology, experimental results, discussion, and future-work notes.

**Evaluation note:** use the reported metrics together with the evaluation setup. In particular, a negative out-of-sample \(R^2\) score can occur when a forecasting model performs worse than predicting the test-period mean; it should be reported transparently rather than hidden.

## Tests

Run the test suite from the repository root:

```bash
pytest tests/ -v
```

If pytest is not installed in your environment, install it with:

```bash
pip install pytest
```

## Project Documentation

- [Comprehensive project report (Markdown)](docs/RetailIQ_Comprehensive_Project_Report.md)
- [10-page project report (PDF)](docs/RetailIQ_10_Pages_Report.pdf)
- [10-page project report (HTML)](docs/RetailIQ_10_Pages_Report.html)
- [Model comparison results](outputs/results/)

## Limitations and Notes

- Forecasts and classifications are model outputs based on the available data and saved artifacts; they are decision-support aids, not guarantees of future outcomes.
- Results depend on the quality, coverage, and schema of the input data.
- For a reproducible local run, use the repository's `requirements.txt` and ensure Git LFS-managed model files are downloaded.
- The dashboard entry point is `app/streamlit_app.py`; there is no separate `run_app.py` entry point in this project.

## License

No license file was present in the supplied project archive. Add a `LICENSE` file if you intend to publish the project under a specific open-source license.

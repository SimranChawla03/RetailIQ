"""
Unit and integration test suite covering mandatory acceptance criteria:
- Test A: Fresh launch historical data loading without upload
- Test B: Resilience when model artifacts are missing
- Test C: Valid daily CSV upload and schema validation
- Test D: Duplicate file rejection and deduplication
- Test E: Persistent reload verification
- Test F: Model metric consistency with saved evaluations
"""
import os
import json
import pandas as pd
from src.data_manager import (
    load_historical_data,
    load_reporting_data,
    validate_upload_content,
    persist_accepted_upload,
    get_upload_manifest,
    compute_file_hash
)
from src.model_loader import load_models
from src.preprocessing import preprocess_data, validate_data
from src.model_predictions import forecast_daily_sales

def test_a_fresh_launch_loads_historical_data():
    """Test A: Dashboard reporting pipeline loads historical data immediately on startup."""
    df = load_reporting_data()
    assert df is not None, "Reporting dataframe must not be None"
    assert not df.empty, "Reporting dataframe must not be empty on fresh launch"
    assert len(df) >= 100000, f"Expected at least 100,000 historical records, found {len(df)}"
    assert "Revenue" in df.columns, "Revenue column must exist"
    assert "Units" in df.columns, "Units column must exist"
    assert df["Revenue"].sum() > 0, "Revenue sum must be greater than zero"

def test_b_missing_model_resilience(tmp_path):
    """Test B: If model artifacts are missing, load_models handles it cleanly without crashing."""
    models = load_models(base_dir=str(tmp_path))
    assert isinstance(models, dict), "load_models should return a dict even when models are absent"
    assert "forecasting_regressor" not in models, "Missing model should not be in dict"

def test_c_valid_upload_validation(tmp_path):
    """Test C: Valid daily CSV upload is accepted and validated."""
    sample_df = pd.DataFrame([
        {
            "Invoice_ID": 9999001,
            "Invoice_Date": "2024-12-31 10:00:00",
            "City": "Mumbai",
            "Store_Format": "Supermarket",
            "Category": "Beverages",
            "Brand": "Amul",
            "Units": 10,
            "Cost_Price": 15.0,
            "Selling_Price": 25.0,
            "Stock_On_Hand": 120,
            "Reorder_Level": 30,
            "Lead_Time_Days": 5
        }
    ])
    csv_bytes = sample_df.to_csv(index=False).encode("utf-8")
    val_res = validate_upload_content(csv_bytes, "new_daily_sales.csv")

    assert val_res["valid"] is True, f"Upload validation failed: {val_res.get('error')}"
    assert val_res["row_count"] == 1
    assert "Revenue" in val_res["processed_df"].columns

    # Test negative units rejection
    bad_units_df = sample_df.copy()
    bad_units_df["Units"] = -10
    val_bad_units = validate_upload_content(bad_units_df.to_csv(index=False).encode("utf-8"), "bad_units.csv")
    assert val_bad_units["valid"] is False
    assert "negative Units" in val_bad_units["error"]

    # Test missing required retail column
    bad_cols_df = pd.DataFrame([{"Invoice_ID": 101, "City": "Delhi"}])
    val_bad_cols = validate_upload_content(bad_cols_df.to_csv(index=False).encode("utf-8"), "missing_cols.csv")
    assert val_bad_cols["valid"] is False
    assert "Required retail fields missing" in val_bad_cols["error"]

    # Test empty file rejection
    val_empty = validate_upload_content(b"", "empty.csv")
    assert val_empty["valid"] is False
    assert "empty" in val_empty["error"].lower()

def test_d_duplicate_upload_detection():
    """Test D: Uploading an identical file is detected and rejected by content hash."""
    ts = pd.Timestamp.now().strftime("%Y%m%d%H%M%S")
    sample_df = pd.DataFrame([
        {
            "Invoice_ID": int(ts[-8:]),
            "Invoice_Date": "2025-02-01 12:00:00",
            "Brand": "Parle",
            "Category": "Snacks",
            "Units": 5,
            "Cost_Price": 10.0,
            "Selling_Price": 20.0
        }
    ])
    csv_bytes = sample_df.to_csv(index=False).encode("utf-8")
    test_fname = f"unique_test_{ts}.csv"
    val_res = validate_upload_content(csv_bytes, test_fname)
    assert val_res["valid"] is True

    # Persist the file
    stored_name = persist_accepted_upload(csv_bytes, test_fname, val_res["file_hash"], val_res["processed_df"])

    try:
        # Attempting to upload the exact same content again should be rejected
        val_res_dup = validate_upload_content(csv_bytes, test_fname)
        assert val_res_dup["valid"] is False
        assert "Duplicate file detected" in val_res_dup["error"]
    finally:
        # Clean up test artifact to avoid polluting uploads directory
        from src.data_manager import delete_accepted_upload
        delete_accepted_upload(stored_name)

def test_d2_dashboard_data_refresh_and_multi_item_invoices():
    """Test D2: Multi-item invoices are preserved, and reporting data includes newly persisted records."""
    rep_df_before = load_reporting_data()
    initial_count = len(rep_df_before)
    ts = pd.Timestamp.now().strftime("%Y%m%d%H%M%S")
    invoice_id = int("77" + ts[-6:])

    # Multi-item invoice (same Invoice_ID, different brands)
    multi_item_df = pd.DataFrame([
        {
            "Invoice_ID": invoice_id,
            "Invoice_Date": "2025-02-05 14:00:00",
            "Brand": "Amul",
            "Category": "Dairy",
            "Units": 3,
            "Cost_Price": 15.0,
            "Selling_Price": 30.0
        },
        {
            "Invoice_ID": invoice_id,
            "Invoice_Date": "2025-02-05 14:00:00",
            "Brand": "Britannia",
            "Category": "Snacks",
            "Units": 2,
            "Cost_Price": 10.0,
            "Selling_Price": 20.0
        }
    ])
    csv_bytes = multi_item_df.to_csv(index=False).encode("utf-8")
    test_fname = f"multi_item_{ts}.csv"
    val_res = validate_upload_content(csv_bytes, test_fname)
    assert val_res["valid"] is True
    assert val_res["row_count"] == 2

    # Persist and reload
    stored_name = persist_accepted_upload(csv_bytes, test_fname, val_res["file_hash"], val_res["processed_df"])
    try:
        rep_df_after = load_reporting_data()

        # Both items of the invoice must be preserved
        invoice_items = rep_df_after[rep_df_after["Invoice_ID"] == invoice_id]
        assert len(invoice_items) == 2, "Multi-item invoice lines must not be collapsed"
        assert set(invoice_items["Brand"].tolist()) == {"Amul", "Britannia"}
        assert len(rep_df_after) == initial_count + 2
    finally:
        from src.data_manager import delete_accepted_upload
        delete_accepted_upload(stored_name)

def test_e_model_evaluation_metrics_consistency():
    """Test E: Saved evaluation results match expected metric structures for regression and dual-view classification."""
    # Task 1: Transaction Revenue Regression
    trans_csv_path = "outputs/results/transaction_regression_comparison.csv"
    assert os.path.exists(trans_csv_path), "Transaction regression comparison CSV must exist"
    trans_df = pd.read_csv(trans_csv_path)
    for col in ["Model", "MAE", "RMSE", "WAPE", "R2"]:
        assert col in trans_df.columns, f"Missing transaction metric column {col}"
    assert (trans_df["R2"] > 0).all(), "Transaction-level revenue regression R2 should be positive"
    assert len(trans_df) >= 4, "Should evaluate 4 transaction regression models"

    # Task 2: Chronological Daily Forecasting
    reg_csv_path = "outputs/results/regression_model_comparison.csv"
    assert os.path.exists(reg_csv_path), "Regression forecasting comparison CSV must exist"
    reg_df = pd.read_csv(reg_csv_path)
    for col in ["Model", "MAE", "RMSE", "WAPE", "R2"]:
        assert col in reg_df.columns, f"Missing forecasting metric column {col}"
    assert (reg_df["WAPE"] >= 0).all()
    assert len(reg_df) >= 5, "Should evaluate baseline and 4 forecasting models"

    # Section A: Demand Trend
    trend_csv = "outputs/results/demand_trend_model_comparison.csv"
    assert os.path.exists(trend_csv), "Demand trend comparison CSV must exist"
    trend_df = pd.read_csv(trend_csv)
    for col in ["Model", "Accuracy", "Balanced Accuracy", "Precision (Macro)", "Recall (Macro)", "F1 (Macro)", "F1 (Weighted)"]:
        assert col in trend_df.columns, f"Missing trend metric column {col}"
    assert len(trend_df) >= 5, "Should include baseline and 4 candidates"

    # Section B: Inventory Decision Support
    inv_csv = "outputs/results/inventory_replenishment_comparison.csv"
    assert os.path.exists(inv_csv), "Inventory replenishment comparison CSV must exist"
    inv_df = pd.read_csv(inv_csv)
    for col in ["Model", "Accuracy", "Balanced Accuracy", "Precision (At-Risk)", "Recall (At-Risk)", "F1 (At-Risk)", "F1 (Macro)", "F1 (Weighted)"]:
        assert col in inv_df.columns, f"Missing inventory metric column {col}"
    assert len(inv_df) >= 5, "Should include baseline and 4 candidates"

def test_f_forecasting_pipeline_inference():
    """Test F: Forecasting function runs rolling forward projections."""
    models = load_models()
    f_model = models.get("forecasting_regressor")
    assert f_model is not None, "forecasting_regressor should be loaded"

    daily_hist = pd.DataFrame({
        "Date": pd.date_range("2024-01-01", periods=30),
        "Revenue": [10000.0 + i * 50 for i in range(30)],
        "Units": [100 + i for i in range(30)]
    })
    forecast_df = forecast_daily_sales(daily_hist, f_model, horizon_days=7)
    assert not forecast_df.empty, "Forecast dataframe should not be empty"
    assert len(forecast_df) == 7, "Should produce exactly 7 days forecast"
    assert (forecast_df["Forecasted_Revenue"] >= 0).all(), "Forecasted revenue must be non-negative"

def test_g_all_models_load_cleanly():
    """Test G: All core regression and classification models load without error, including Random Forest."""
    models = load_models()
    expected_models = [
        "regression",
        "knn",
        "decision_tree",
        "svm",
        "random_forest",
        "classification_scaler",
        "forecasting_regressor",
        "forecasting_scaler",
        "demand_trend_classifier",
        "demand_trend_scaler",
        "inventory_decision_classifier",
        "inventory_decision_scaler"
    ]
    for m in expected_models:
        assert m in models, f"Model artifact '{m}' failed to load or is missing"

def test_h_model_inference_with_incomplete_upload_data():
    """Test H: Inference pipeline and restocking handle uploaded sales records missing inventory fields."""
    from src.model_predictions import predict_revenue, predict_inventory
    from src.restocking import calculate_restocking

    models = load_models()
    # Simulate a transaction dataset with only basic sales columns
    incomplete_df = pd.DataFrame([
        {
            "Invoice_ID": 88001,
            "Invoice_Date": "2025-01-10 10:00:00",
            "Brand": "Amul",
            "Category": "Dairy",
            "Units": 5,
            "Selling_Price": 50.0,
            "Revenue": 250.0,
            "Month": 1,
            "Day_of_Week": 4
        }
    ])

    # 1. Revenue prediction should not crash on missing Customer_Age / Loyalty_Flag / Cost_Price
    rev_res = predict_revenue(incomplete_df, models["regression"])
    assert "Predicted_Revenue" in rev_res.columns
    assert not rev_res["Predicted_Revenue"].isna().any()

    # 2. Inventory classification should not crash
    inv_res = predict_inventory(incomplete_df, models)
    assert any(col in inv_res.columns for col in ["Random_Forest_Prediction", "Decision_Tree_Prediction", "KNN_Prediction"])

    # 3. Restocking calculation should not raise IntCastingNaNError
    restock_res = calculate_restocking(incomplete_df)
    assert "Restock_Status" in restock_res.columns
    assert "Suggested_Quantity" in restock_res.columns

def test_i_upload_delete_workflow():
    """Test I: Full upload and deletion cycle persists, updates reporting data, and deletes cleanly."""
    from src.data_manager import delete_accepted_upload

    before_df = load_reporting_data()
    before_len = len(before_df)

    sample = pd.DataFrame([{
        "Invoice_ID": 9999888,
        "Invoice_Date": "2025-03-01",
        "Brand": "Tata",
        "Category": "Beverages",
        "Units": 10,
        "Selling_Price": 100.0,
        "Revenue": 1000.0
    }])
    csv_bytes = sample.to_csv(index=False).encode("utf-8")
    val_res = validate_upload_content(csv_bytes, "workflow_test.csv")
    assert val_res["valid"] is True

    stored_name = persist_accepted_upload(csv_bytes, "workflow_test.csv", val_res["file_hash"], val_res["processed_df"])

    after_upload = load_reporting_data()
    assert len(after_upload) == before_len + 1

    # Delete the upload
    deleted_info = delete_accepted_upload(stored_name)
    assert deleted_info["stored_filename"] == stored_name

    after_delete = load_reporting_data()
    assert len(after_delete) == before_len


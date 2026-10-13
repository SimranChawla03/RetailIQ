"""
RetailIQ — Intelligent Retail Decision Support System.
Upgraded Master Application.
Preserves dark aesthetic, loads historical data on startup, provides comprehensive
business analytics, daily forecasting, ML model evaluation, downloadable reports,
and persistent daily CSV uploads.
"""
import os
import sys
import io
import json
import streamlit as st
import pandas as pd
import numpy as np

# ------------------------------------------------------------
# PROJECT SETUP
# ------------------------------------------------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = CURRENT_DIR if os.path.exists(os.path.join(CURRENT_DIR, "src")) else os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.preprocessing import validate_data, preprocess_data, get_module_support
from src.model_loader import load_models
from src.model_predictions import predict_revenue, predict_inventory, forecast_daily_sales
from src.restocking import calculate_restocking
from src.data_manager import (
    load_reporting_data,
    validate_upload_content,
    persist_accepted_upload,
    get_upload_manifest,
    delete_accepted_upload
)

st.set_page_config(
    page_title="RetailIQ",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)


# PROFESSIONAL UI 

st.markdown(
    """
    <style>
    .stApp, .main, [data-testid="stAppViewContainer"] {
        background: #050505 !important;
        color: #f2f2f2 !important;
    }
    [data-testid="stHeader"] { background: #050505 !important; }
    .block-container { max-width: 1450px; padding-top: 1.4rem; padding-bottom: 3rem; }

    section[data-testid="stSidebar"] {
        background: #080808 !important;
        border-right: 1px solid #1d1d1d;
    }
    section[data-testid="stSidebar"] > div { padding: 1.25rem .9rem 1.5rem; }

    .brand { font-size: 1.7rem; font-weight: 800; color: #ffffff; margin-bottom: 0.15rem; }
    .brand span { color: #ffffff; }
    .brand-sub { color: #777777; font-size: 0.78rem; margin-bottom: 1.6rem; }
    .menu-label { color: #606060; font-size: 0.65rem; font-weight: 700; letter-spacing: .15em; margin: 1.25rem 0 .65rem; }

    section[data-testid="stSidebar"] .stButton > button {
        width: 100%; text-align: left; background: #080808; border: 1px solid transparent;
        color: #a7a7a7; border-radius: 7px; padding: .62rem .75rem; margin-bottom: .12rem;
    }
    section[data-testid="stSidebar"] .stButton > button:hover {
        background: #141414; border-color: #292929; color: #ffffff;
    }

    .app-header {
        display: flex; align-items: center; gap: 18px;
        width: 100%; box-sizing: border-box;
        background: #080808; border: 1px solid #202020;
        border-radius: 10px; padding: 16px 20px;
        margin-bottom: 24px;
    }
    .header-brand {
        color: #ffffff; font-size: 1.65rem; font-weight: 800;
        letter-spacing: -.025em; line-height: 1; white-space: nowrap;
    }
    .header-tagline {
        color: #777777; font-size: .86rem; line-height: 1.2;
        flex: 1; min-width: 0;
    }
    .header-status {
        color: #a9a9a9; font-size: .68rem; font-weight: 700;
        letter-spacing: .09em; text-transform: uppercase;
        border: 1px solid #292929; border-radius: 999px;
        padding: 6px 10px; white-space: nowrap;
    }
    .status { color: #bdbdbd; font-weight: 600; }
    @media (max-width: 700px) {
        .app-header { flex-wrap: wrap; gap: 8px 14px; padding: 14px 16px; }
        .header-brand { font-size: 1.45rem; }
        .header-tagline { flex-basis: 100%; order: 3; font-size: .78rem; }
        .header-status { margin-left: auto; }
    }
    .page-title { font-size: 1.55rem; font-weight: 750; color: #ffffff; margin-top: .6rem; margin-bottom: .25rem; }
    .page-subtitle { color: #777777; font-size: .86rem; margin-bottom: 1.2rem; }

    .panel, .action-box, .success-box, .warning-box, .danger-box, .welcome-box, .info-card {
        background: #0b0b0b !important; border: 1px solid #222222 !important;
        border-radius: 10px; padding: 1rem 1.15rem; margin-bottom: 1rem;
    }
    .welcome-box { padding: 1.5rem; }
    .welcome-title { font-size: 1.45rem; font-weight: 750; color: #ffffff; margin-bottom: .35rem; }
    .welcome-text, .action-text { color: #858585; font-size: .9rem; }
    .action-title { color: #ffffff; font-weight: 700; font-size: 1rem; }
    .success-box { border-color: #24452f !important; color: #9fd7ad; }
    .warning-box { border-color: #4a4124 !important; color: #d7c48d; }
    .danger-box { border-color: #4a2828 !important; color: #e0a0a0; }

    div[data-testid="stMetric"] {
        background: #0a0a0a !important; border: 1px solid #222222 !important;
        border-radius: 10px; padding: 1.05rem 1.15rem; min-height: 115px;
    }
    div[data-testid="stMetric"] label { color: #777777 !important; font-size: .66rem !important; font-weight: 700 !important; letter-spacing: .1em; }
    div[data-testid="stMetricValue"] { color: #ffffff !important; font-size: 1.8rem !important; font-weight: 750 !important; }

    [data-testid="stDataFrame"] { border: 1px solid #222222; border-radius: 9px; overflow: hidden; }
    [data-testid="stFileUploader"] { background: #090909; border: 1px solid #222222; border-radius: 9px; padding: .4rem; }
    div[data-baseweb="select"] > div { background: #090909 !important; border-color: #292929 !important; }
    .stButton > button { border-radius: 7px; border: 1px solid #303030; background: #111111; color: #ffffff; }
    .stButton > button:hover { border-color: #555555; background: #171717; }
    hr { border-color: #1e1e1e; }
    </style>
    """,
    unsafe_allow_html=True,
)

# MODEL CACHING

@st.cache_resource(show_spinner="Loading machine learning models...")
def get_cached_models():
    return load_models()

try:
    models = get_cached_models()
    model_load_error = None
except Exception as exc:
    models = {}
    model_load_error = str(exc)


# HELPER FUNCTIONS
def money(value):
    try:
        return f"₹{float(value):,.0f}"
    except Exception:
        return "₹0"

def unique_columns(columns):
    seen = set()
    result = []
    for c in columns:
        if c and c not in seen:
            seen.add(c)
            result.append(c)
    return result

def get_product_column(df):
    for col in ["Product", "Product_Name", "Item", "Brand"]:
        if col in df.columns:
            return col
    return None

def page_header(title, subtitle):
    st.markdown(f'<div class="page-title">{title}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="page-subtitle">{subtitle}</div>', unsafe_allow_html=True)

def run_models(df):
    processed = df.copy()
    try:
        support = get_module_support(processed)
    except Exception:
        support = {}

    if not models:
        return processed, support

    if support.get("Sales / Demand Prediction", {}).get("available", False):
        try:
            if "regression" in models:
                processed = predict_revenue(processed, models["regression"])
        except Exception:
            pass

    if support.get("Inventory Classification", {}).get("available", False):
        try:
            processed = predict_inventory(processed, models)
        except Exception:
            pass

    if support.get("Restocking Recommendation", {}).get("available", False):
        try:
            processed = calculate_restocking(processed)
            if "Suggested_Quantity" in processed.columns and "Suggested_Order_Quantity" not in processed.columns:
                processed["Suggested_Order_Quantity"] = processed["Suggested_Quantity"]
        except Exception:
            pass

    return processed, support


# CENTRAL DATA LOADING & FIRST-LAUNCH INITIALIZATION

@st.cache_data(show_spinner="Loading retail reporting dataset...")
def get_cached_reporting_data():
    """Load the latest persisted reporting data and run analytics on it."""
    raw_reporting = load_reporting_data()
    if raw_reporting is None or raw_reporting.empty:
        return raw_reporting, {}
    processed_df, support = run_models(raw_reporting)
    return processed_df, support

def init_state():
    if "page" not in st.session_state:
        st.session_state.page = "Overview"
    if "upload_preview" not in st.session_state:
        st.session_state.upload_preview = None
    # Increment after a successful upload to reset Streamlit's file uploader.
    if "upload_widget_version" not in st.session_state:
        st.session_state.upload_widget_version = 0

init_state()

try:
    display_df, module_support = get_cached_reporting_data()
    data_load_error = None
except Exception as exc:
    display_df, module_support = None, {}
    data_load_error = str(exc)


# SIDEBAR NAVIGATION
with st.sidebar:
    st.markdown(
        '<div class="brand">RetailIQ</div><div class="brand-sub">Intelligent retail decision support</div>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="menu-label">BUSINESS</div>', unsafe_allow_html=True)
    business_pages = [
        "Overview",
        "Daily Sales",
        "Weekly & Monthly",
        "Product Performance",
        "Business Insights",
    ]
    for p in business_pages:
        if st.button(p, use_container_width=True, key=f"nav_{p}"):
            st.session_state.page = p
            st.rerun()

    st.markdown('<div class="menu-label">INTELLIGENCE & ML</div>', unsafe_allow_html=True)
    ml_pages = [
        "Sales Forecasting",
        "ML Models & Evaluation",
        "Inventory & Restock",
    ]
    for p in ml_pages:
        if st.button(p, use_container_width=True, key=f"nav_{p}"):
            st.session_state.page = p
            st.rerun()

    st.markdown('<div class="menu-label">OPERATIONS</div>', unsafe_allow_html=True)
    ops_pages = [
        "Update Sales Data",
        "Downloadable Reports"
    ]
    for p in ops_pages:
        if st.button(p, use_container_width=True, key=f"nav_{p}"):
            st.session_state.page = p
            st.rerun()

    st.divider()
    if display_df is not None:
        st.markdown('<div style="color:#666;font-size:.68rem;font-weight:700;letter-spacing:.12em;">DATA STATUS</div>', unsafe_allow_html=True)
        st.markdown('<div style="color:#a7a7a7;font-weight:600;margin-top:4px;">Dashboard Connected</div>', unsafe_allow_html=True)
        st.caption(f"{len(display_df):,} records active")
    else:
        st.caption("Historical dataset offline")


# HEADER
st.markdown(
    '<div class="app-header"><div class="header-brand">RetailIQ</div><div class="header-tagline">Intelligent Retail Decision Support System</div><div class="header-status">Workspace Active</div></div>',
    unsafe_allow_html=True,
)

if data_load_error:
    st.error(f"Critical error: Failed to load reporting dataset. {data_load_error}")
    st.stop()


# PAGE 1: BUSINESS OVERVIEW
if st.session_state.page in ["Overview", "Home"]:
    page_header("Business Overview", "Comprehensive executive summary of store performance and live catalog health.")

    total_revenue = display_df["Revenue"].sum() if "Revenue" in display_df.columns else 0
    total_units = display_df["Units"].sum() if "Units" in display_df.columns else 0
    total_transactions = display_df["Invoice_ID"].nunique() if "Invoice_ID" in display_df.columns else len(display_df)
    product_col = get_product_column(display_df)
    total_products = display_df[product_col].nunique() if product_col else 0
    restock_now = int((display_df["Restock_Status"] == "Restock Now").sum()) if "Restock_Status" in display_df.columns else 0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("TOTAL REVENUE", money(total_revenue))
    c2.metric("UNITS SOLD", f"{total_units:,.0f}")
    c3.metric("TRANSACTIONS", f"{total_transactions:,}")
    c4.metric("RESTOCK ALERTS", f"{restock_now:,}")

    col_left, col_right = st.columns([2, 1])

    with col_left:
        st.markdown('<div class="page-title">Monthly Revenue Trend</div>', unsafe_allow_html=True)
        if "Invoice_Date" in display_df.columns and "Revenue" in display_df.columns:
            temp = display_df.copy()
            temp["Invoice_Date"] = pd.to_datetime(temp["Invoice_Date"], errors="coerce")
            temp = temp.dropna(subset=["Invoice_Date"])
            monthly = temp.assign(Month=temp["Invoice_Date"].dt.to_period("M").astype(str)).groupby("Month")["Revenue"].sum()
            st.line_chart(monthly, height=310)

    with col_right:
        st.markdown('<div class="page-title">Top Product Categories</div>', unsafe_allow_html=True)
        if "Category" in display_df.columns and "Revenue" in display_df.columns:
            cat_sales = display_df.groupby("Category")["Revenue"].sum().sort_values(ascending=False).head(6)
            st.bar_chart(cat_sales, height=310)

    st.markdown('<div class="page-title">Key Executive Highlights</div>', unsafe_allow_html=True)
    h1, h2, h3 = st.columns(3)
    avg_order_val = total_revenue / max(1, total_transactions)
    top_brand = display_df.groupby("Brand")["Revenue"].sum().idxmax() if "Brand" in display_df.columns else "N/A"
    h1.markdown(f'<div class="panel"><b>Average Basket Value:</b> {money(avg_order_val)}<br><span style="color:#777;font-size:0.85rem;">Calculated across {total_transactions:,} recorded customer transactions.</span></div>', unsafe_allow_html=True)
    h2.markdown(f'<div class="panel"><b>Leading Brand:</b> {top_brand}<br><span style="color:#777;font-size:0.85rem;">Highest total sales contributor in current catalog history.</span></div>', unsafe_allow_html=True)
    h3.markdown(f'<div class="panel"><b>Replenishment Priority:</b> {restock_now:,} items below threshold.<br><span style="color:#777;font-size:0.85rem;">Action required in Restock Planner.</span></div>', unsafe_allow_html=True)


# PAGE 2: DAILY SALES ANALYSIS

elif st.session_state.page == "Daily Sales":
    page_header("Daily Sales Analysis", "Inspect daily revenue, unit volume, and period-over-period changes.")

    if "Invoice_Date" in display_df.columns:
        temp = display_df.copy()
        temp["Invoice_Date"] = pd.to_datetime(temp["Invoice_Date"], errors="coerce")
        temp = temp.dropna(subset=["Invoice_Date"])
        temp["Date"] = temp["Invoice_Date"].dt.date
        available_dates = sorted(temp["Date"].unique())

        selected_date = st.selectbox(
            "Select Reporting Date",
            available_dates,
            index=len(available_dates) - 1,
            key="daily_date_selector"
        )

        daily_data = temp[temp["Date"] == selected_date]
        prev_date = selected_date - pd.Timedelta(days=1)
        prev_data = temp[temp["Date"] == prev_date]

        same_day_prev_week = selected_date - pd.Timedelta(days=7)
        prev_week_data = temp[temp["Date"] == same_day_prev_week]

        daily_rev = daily_data["Revenue"].sum()
        daily_units = daily_data["Units"].sum()
        daily_tx = daily_data["Invoice_ID"].nunique()

        prev_rev = prev_data["Revenue"].sum() if not prev_data.empty else None
        prev_week_rev = prev_week_data["Revenue"].sum() if not prev_week_data.empty else None

        rev_diff_prev = (daily_rev - prev_rev) if prev_rev is not None else None
        rev_diff_week = (daily_rev - prev_week_rev) if prev_week_rev is not None else None

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("DAILY REVENUE", money(daily_rev))
        c2.metric("DAILY UNITS", f"{daily_units:,.0f}")
        c3.metric("VS PREVIOUS DAY", money(rev_diff_prev) if rev_diff_prev is not None else "N/A", delta=f"{((daily_rev-prev_rev)/prev_rev*100):.1f}%" if prev_rev else None)
        c4.metric("VS SAME DAY LAST WEEK", money(rev_diff_week) if rev_diff_week is not None else "N/A", delta=f"{((daily_rev-prev_week_rev)/prev_week_rev*100):.1f}%" if prev_week_rev else None)

        st.markdown('<div class="page-title">Product & Brand Sales on Selected Date</div>', unsafe_allow_html=True)
        col_b, col_c = st.columns(2)
        with col_b:
            st.caption("Sales by Brand")
            st.bar_chart(daily_data.groupby("Brand")["Revenue"].sum().sort_values(ascending=False), height=280)
        with col_c:
            st.caption("Sales by Category")
            st.bar_chart(daily_data.groupby("Category")["Revenue"].sum().sort_values(ascending=False), height=280)

        st.markdown('<div class="page-title">Top Selling Items Today</div>', unsafe_allow_html=True)
        prod_breakdown = daily_data.groupby(["Brand", "Category"]).agg(
            Units_Sold=("Units", "sum"),
            Revenue_Generated=("Revenue", "sum")
        ).reset_index().sort_values("Revenue_Generated", ascending=False)
        prod_breakdown["Revenue_Generated"] = prod_breakdown["Revenue_Generated"].apply(money)
        st.dataframe(prod_breakdown, use_container_width=True, hide_index=True)
    else:
        st.warning("Invoice_Date column is not available in the dataset.")


# PAGE 3: WEEKLY & MONTHLY ANALYSIS

elif st.session_state.page == "Weekly & Monthly":
    page_header("Weekly & Monthly Analysis", "Period-over-period aggregate performance tracking.")

    temp = display_df.copy()
    temp["Invoice_Date"] = pd.to_datetime(temp["Invoice_Date"], errors="coerce")
    temp = temp.dropna(subset=["Invoice_Date"])

    tab_w, tab_m = st.tabs([" Weekly Summary", " Monthly Summary"])

    with tab_w:
        temp["Week"] = temp["Invoice_Date"].dt.to_period("W").astype(str)
        weekly = temp.groupby("Week").agg(
            Revenue=("Revenue", "sum"),
            Units=("Units", "sum"),
            Transactions=("Invoice_ID", "nunique")
        ).reset_index()

        st.line_chart(weekly.set_index("Week")["Revenue"], height=300)
        st.dataframe(weekly, use_container_width=True, hide_index=True)

    with tab_m:
        temp["Month_Period"] = temp["Invoice_Date"].dt.to_period("M").astype(str)
        monthly = temp.groupby("Month_Period").agg(
            Revenue=("Revenue", "sum"),
            Units=("Units", "sum"),
            Transactions=("Invoice_ID", "nunique")
        ).reset_index()

        monthly["Revenue_Growth_%"] = monthly["Revenue"].pct_change().fillna(0).mul(100).round(2)
        st.bar_chart(monthly.set_index("Month_Period")["Revenue"], height=300)
        st.dataframe(monthly, use_container_width=True, hide_index=True)


# PAGE 4: PRODUCT PERFORMANCE

elif st.session_state.page == "Product Performance":
    page_header("Product Performance", "Rankings, revenue share, and catalog velocity breakdown.")

    p_col = get_product_column(display_df)
    if p_col:
        prod_perf = display_df.groupby(p_col).agg(
            Total_Revenue=("Revenue", "sum"),
            Units_Sold=("Units", "sum"),
            Avg_Price=("Selling_Price", "mean"),
            Transactions=("Invoice_ID", "nunique")
        ).reset_index().sort_values("Total_Revenue", ascending=False)

        total_cat_rev = prod_perf["Total_Revenue"].sum()
        prod_perf["Revenue_Share_%"] = (prod_perf["Total_Revenue"] / total_cat_rev * 100).round(2)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown('<div class="page-title">Top Products by Revenue</div>', unsafe_allow_html=True)
            st.bar_chart(prod_perf.set_index(p_col)["Total_Revenue"].head(10), height=300)
        with c2:
            st.markdown('<div class="page-title">Top Products by Unit Volume</div>', unsafe_allow_html=True)
            st.bar_chart(prod_perf.set_index(p_col)["Units_Sold"].head(10), height=300)

        st.markdown('<div class="page-title">Complete Catalog Performance Table</div>', unsafe_allow_html=True)
        prod_perf_display = prod_perf.copy()
        prod_perf_display["Total_Revenue"] = prod_perf_display["Total_Revenue"].apply(money)
        prod_perf_display["Avg_Price"] = prod_perf_display["Avg_Price"].apply(money)
        st.dataframe(prod_perf_display, use_container_width=True, hide_index=True)
    else:
        st.info("Product column not found.")


# PAGE 5: BUSINESS INSIGHTS & EVIDENCE-BASED RECOMMENDATIONS

elif st.session_state.page == "Business Insights":
    page_header("Business Insights & Evidence-Based Recommendations", "Algorithmic decision support grounded in transaction facts.")

    temp = display_df.copy()
    temp["Invoice_Date"] = pd.to_datetime(temp["Invoice_Date"], errors="coerce")
    temp = temp.dropna(subset=["Invoice_Date"])
    temp["Date"] = temp["Invoice_Date"].dt.date
    dates = sorted(temp["Date"].unique())

    # Evidence 1: Peak Sales Day
    daily_totals = temp.groupby("Date")["Revenue"].sum()
    peak_date = daily_totals.idxmax()
    peak_rev = daily_totals.max()

    # Evidence 2: Category Concentration
    cat_totals = temp.groupby("Category")["Revenue"].sum()
    top_cat = cat_totals.idxmax()
    top_cat_share = (cat_totals.max() / cat_totals.sum()) * 100

    # Evidence 3: Urgent Restocking
    restock_count = int((display_df["Restock_Status"] == "Restock Now").sum()) if "Restock_Status" in display_df.columns else 0

    st.markdown(
        f"""
        <div class="panel">
            <h4 style="color:#ffffff;margin-top:0;"> Insight 1: Sales Concentration by Category</h4>
            <p><b>Finding:</b> Category <b>{top_cat}</b> is your highest revenue driver, accounting for <b>{top_cat_share:.1f}%</b> of total catalog revenue ({money(cat_totals.max())}).</p>
            <p style="color:#a7a7a7;"><b>Recommended Action:</b> Prioritize prime shelf space, maintain vendor relationships, and prevent stockouts for leading brands within {top_cat}.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        f"""
        <div class="panel">
            <h4 style="color:#ffffff;margin-top:0;"> Insight 2: Historical Peak Demand Spike</h4>
            <p><b>Finding:</b> Peak store revenue occurred on <b>{peak_date}</b> generating <b>{money(peak_rev)}</b> across transactions.</p>
            <p style="color:#a7a7a7;"><b>Recommended Action:</b> Audit promotional calendars, weekend traffic, and seasonal footfall patterns corresponding to that period to recreate successful campaign dynamics.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    if restock_count > 0:
        st.markdown(
            f"""
            <div class="panel" style="border-color:#4a2828 !important;">
                <h4 style="color:#e0a0a0;margin-top:0;"> Insight 3: Critical Replenishment Alert</h4>
                <p><b>Finding:</b> <b>{restock_count:,} items</b> in the catalog currently have inventory levels at or below their minimum reorder point.</p>
                <p style="color:#a7a7a7;"><b>Recommended Action:</b> Navigate to the <b>Inventory & Restock</b> hub and download the pre-generated Purchase Order CSV to initiate immediate supplier reorders.</p>
            </div>
            """,
            unsafe_allow_html=True
        )


# PAGE 6: SALES FORECASTING PAGE (Chronological Models)

elif st.session_state.page == "Sales Forecasting":
    page_header("Sales Forecasting", "Predict future store sales using trained machine learning regression models.")

    temp = display_df.copy()
    temp["Invoice_Date"] = pd.to_datetime(temp["Invoice_Date"], errors="coerce")
    temp = temp.dropna(subset=["Invoice_Date"])
    temp["Date"] = temp["Invoice_Date"].dt.date
    daily_hist = temp.groupby("Date").agg(
        Revenue=("Revenue", "sum"),
        Units=("Units", "sum"),
        Transactions=("Invoice_ID", "nunique")
    ).reset_index()

    f_model = models.get("forecasting_regressor")
    if f_model is not None:
        horizon = st.slider("Select Forecast Horizon (Days)", min_value=3, max_value=30, value=7, step=1)
        forecast_df = forecast_daily_sales(daily_hist, f_model, horizon_days=horizon)

        if not forecast_df.empty:
            c1, c2 = st.columns(2)
            c1.metric("PROJECTED PERIOD REVENUE", money(forecast_df["Forecasted_Revenue"].sum()))
            c2.metric("PROJECTED UNITS", f"{forecast_df['Forecasted_Units'].sum():,}")

            st.markdown('<div class="page-title">Daily Sales Projections</div>', unsafe_allow_html=True)
            st.line_chart(forecast_df.set_index("Date")["Forecasted_Revenue"], height=300)

            st.markdown('<div class="page-title">Forecast Table</div>', unsafe_allow_html=True)
            display_fc = forecast_df.copy()
            display_fc["Forecasted_Revenue"] = display_fc["Forecasted_Revenue"].apply(money)
            st.dataframe(display_fc, use_container_width=True, hide_index=True)
    else:
        st.warning("Forecasting model artifact not available. Please ensure models/forecasting/forecast_regressor.pkl is loaded.")


# PAGE 7: ML MODELS & EVALUATION PAGE

elif st.session_state.page == "ML Models & Evaluation":
    page_header("ML Models & Evaluation", "Verified evaluation results on untouched held-out test splits.")

    tab_reg, tab_clf = st.tabs(["Regression & Forecasting Evaluation", " Demand Classification Evaluation"])

    with tab_reg:
        st.markdown('<div class="page-title">Transaction-Level Revenue Regression</div>', unsafe_allow_html=True)
        trans_eval_path = os.path.join(PROJECT_ROOT, "outputs/results/transaction_regression_comparison.csv")
        if os.path.exists(trans_eval_path):
            trans_comp = pd.read_csv(trans_eval_path)
            st.dataframe(trans_comp, use_container_width=True, hide_index=True)
        else:
            st.info("Transaction regression comparison results not found.")


    with tab_clf:
        st.markdown('<div class="page-title">Classification & Decision Support Models</div>', unsafe_allow_html=True)
        st.markdown("Inventory Replenishment Decision Support")

        inv_eval_path = os.path.join(PROJECT_ROOT, "outputs/results/inventory_replenishment_comparison.csv")
        if os.path.exists(inv_eval_path):
            inv_comp = pd.read_csv(inv_eval_path)
            st.dataframe(inv_comp, use_container_width=True, hide_index=True)

        meta_inv_path = os.path.join(PROJECT_ROOT, "models/classification/upgraded/inventory_decision_metadata.json")
        if os.path.exists(meta_inv_path):
            try:
                with open(meta_inv_path, "r") as f:
                    meta_i = json.load(f)
                st.markdown("#### Decision Support Confusion Matrix (Stratified Held-Out Test Set: 20,000 Records)")
                cms_i = meta_i.get("confusion_matrices", {})
                if "Gradient Boosting" in cms_i:
                    cm_i_df = pd.DataFrame(cms_i["Gradient Boosting"], index=["Actual Stock Safe", "Actual Restock Required"], columns=["Pred Safe", "Pred Restock"])
                    st.dataframe(cm_i_df, use_container_width=True)
            except Exception:
                pass



# PAGE 8: INVENTORY & RESTOCK HUB (Preserved & Enhanced)

elif st.session_state.page in ["Inventory & Restock", "Inventory", "Restock Planner"]:
    page_header("Inventory & Restock", "Stock health tracking, operational reorders, and purchase order planning.")

    prediction_col = (
        "Random_Forest_Prediction"
        if "Random_Forest_Prediction" in display_df.columns
        else ("Decision_Tree_Prediction" if "Decision_Tree_Prediction" in display_df.columns else None)
    )

    restock_now_df = display_df[display_df["Restock_Status"] == "Restock Now"].copy() if "Restock_Status" in display_df.columns else pd.DataFrame()
    restock_soon_df = display_df[display_df["Restock_Status"] == "Restock Soon"].copy() if "Restock_Status" in display_df.columns else pd.DataFrame()
    healthy_count = max(0, len(display_df) - len(restock_now_df) - len(restock_soon_df))
    suggested_units = restock_now_df["Suggested_Order_Quantity"].fillna(0).sum() if not restock_now_df.empty and "Suggested_Order_Quantity" in restock_now_df.columns else 0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("HEALTHY STOCK", f"{healthy_count:,}")
    c2.metric("WATCHLIST (LOW)", f"{len(restock_soon_df):,}")
    c3.metric("REORDER NOW", f"{len(restock_now_df):,}")
    c4.metric("UNITS TO ORDER", f"{suggested_units:,.0f}")

    tab1, tab2, tab3 = st.tabs([" Immediate Reorders", " Watchlist (Restock Soon)", " Catalog Stock Status"])

    with tab1:
        if not restock_now_df.empty:
            st.markdown(f'<div class="danger-box"><b>{len(restock_now_df):,} items require immediate purchase orders.</b> Stock is at or below the minimum reorder threshold.</div>', unsafe_allow_html=True)
            cols = unique_columns([c for c in [get_product_column(display_df), "Brand", "Category", "Stock_On_Hand", "Reorder_Level", "Lead_Time_Days", "Suggested_Order_Quantity"] if c and c in restock_now_df.columns])
            order_table = restock_now_df[cols].copy().rename(columns={"Stock_On_Hand": "Current Stock", "Reorder_Level": "Reorder Level", "Suggested_Order_Quantity": "Suggested Order", "Lead_Time_Days": "Lead Time (Days)"})
            st.dataframe(order_table.head(100), use_container_width=True, hide_index=True)
            csv_bytes = order_table.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download Purchase Order (CSV)",
                data=csv_bytes,
                file_name="retailiq_purchase_orders.csv",
                mime="text/csv",
                key="download_urgent_reorders_hub"
            )
        else:
            st.markdown('<div class="success-box"><b>All stock levels sufficient.</b> No immediate purchase orders required today.</div>', unsafe_allow_html=True)

    with tab2:
        if not restock_soon_df.empty:
            st.markdown(f'<div class="warning-box"><b>{len(restock_soon_df):,} items are running low.</b> Current inventory is below expected lead-time demand.</div>', unsafe_allow_html=True)
            cols = unique_columns([c for c in [get_product_column(display_df), "Brand", "Category", "Stock_On_Hand", "Lead_Time_Days", "Suggested_Order_Quantity"] if c and c in restock_soon_df.columns])
            watchlist_table = restock_soon_df[cols].copy().rename(columns={"Stock_On_Hand": "Current Stock", "Suggested_Order_Quantity": "Suggested Order", "Lead_Time_Days": "Lead Time (Days)"})
            st.dataframe(watchlist_table.head(100), use_container_width=True, hide_index=True)
        else:
            st.markdown('<div class="success-box"><b>No items on watchlist.</b> Upcoming demand is covered by current stock.</div>', unsafe_allow_html=True)

    with tab3:
        catalog_cols = unique_columns([c for c in [get_product_column(display_df), "Brand", "Category", "Stock_On_Hand", "Reorder_Level", prediction_col, "Restock_Status"] if c and c in display_df.columns])
        st.dataframe(display_df[catalog_cols].head(100), use_container_width=True, hide_index=True)


# PAGE 9: UPDATE SALES DATA (Safe CSV Upload & Persistence)

elif st.session_state.page == "Update Sales Data":
    page_header("Update Sales Data", "Upload daily sales CSV files with schema validation, deduplication, and persistence.")

    st.markdown('<div class="action-box"><div class="action-title">Upload daily retail data</div><div class="action-text">Validate schema, verify row counts, preview contents, and confirm additions into persistent storage.</div></div>', unsafe_allow_html=True)

    uploader_key = f"daily_csv_uploader_{st.session_state.upload_widget_version}"
    uploaded_file = st.file_uploader(
        "Select Daily Sales CSV",
        type=["csv", "xlsx"],
        key=uploader_key,
    )

    if uploaded_file is not None:
        file_bytes = uploaded_file.getvalue()
        val_res = validate_upload_content(file_bytes, uploaded_file.name)

        if not val_res["valid"]:
            st.markdown(f'<div class="danger-box"><b>Upload Rejected:</b> {val_res["error"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="success-box"><b>Validation Passed:</b> File contains {val_res["row_count"]:,} valid records ready for import.</div>', unsafe_allow_html=True)
            st.markdown('<div class="page-title">Upload Preview</div>', unsafe_allow_html=True)
            preview_cols = [c for c in ["Invoice_ID", "Invoice_Date", "Brand", "Category", "Units", "Selling_Price", "Revenue"] if c in val_res["processed_df"].columns]
            if not preview_cols:
                preview_cols = val_res["processed_df"].columns[:6].tolist()
            st.dataframe(val_res["processed_df"][preview_cols].head(10), use_container_width=True, hide_index=True)

            if st.button("Confirm and Commit to Dashboard", key="confirm_upload_btn"):
                try:
                    persist_accepted_upload(
                        file_bytes,
                        uploaded_file.name,
                        val_res["file_hash"],
                        val_res["processed_df"],
                    )
                    # Remove cached reporting/model results so every page reloads
                    # the newly persisted dataset on the next rerun.
                    st.cache_data.clear()
                    st.session_state["upload_widget_version"] += 1
                    st.session_state.pop("daily_date_selector", None)
                    st.session_state["upload_success_msg"] = (
                        f"Successfully imported '{uploaded_file.name}' "
                        f"({val_res['row_count']:,} records). "
                        "Reporting data and analytics have been refreshed."
                    )
                    st.rerun()
                except Exception as exc:
                    st.error(f"Could not commit this upload: {exc}")

    if "upload_success_msg" in st.session_state and st.session_state["upload_success_msg"]:
        st.markdown(f'<div class="success-box"><b>Update Applied:</b> {st.session_state["upload_success_msg"]}</div>', unsafe_allow_html=True)
        # Clear after displaying once
        del st.session_state["upload_success_msg"]

    # Upload history with per-file deletion controls.
    manifest = get_upload_manifest()
    st.markdown('<div class="page-title">Persistent Upload History</div>', unsafe_allow_html=True)
    if not manifest:
        st.info("No additional files have been uploaded. The approved historical dataset is kept separately and cannot be deleted here.")
    else:
        st.caption("Delete removes only the selected uploaded file from reporting data. The original historical dataset is not affected.")
        for entry in reversed(manifest):
            stored_name = entry.get("stored_filename", "")
            original_name = entry.get("original_filename", stored_name)
            row_count = entry.get("rows", "—")
            uploaded_at = entry.get("timestamp", "")
            col_info, col_delete = st.columns([6, 1])
            with col_info:
                st.markdown(
                    f"**{original_name}**  \n\n"
                    f"<span style='color:#999;font-size:0.85rem'>Rows: {row_count} · {uploaded_at}</span>",
                    unsafe_allow_html=True,
                )
            with col_delete:
                if st.button("Delete", key=f"delete_upload_{entry.get('file_hash', stored_name)}"):
                    st.session_state["pending_upload_delete"] = stored_name
                    st.rerun()

    pending_delete = st.session_state.get("pending_upload_delete")
    if pending_delete:
        pending_entry = next((item for item in manifest if item.get("stored_filename") == pending_delete), None)
        if pending_entry:
            st.warning(f"Delete '{pending_entry.get('original_filename', pending_delete)}' from uploaded data? This cannot be undone.")
            confirm_col, cancel_col = st.columns(2)
            with confirm_col:
                if st.button("Confirm delete", key="confirm_upload_delete", type="primary"):
                    try:
                        removed = delete_accepted_upload(pending_delete)
                        st.cache_data.clear()
                        st.session_state.pop("pending_upload_delete", None)
                        st.session_state.pop("daily_date_selector", None)
                        st.session_state["upload_delete_success"] = (
                            f"Removed '{removed.get('original_filename', pending_delete)}' from reporting data."
                        )
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Could not delete upload: {exc}")
            with cancel_col:
                if st.button("Cancel", key="cancel_upload_delete"):
                    st.session_state.pop("pending_upload_delete", None)
                    st.rerun()
        else:
            st.session_state.pop("pending_upload_delete", None)

    delete_success = st.session_state.pop("upload_delete_success", None)
    if delete_success:
        st.success(delete_success)

# ============================================================
# PAGE 10: DOWNLOADABLE REPORTS
# ============================================================
elif st.session_state.page == "Downloadable Reports":
    page_header("Downloadable Reports", "Export data and analytics for executive and supply chain workflows.")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown('<div class="panel"><b>Daily Sales Summary</b><br><span style="color:#777;font-size:0.85rem;">Export aggregated daily revenue and units.</span>', unsafe_allow_html=True)
        temp = display_df.copy()
        temp["Date"] = pd.to_datetime(temp["Invoice_Date"]).dt.date
        daily_export = temp.groupby("Date").agg(Revenue=("Revenue", "sum"), Units=("Units", "sum"), Invoices=("Invoice_ID", "nunique")).reset_index()
        st.download_button("Export Daily Sales (CSV)", daily_export.to_csv(index=False).encode("utf-8"), "daily_sales_report.csv", "text/csv")
        st.markdown('</div>', unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="panel"><b>Product Performance</b><br><span style="color:#777;font-size:0.85rem;">Export brand and category revenue rankings.</span>', unsafe_allow_html=True)
        prod_export = display_df.groupby(["Brand", "Category"]).agg(Revenue=("Revenue", "sum"), Units=("Units", "sum")).reset_index()
        st.download_button("Export Catalog Rankings (CSV)", prod_export.to_csv(index=False).encode("utf-8"), "product_performance_report.csv", "text/csv")
        st.markdown('</div>', unsafe_allow_html=True)

    with c3:
        st.markdown('<div class="panel"><b>ML Model Comparison</b><br><span style="color:#777;font-size:0.85rem;">Export verified model metrics and benchmark scores.</span>', unsafe_allow_html=True)
        reg_eval_path = os.path.join(PROJECT_ROOT, "outputs/results/transaction_regression_comparison.csv")
        if not os.path.exists(reg_eval_path):
            reg_eval_path = os.path.join(PROJECT_ROOT, "outputs/results/regression_model_comparison.csv")
        if os.path.exists(reg_eval_path):
            ml_export = pd.read_csv(reg_eval_path)
            st.download_button("Export Model Scores (CSV)", ml_export.to_csv(index=False).encode("utf-8"), "ml_model_evaluation.csv", "text/csv")
        st.markdown('</div>', unsafe_allow_html=True)

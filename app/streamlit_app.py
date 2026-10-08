import os

import sys

import streamlit as st

import pandas as pd



# ------------------------------------------------------------

# PROJECT SETUP

# ------------------------------------------------------------

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

PROJECT_ROOT = CURRENT_DIR if os.path.exists(os.path.join(CURRENT_DIR, "src")) else os.path.dirname(CURRENT_DIR)

if PROJECT_ROOT not in sys.path:

    sys.path.insert(0, PROJECT_ROOT)



from src.preprocessing import validate_data, preprocess_data, get_module_support

from src.model_loader import load_models

from src.model_predictions import predict_revenue, predict_inventory

from src.restocking import calculate_restocking

from src.clustering import predict_clusters

from src.association import get_associations



st.set_page_config(

    page_title="RetailIQ",

    page_icon=None,

    layout="wide",

    initial_sidebar_state="expanded",

)



# ------------------------------------------------------------

# PROFESSIONAL DARK UI

# ------------------------------------------------------------

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

    .page-title { font-size: 1.65rem; font-weight: 750; color: #ffffff; margin-top: .5rem; margin-bottom: .2rem; }

    .page-subtitle { color: #777777; font-size: .86rem; margin-bottom: 1.2rem; }



    .panel, .action-box, .success-box, .warning-box, .danger-box, .welcome-box {

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



# ------------------------------------------------------------

# MODELS

# ------------------------------------------------------------

@st.cache_resource(show_spinner="Loading machine learning models...")
def get_cached_models():
    return load_models()


try:

    models = get_cached_models()

    model_load_error = None

except Exception as exc:

    models = None

    model_load_error = str(exc)



# ------------------------------------------------------------

# SESSION STATE

# ------------------------------------------------------------

def init_state():

    defaults = {

        "page": "Overview",

        "cumulative_raw_df": None,

        "cumulative_processed_df": None,

        "uploaded_df": None,

        "display_df": None,

        "uploaded_filename": None,

        "upload_history": [],

        "upload_widget_version": 0,

        "module_support": {},

        "data_update_summary": None,

    }

    for key, value in defaults.items():

        if key not in st.session_state:

            st.session_state[key] = value



init_state()





def module_available(name):

    return bool(st.session_state.get("module_support", {}).get(name, {}).get("available", False))





def missing_for(name):

    return st.session_state.get("module_support", {}).get(name, {}).get("missing_columns", [])





def money(value):

    try:

        return f"₹{float(value):,.0f}"

    except Exception:

        return "₹0"





def unique_columns(columns):

    """Return columns in the same order without duplicates."""

    seen = set()

    result = []

    for column in columns:

        if column and column not in seen:

            seen.add(column)

            result.append(column)

    return result





def get_product_column(df):

    for col in ["Product", "Product_Name", "Item", "Brand"]:

        if col in df.columns:

            return col

    return None





def require_data():

    if st.session_state.display_df is None:

        st.markdown(

            '<div class="warning-box"><b>No sales data loaded.</b><br>Open Update Sales Data and upload a retail CSV or Excel file.</div>',

            unsafe_allow_html=True,

        )

        st.stop()

    return st.session_state.display_df





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

            # Keep one clear UI-facing name while remaining compatible with src/restocking.py.

            if "Suggested_Quantity" in processed.columns and "Suggested_Order_Quantity" not in processed.columns:

                processed["Suggested_Order_Quantity"] = processed["Suggested_Quantity"]

        except Exception:

            pass



    if support.get("Product Clustering", {}).get("available", False):

        try:

            processed = predict_clusters(processed, models)

        except Exception:

            pass



    return processed, support



# ------------------------------------------------------------

# SIDEBAR

# ------------------------------------------------------------

with st.sidebar:

    st.markdown(

        '<div class="brand">RetailIQ</div><div class="brand-sub">Intelligent retail decision support</div>',

        unsafe_allow_html=True,

    )



    st.markdown('<div class="menu-label">BUSINESS</div>', unsafe_allow_html=True)

    pages = [
        "Overview",
        "Inventory & Restock",
        "Sales & Forecast",
        "Product Intelligence",
    ]

    for label in pages:

        if st.button(label, use_container_width=True, key=f"nav_{label}"):

            st.session_state.page = label

            st.rerun()



    st.markdown('<div class="menu-label">DATA</div>', unsafe_allow_html=True)

    if st.button("Update Sales Data", use_container_width=True, key="nav_update"):

        st.session_state.page = "Update Sales Data"

        st.rerun()



    st.divider()

    if st.session_state.display_df is not None:

        st.markdown('<div style="color:#666;font-size:.68rem;font-weight:700;letter-spacing:.12em;">DATA STATUS</div>', unsafe_allow_html=True)

        st.markdown('<div style="color:#a7a7a7;font-weight:600;margin-top:4px;">Dashboard connected</div>', unsafe_allow_html=True)

        st.caption(f"{len(st.session_state.display_df):,} records")

    else:

        st.caption("No sales data loaded")



# ------------------------------------------------------------

# HEADER

# ------------------------------------------------------------

st.markdown(

    '<div class="app-header"><div class="header-brand">RetailIQ</div><div class="header-tagline">Intelligent Retail Decision Support System</div><div class="header-status">Workspace Ready</div></div>',

    unsafe_allow_html=True,

)



# ------------------------------------------------------------

# UPDATE SALES DATA

# ------------------------------------------------------------

def build_sample_transaction_rules():
    """Generate realistic supermarket baskets and calculate association rules."""
    import random
    from itertools import combinations

    rng = random.Random(42)
    patterns = [
        ("Bread", "Butter", "Milk"),
        ("Tea", "Biscuits", "Sugar"),
        ("Coffee", "Milk", "Biscuits"),
        ("Pasta", "Pasta Sauce", "Cheese"),
        ("Rice", "Cooking Oil", "Dal"),
        ("Atta", "Cooking Oil", "Dal"),
        ("Eggs", "Bread", "Milk"),
        ("Cereal", "Milk", "Bananas"),
        ("Chips", "Soft Drink", "Chocolate"),
        ("Tomatoes", "Onions", "Cooking Oil"),
        ("Shampoo", "Conditioner", "Body Wash"),
        ("Detergent", "Fabric Softener", "Dishwash Liquid"),
    ]

    all_items = sorted({item for basket in patterns for item in basket})
    transactions = []
    for _ in range(1000):
        basket = set(rng.choice(patterns))
        for item in all_items:
            if item not in basket and rng.random() < 0.035:
                basket.add(item)
        transactions.append(basket)

    n = len(transactions)
    item_count = {item: sum(item in basket for basket in transactions) for item in all_items}
    pair_count = {}
    for basket in transactions:
        for a, b in combinations(sorted(basket), 2):
            pair_count[(a, b)] = pair_count.get((a, b), 0) + 1

    rows = []
    for (a, b), count in pair_count.items():
        support = count / n
        if support < 0.03:
            continue
        for antecedent, consequent in ((a, b), (b, a)):
            confidence = count / item_count[antecedent]
            consequent_support = item_count[consequent] / n
            lift = confidence / consequent_support if consequent_support else 0
            if confidence >= 0.30 and lift >= 1.05:
                rows.append({
                    "Product": antecedent,
                    "Recommended_Product": consequent,
                    "Support": round(support, 3),
                    "Confidence": round(confidence * 100, 2),
                    "Lift": round(lift, 2),
                })

    return pd.DataFrame(rows).sort_values(
        ["Lift", "Confidence", "Support"], ascending=False
    ).reset_index(drop=True)




if st.session_state.page == "Update Sales Data":

    page_header("Update Sales Data", "Upload retail sales data to refresh the current dashboard.")



    st.markdown(

        '<div class="action-box"><div class="action-title">Upload retail data</div><div class="action-text">CSV and Excel files are supported. Multiple uploads can be combined during the current session.</div></div>',

        unsafe_allow_html=True,

    )



    uploaded_files = st.file_uploader(

        "Upload retail sales files",

        type=["csv", "xlsx"],

        accept_multiple_files=True,

        key=f"main_data_uploader_{st.session_state.upload_widget_version}",

        help="Select one or more CSV or Excel files. All selected files are combined for the current dashboard session.",

    )



    if uploaded_files:

        try:

            new_frames = []

            new_history = []



            for uploaded_file in uploaded_files:

                raw_df = (

                    pd.read_excel(uploaded_file)

                    if uploaded_file.name.lower().endswith(".xlsx")

                    else pd.read_csv(uploaded_file)

                )

                if raw_df.empty:

                    st.error(f"{uploaded_file.name} is empty.")

                    st.stop()

                new_frames.append(raw_df)

                new_history.append({"filename": uploaded_file.name, "rows": len(raw_df)})



            current_raw = st.session_state.cumulative_raw_df
            new_raw = pd.concat(new_frames, ignore_index=True)
            combined_raw = new_raw.copy() if current_raw is None else pd.concat([current_raw, new_raw], ignore_index=True)

            with st.spinner("Processing retail data..."):
                # Normalize each file BEFORE combining them. Different files can use
                # different aliases (for example Product_Name vs Brand or Quantity vs Units).
                # Combining raw files first can separate equivalent fields into different
                # columns and cause one dataset's values to be ignored.
                new_processed_frames = []
                for uploaded_file, raw_df in zip(uploaded_files, new_frames):
                    processed_one = preprocess_data(raw_df.copy())
                    is_valid, missing_columns = validate_data(processed_one)

                    if not is_valid:
                        st.error(f"{uploaded_file.name} is not compatible with RetailIQ.")
                        if missing_columns:
                            st.write("Required retail fields not found:")
                            for column in missing_columns:
                                st.write(f"- {column}")
                        st.stop()

                    new_processed_frames.append(processed_one)

                new_processed = pd.concat(new_processed_frames, ignore_index=True, sort=False)
                current_processed = st.session_state.cumulative_processed_df
                combined_processed = (
                    new_processed.copy()
                    if current_processed is None
                    else pd.concat([current_processed, new_processed], ignore_index=True, sort=False)
                )

                processed_df, support = run_models(combined_processed)

            st.session_state.cumulative_raw_df = combined_raw
            st.session_state.cumulative_processed_df = combined_processed
            st.session_state.uploaded_df = new_processed
            st.session_state.display_df = processed_df
            st.session_state.uploaded_filename = ", ".join(item["filename"] for item in new_history)

            st.session_state.upload_history.extend(new_history)

            st.session_state.data_update_summary = {

                "files_added": len(st.session_state.upload_history),

                "latest_upload_files": len(new_history),

                "latest_upload_records": len(new_raw),

                "total_records": len(combined_raw),

            }

            st.session_state.upload_widget_version += 1

            st.rerun()



        except Exception as exc:

            st.error(f"The file could not be processed: {exc}")

            st.stop()



    if st.session_state.data_update_summary:

        summary = st.session_state.data_update_summary

        st.markdown('<div class="page-title">Current Data</div>', unsafe_allow_html=True)

        c1, c2, c3 = st.columns(3)

        c1.metric("FILES LOADED", f"{summary['files_added']:,}")

        c2.metric("LATEST RECORDS", f"{summary['latest_upload_records']:,}")

        c3.metric("TOTAL RECORDS", f"{summary['total_records']:,}")



        st.markdown(

            f'<div class="success-box">The dashboard is using {summary["total_records"]:,} records from {summary["files_added"]:,} uploaded file(s). Data is held in the current Streamlit session.</div>',

            unsafe_allow_html=True,

        )



        if st.session_state.upload_history:

            st.markdown('<div class="page-title">Upload History</div>', unsafe_allow_html=True)

            st.dataframe(pd.DataFrame(st.session_state.upload_history), use_container_width=True, hide_index=True)



        if st.button("Clear Current Data", key="clear_current_data"):

            for key, value in {

                "cumulative_raw_df": None, "cumulative_processed_df": None, "uploaded_df": None, "display_df": None,

                "uploaded_filename": None, "upload_history": [], "module_support": {},

                "data_update_summary": None,

            }.items():

                st.session_state[key] = value

            st.session_state.upload_widget_version += 1

            st.rerun()



    st.stop()



# ------------------------------------------------------------

# OVERVIEW

# ------------------------------------------------------------

elif st.session_state.page in ["Overview", "Home"]:

    if st.session_state.display_df is None:

        st.markdown(

            '<div class="welcome-box"><div class="welcome-title">Welcome to RetailIQ</div><div class="welcome-text">A professional retail intelligence workspace for sales, inventory and restocking decisions.</div></div>',

            unsafe_allow_html=True,

        )

        st.markdown('<div class="page-title">Getting Started</div>', unsafe_allow_html=True)

        c1, c2, c3 = st.columns(3)

        c1.markdown('<div class="action-box"><div class="action-title">1. Upload data</div><div class="action-text">Open Update Sales Data and upload your retail CSV or Excel file.</div></div>', unsafe_allow_html=True)

        c2.markdown('<div class="action-box"><div class="action-title">2. Review insights</div><div class="action-text">RetailIQ calculates sales, inventory, product and restocking insights.</div></div>', unsafe_allow_html=True)

        c3.markdown('<div class="action-box"><div class="action-title">3. Take action</div><div class="action-text">Use the recommendations to identify stock and sales priorities.</div></div>', unsafe_allow_html=True)

        if st.button("Open Update Sales Data", key="home_upload"):

            st.session_state.page = "Update Sales Data"

            st.rerun()

        st.stop()



    df = require_data()

    page_header("Business Overview", "A concise view of current retail performance.")



    total_revenue = df["Revenue"].sum() if "Revenue" in df.columns else 0

    total_units = df["Units"].sum() if "Units" in df.columns else 0

    product_col = get_product_column(df)

    total_products = df[product_col].nunique() if product_col else 0

    restock_now = int((df["Restock_Status"] == "Restock Now").sum()) if "Restock_Status" in df.columns else 0



    c1, c2, c3, c4 = st.columns(4)

    c1.metric("SALES", money(total_revenue))

    c2.metric("UNITS SOLD", f"{total_units:,.0f}")

    c3.metric("PRODUCTS", f"{total_products:,}")

    c4.metric("RESTOCK NOW", f"{restock_now:,}")



    if "Restock_Status" in df.columns:

        urgent = df[df["Restock_Status"] == "Restock Now"].copy()

        st.markdown('<div class="page-title">Priority Actions</div>', unsafe_allow_html=True)

        if not urgent.empty:

            st.markdown(f'<div class="danger-box"><b>{len(urgent):,} records require restocking.</b> Review the Restock Planner for recommended quantities.</div>', unsafe_allow_html=True)

        else:

            st.markdown('<div class="success-box"><b>No urgent restocking actions.</b> Current stock levels are above the immediate reorder threshold.</div>', unsafe_allow_html=True)



    if "Invoice_Date" in df.columns and "Revenue" in df.columns:

        temp = df.copy()

        temp["Invoice_Date"] = pd.to_datetime(temp["Invoice_Date"], errors="coerce")

        temp = temp.dropna(subset=["Invoice_Date"])

        if not temp.empty:

            monthly = temp.assign(Month=temp["Invoice_Date"].dt.to_period("M").astype(str)).groupby("Month")["Revenue"].sum()

            st.markdown('<div class="page-title">Sales Trend</div>', unsafe_allow_html=True)

            st.line_chart(monthly, height=320)



    if "Category" in df.columns and "Revenue" in df.columns:

        st.markdown('<div class="page-title">Top Categories</div>', unsafe_allow_html=True)

        category_sales = df.groupby("Category")["Revenue"].sum().sort_values(ascending=False).head(8)

        st.bar_chart(category_sales, height=300)



# ------------------------------------------------------------
# INVENTORY & RESTOCK (UNIFIED HUB)
# ------------------------------------------------------------
elif st.session_state.page in ["Inventory & Restock", "Inventory", "Restock Planner", "Needs Attention"]:
    df = require_data()
    page_header("Inventory & Restock", "Unified stock health tracking, inventory classification, and purchase order planning.")

    prediction_col = (
        "Random_Forest_Prediction"
        if "Random_Forest_Prediction" in df.columns
        else ("Decision_Tree_Prediction" if "Decision_Tree_Prediction" in df.columns else ("KNN_Prediction" if "KNN_Prediction" in df.columns else None))
    )

    total_records = len(df)
    restock_now_df = df[df["Restock_Status"] == "Restock Now"].copy() if "Restock_Status" in df.columns else pd.DataFrame()
    restock_soon_df = df[df["Restock_Status"] == "Restock Soon"].copy() if "Restock_Status" in df.columns else pd.DataFrame()

    healthy_count = (df[prediction_col] == "Healthy Stock").sum() if prediction_col else max(0, total_records - len(restock_now_df) - len(restock_soon_df))
    suggested_units = restock_now_df["Suggested_Order_Quantity"].fillna(0).sum() if not restock_now_df.empty and "Suggested_Order_Quantity" in restock_now_df.columns else 0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("HEALTHY STOCK", f"{healthy_count:,}")
    c2.metric("LOW / WATCHLIST", f"{len(restock_soon_df):,}")
    c3.metric("REORDER NOW", f"{len(restock_now_df):,}")
    c4.metric("UNITS TO ORDER", f"{suggested_units:,.0f}")

    if prediction_col:
        health_series = df[prediction_col].value_counts().reindex(["Healthy Stock", "Low Stock", "Overstocked"], fill_value=0)
        st.markdown('<div class="page-title">Stock Health Breakdown</div>', unsafe_allow_html=True)
        st.bar_chart(health_series, height=260)

    st.markdown('<div class="page-title">Operational Reorder Hub</div>', unsafe_allow_html=True)
    tab1, tab2, tab3 = st.tabs(["🚨 Immediate Reorders", "⚠️ Watchlist (Restock Soon)", "📦 Catalog Stock Status"])

    with tab1:
        if not restock_now_df.empty:
            st.markdown(
                f'<div class="danger-box"><b>{len(restock_now_df):,} items require immediate purchase orders.</b> Stock is at or below the minimum reorder threshold.</div>',
                unsafe_allow_html=True,
            )
            cols = unique_columns([c for c in [get_product_column(df), "Brand", "Category", "Stock_On_Hand", "Reorder_Level", "Lead_Time_Days", "Suggested_Order_Quantity"] if c and c in restock_now_df.columns])
            order_table = restock_now_df[cols].copy().rename(columns={"Stock_On_Hand": "Current Stock", "Reorder_Level": "Reorder Level", "Suggested_Order_Quantity": "Suggested Order", "Lead_Time_Days": "Lead Time (Days)"})

            if len(order_table) > 100:
                st.caption(f"Showing top 100 of {len(order_table):,} urgent reorder items:")
                st.dataframe(order_table.head(100), use_container_width=True, hide_index=True)
            else:
                st.dataframe(order_table, use_container_width=True, hide_index=True)

            csv_bytes = order_table.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download Purchase Order (CSV)",
                data=csv_bytes,
                file_name="retailiq_purchase_orders.csv",
                mime="text/csv",
                key="download_urgent_reorders",
            )
        else:
            st.markdown('<div class="success-box"><b>All stock levels sufficient.</b> No immediate purchase orders required today.</div>', unsafe_allow_html=True)

    with tab2:
        if not restock_soon_df.empty:
            st.markdown(
                f'<div class="warning-box"><b>{len(restock_soon_df):,} items are running low.</b> Current inventory is below expected lead-time demand.</div>',
                unsafe_allow_html=True,
            )
            cols = unique_columns([c for c in [get_product_column(df), "Brand", "Category", "Stock_On_Hand", "Lead_Time_Days", "Suggested_Order_Quantity"] if c and c in restock_soon_df.columns])
            watchlist_table = restock_soon_df[cols].copy().rename(columns={"Stock_On_Hand": "Current Stock", "Suggested_Order_Quantity": "Suggested Order", "Lead_Time_Days": "Lead Time (Days)"})
            st.dataframe(watchlist_table.head(100), use_container_width=True, hide_index=True)
        else:
            st.markdown('<div class="success-box"><b>No items on watchlist.</b> Upcoming demand is covered by current stock.</div>', unsafe_allow_html=True)

    with tab3:
        st.markdown('<div class="page-title">Catalog Inventory Status</div>', unsafe_allow_html=True)
        catalog_cols = unique_columns([c for c in [get_product_column(df), "Brand", "Category", "Stock_On_Hand", "Reorder_Level", prediction_col, "Restock_Status"] if c and c in df.columns])
        st.dataframe(df[catalog_cols].head(100), use_container_width=True, hide_index=True)



# ------------------------------------------------------------

# SALES & FORECAST

# ------------------------------------------------------------

elif st.session_state.page == "Sales & Forecast":

    df = require_data()

    page_header("Sales & Forecast", "Review actual sales and model-based expected sales values.")



    actual_sales = df["Revenue"].sum() if "Revenue" in df.columns else 0

    expected_sales = df["Predicted_Revenue"].sum() if "Predicted_Revenue" in df.columns else None



    c1, c2, c3 = st.columns(3)

    c1.metric("ACTUAL SALES", money(actual_sales))

    c2.metric("EXPECTED SALES", money(expected_sales) if expected_sales is not None else "—")

    c3.metric("EXPECTED CHANGE", money(expected_sales - actual_sales) if expected_sales is not None else "—")



    if "Invoice_Date" in df.columns and "Revenue" in df.columns:

        temp = df.copy()

        temp["Invoice_Date"] = pd.to_datetime(temp["Invoice_Date"], errors="coerce")

        temp = temp.dropna(subset=["Invoice_Date"])

        if not temp.empty:

            monthly = temp.assign(Month=temp["Invoice_Date"].dt.to_period("M").astype(str)).groupby("Month")["Revenue"].sum()

            st.markdown('<div class="page-title">Monthly Sales</div>', unsafe_allow_html=True)

            st.line_chart(monthly, height=320)



    if "Predicted_Revenue" in df.columns and "Brand" in df.columns:

        st.markdown('<div class="page-title">Expected Sales by Brand</div>', unsafe_allow_html=True)

        brand_forecast = df.groupby("Brand")["Predicted_Revenue"].sum().sort_values(ascending=False).head(10)

        st.bar_chart(brand_forecast, height=320)



        st.markdown('<div class="page-title">Forecast Details</div>', unsafe_allow_html=True)

        cols = unique_columns([c for c in [get_product_column(df), "Brand", "Category", "Revenue", "Predicted_Revenue"] if c and c in df.columns])

        st.dataframe(df[cols].head(100), use_container_width=True, hide_index=True)

    else:

        st.info("Sales forecast is not available for this dataset.")



# ------------------------------------------------------------
# PRODUCT INTELLIGENCE (COMBOS & SEGMENTATION)
# ------------------------------------------------------------
elif st.session_state.page in ["Product Intelligence", "Product Combos", "Product Groups"]:
    df = require_data()
    page_header("Product Intelligence", "Cross-sell association recommendations and algorithmic product segmentation.")

    p_tab1, p_tab2 = st.tabs(["🛒 Cross-Sell Combos (Market Basket)", "👥 Product & Customer Segments (K-Means)"])

    with p_tab1:
        using_real_rules = False
        if models and "apriori" in models and not models["apriori"].empty:
            apriori_rules = models["apriori"]
            using_real_rules = True
        else:
            apriori_rules = build_sample_transaction_rules()

        if apriori_rules.empty:
            st.info("No strong product combinations were found.")
        else:
            products = sorted(apriori_rules["Product"].dropna().astype(str).unique().tolist())
            selected_product = st.selectbox("Select a product to view frequently bought together items", products, key="pi_combo_select")
            associations = get_associations(selected_product, apriori_rules, min_lift=1.0)

            if associations is None or len(associations) == 0:
                st.info("No strong product combinations were found for this product.")
            else:
                source_label = (
                    "pre-trained association rules from transaction data"
                    if using_real_rules
                    else "a generated market basket pattern"
                )
                st.markdown(
                    f'<div class="success-box"><b>Recommended combinations for {selected_product}</b><br>'
                    f'These recommendations are based on {source_label} using association-rule analysis (Lift > 1.0).</div>',
                    unsafe_allow_html=True,
                )
                st.dataframe(associations, use_container_width=True, hide_index=True)

    with p_tab2:
        cluster_col = "KMeans_Cluster" if "KMeans_Cluster" in df.columns else ("Cluster" if "Cluster" in df.columns else None)
        if cluster_col is None:
            st.info("Product grouping is not available for this dataset.")
        else:
            group_counts = df[cluster_col].value_counts().sort_index()
            c1, c2 = st.columns(2)
            c1.metric("PRODUCT GROUPS", f"{df[cluster_col].nunique():,}")
            c2.metric("PRODUCT RECORDS", f"{len(df):,}")

            st.markdown('<div class="page-title">Products in Each Group</div>', unsafe_allow_html=True)
            st.bar_chart(group_counts, height=280)

            if "Revenue" in df.columns:
                st.markdown('<div class="page-title">Sales by Product Group</div>', unsafe_allow_html=True)
                group_revenue = df.groupby(cluster_col)["Revenue"].sum().sort_values(ascending=False)
                st.bar_chart(group_revenue, height=280)

            st.markdown('<div class="page-title">Group Summary Details</div>', unsafe_allow_html=True)
            summary = df.groupby(cluster_col).size().reset_index(name="Records")
            if "Revenue" in df.columns:
                summary = summary.merge(df.groupby(cluster_col)["Revenue"].sum().reset_index(name="Total Revenue"), on=cluster_col, how="left")
                summary["Revenue Share %"] = (summary["Total Revenue"] / summary["Total Revenue"].sum() * 100).round(1)
            st.dataframe(summary, use_container_width=True, hide_index=True)

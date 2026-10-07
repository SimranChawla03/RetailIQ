import re
import pandas as pd


# ============================================================
# COLUMN ALIASES
# ============================================================

COLUMN_ALIASES = {

    "Invoice_ID": [
        "invoice_id",
        "invoiceid",
        "invoice",
        "transaction_id",
        "transactionid",
        "bill_id",
        "bill_no",
        "order_id",
        "id"
    ],

    "Invoice_Date": [
        "invoice_date",
        "invoicedate",
        "date",
        "sale_date",
        "sales_date",
        "transaction_date",
        "order_date",
        "purchase_date"
    ],

    "City": [
        "city",
        "location",
        "store_city",
        "town"
    ],

    "Store_Format": [
        "store_format",
        "store_type",
        "store",
        "shop_type",
        "format"
    ],

    "Category": [
        "category",
        "product_category",
        "product_type",
        "item_category",
        "department",
        "segment"
    ],

    "Brand": [
        "brand",
        "product",
        "product_name",
        "productname",
        "item",
        "item_name",
        "itemname",
        "sku_name"
    ],

    "Channel": [
        "channel",
        "sales_channel",
        "order_channel",
        "purchase_channel"
    ],

    "Payment_Mode": [
        "payment_mode",
        "payment_method",
        "payment_type",
        "payment",
        "mode_of_payment"
    ],

    "Units": [
        "units",
        "unit",
        "quantity",
        "qty",
        "qty_sold",
        "quantity_sold",
        "sales_qty",
        "sold_quantity",
        "units_sold",
        "volume"
    ],

    "Cost_Price": [
        "cost_price",
        "cost",
        "purchase_price",
        "buying_price",
        "unit_cost",
        "cost_per_unit"
    ],

    "Selling_Price": [
        "selling_price",
        "sale_price",
        "sellingprice",
        "price",
        "unit_price",
        "retail_price",
        "selling_rate",
        "sales_price"
    ],

    "Revenue": [
        "revenue",
        "sales",
        "total_sales",
        "sales_value",
        "sales_amount",
        "total_revenue",
        "turnover",
        "amount",
        "total_amount"
    ],

    "Stock_On_Hand": [
        "stock_on_hand",
        "stock",
        "inventory",
        "current_stock",
        "available_stock",
        "stock_quantity",
        "quantity_in_stock",
        "inventory_level"
    ],

    "Reorder_Level": [
        "reorder_level",
        "reorder_point",
        "reorder_threshold",
        "minimum_stock",
        "min_stock",
        "minimum_inventory",
        "reorder_qty"
    ],

    "Lead_Time_Days": [
        "lead_time_days",
        "lead_time",
        "delivery_days",
        "supplier_lead_time",
        "shipping_days",
        "delivery_time"
    ],

    "Customer_Age": [
        "customer_age",
        "age",
        "buyer_age",
        "customerage"
    ],

    "Customer_Gender": [
        "customer_gender",
        "gender",
        "sex",
        "buyer_gender"
    ],

    "Loyalty_Flag": [
        "loyalty_flag",
        "loyalty",
        "loyalty_member",
        "member_flag",
        "membership",
        "is_member",
        "customer_loyalty"
    ],

    "Discount": [
        "discount",
        "discount_rate",
        "discount_percent",
        "discount_percentage",
        "offer"
    ]
}


# ============================================================
# NORMALIZE COLUMN NAME
# ============================================================

def normalize_column_name(column):

    column = str(column).strip().lower()

    column = re.sub(
        r"[^a-z0-9]+",
        "_",
        column
    )

    column = re.sub(
        r"_+",
        "_",
        column
    )

    return column.strip("_")


# ============================================================
# BUILD ALIAS LOOKUP
# ============================================================

def build_alias_lookup():

    lookup = {}

    for standard_name, aliases in COLUMN_ALIASES.items():

        lookup[
            normalize_column_name(standard_name)
        ] = standard_name

        for alias in aliases:

            lookup[
                normalize_column_name(alias)
            ] = standard_name

    return lookup


ALIAS_LOOKUP = build_alias_lookup()


# ============================================================
# STANDARDIZE COLUMN NAMES
# ============================================================

def standardize_columns(df):

    df = df.copy()

    rename_map = {}
    used_standard_names = set()

    for original_column in df.columns:

        normalized = normalize_column_name(
            original_column
        )

        standard_name = ALIAS_LOOKUP.get(
            normalized
        )

        if standard_name is None:
            continue

        # Prevent two different source columns
        # from becoming the same standard column.
        if standard_name in used_standard_names:
            continue

        rename_map[original_column] = standard_name
        used_standard_names.add(standard_name)

    df = df.rename(
        columns=rename_map
    )

    return df, rename_map


# ============================================================
# DERIVE FEATURES
# ============================================================

def create_derived_features(df):

    df = df.copy()

    # --------------------------------------------------------
    # REVENUE
    # --------------------------------------------------------

    if (
        "Revenue" not in df.columns
        and
        "Units" in df.columns
        and
        "Selling_Price" in df.columns
    ):

        units = pd.to_numeric(
            df["Units"],
            errors="coerce"
        )

        price = pd.to_numeric(
            df["Selling_Price"],
            errors="coerce"
        )

        df["Revenue"] = units * price

    # --------------------------------------------------------
    # MARGIN
    # --------------------------------------------------------

    if (
        "Margin" not in df.columns
        and
        "Selling_Price" in df.columns
        and
        "Cost_Price" in df.columns
    ):

        selling_price = pd.to_numeric(
            df["Selling_Price"],
            errors="coerce"
        )

        cost_price = pd.to_numeric(
            df["Cost_Price"],
            errors="coerce"
        )

        df["Margin"] = (
            selling_price - cost_price
        )

    # --------------------------------------------------------
    # DATE FEATURES
    # --------------------------------------------------------

    if "Invoice_Date" in df.columns:

        df["Invoice_Date"] = pd.to_datetime(
            df["Invoice_Date"],
            errors="coerce"
        )

        df["Month"] = (
            df["Invoice_Date"].dt.month
        )

        df["Day_of_Week"] = (
            df["Invoice_Date"].dt.dayofweek
        )

        df["Year"] = (
            df["Invoice_Date"].dt.year
        )

    return df


# ============================================================
# PREPROCESS DATA
# ============================================================

def preprocess_data(df):

    df = df.copy()

    # Standardize different CSV column names
    df, _ = standardize_columns(df)

    # Create derived features
    df = create_derived_features(df)

    # --------------------------------------------------------
    # CUSTOMER AGE
    # --------------------------------------------------------

    if "Customer_Age" in df.columns:

        df["Customer_Age"] = pd.to_numeric(
            df["Customer_Age"],
            errors="coerce"
        )

        median_age = df["Customer_Age"].median()

        if pd.isna(median_age):
            median_age = 0

        df["Customer_Age"] = (
            df["Customer_Age"]
            .fillna(median_age)
        )

    # --------------------------------------------------------
    # CUSTOMER GENDER
    # --------------------------------------------------------

    if "Customer_Gender" in df.columns:

        mode = df["Customer_Gender"].mode()

        if not mode.empty:

            df["Customer_Gender"] = (
                df["Customer_Gender"]
                .fillna(mode.iloc[0])
            )

    # --------------------------------------------------------
    # NUMERIC COLUMNS
    # --------------------------------------------------------

    numeric_columns = [

        "Units",
        "Cost_Price",
        "Selling_Price",
        "Revenue",
        "Margin",
        "Stock_On_Hand",
        "Reorder_Level",
        "Lead_Time_Days",
        "Customer_Age",
        "Loyalty_Flag",
        "Discount"

    ]

    for column in numeric_columns:

        if column in df.columns:

            df[column] = pd.to_numeric(
                df[column],
                errors="coerce"
            )

    return df


# ============================================================
# MODULE REQUIREMENTS
# ============================================================

MODULE_REQUIREMENTS = {

    "Sales / Demand Prediction": [

        "Units",
        "Cost_Price",
        "Selling_Price",
        "Month",
        "Day_of_Week",
        "Customer_Age",
        "Loyalty_Flag"

    ],

    "Inventory Classification": [

        "Units",
        "Selling_Price",
        "Revenue",
        "Margin",
        "Lead_Time_Days",
        "Customer_Age",
        "Loyalty_Flag",
        "Month",
        "Day_of_Week"

    ],

    "Product Clustering": [

        "Units",
        "Revenue",
        "Selling_Price",
        "Margin",
        "Stock_On_Hand",
        "Reorder_Level",
        "Lead_Time_Days"

    ],

    "Restocking Recommendation": [

        "Brand",
        "Category",
        "Units",
        "Stock_On_Hand",
        "Reorder_Level",
        "Lead_Time_Days"

    ]

}


# ============================================================
# CHECK MODULE SUPPORT
# ============================================================

def get_module_support(df):

    support = {}

    for module, required_columns in (
        MODULE_REQUIREMENTS.items()
    ):

        missing = [
            column
            for column in required_columns
            if column not in df.columns
        ]

        support[module] = {

            "available": len(missing) == 0,

            "missing_columns": missing,

            "required_columns": required_columns

        }

    return support


# ============================================================
# VALIDATE CORE RETAIL DATA
# ============================================================

def validate_data(df):

    df, _ = standardize_columns(df)

    df = create_derived_features(df)

    # A compatible retail dataset should at minimum
    # contain product/category information plus
    # quantity and selling/revenue information.

    required_groups = [

        ["Brand"],
        ["Category"],
        ["Units"],
        ["Selling_Price"],
        ["Revenue"]

    ]

    missing_groups = []

    for group in required_groups:

        available = any(
            column in df.columns
            for column in group
        )

        if not available:

            missing_groups.append(
                group[0]
            )

    if missing_groups:

        return False, missing_groups

    return True, []


# ============================================================
# DATASET PROFILE
# ============================================================

def get_dataset_profile(df):

    profile = {

        "rows":
            len(df),

        "columns":
            len(df.columns),

        "brands":
            (
                df["Brand"].nunique()
                if "Brand" in df.columns
                else None
            ),

        "categories":
            (
                df["Category"].nunique()
                if "Category" in df.columns
                else None
            ),

        "revenue_available":
            "Revenue" in df.columns,

        "inventory_available":
            "Stock_On_Hand" in df.columns,

        "date_available":
            "Invoice_Date" in df.columns

    }

    return profile
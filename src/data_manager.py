"""
Centralized data-loading pipeline and persistent upload manager.
Handles:
1. Loading approved historical data on first launch automatically.
2. Managing accepted daily uploads persistently in data/uploads/.
3. Duplicate file detection via SHA-256 content hashing.
4. Duplicate record detection without dropping legitimate multi-item invoices.
5. Atomic persistence to avoid corruption.
"""
import os
import io
import json
import hashlib
import pandas as pd
from src.preprocessing import (
    standardize_columns,
    create_derived_features,
    preprocess_data,
    validate_data,
    COLUMN_ALIASES,
    normalize_column_name
)

UPLOAD_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "uploads")
HISTORICAL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "processed", "cleaned_data.csv")
MANIFEST_PATH = os.path.join(UPLOAD_DIR, "manifest.json")

def ensure_upload_dir():
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    if not os.path.exists(MANIFEST_PATH):
        with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
            json.dump([], f)

def get_upload_manifest():
    ensure_upload_dir()
    try:
        with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        raise RuntimeError(f"Could not read upload manifest at {MANIFEST_PATH}: {exc}") from exc

def save_upload_manifest(manifest):
    ensure_upload_dir()
    temp_path = MANIFEST_PATH + ".tmp"
    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    os.replace(temp_path, MANIFEST_PATH)

def compute_file_hash(file_bytes):
    return hashlib.sha256(file_bytes).hexdigest()

def is_duplicate_file(file_hash):
    manifest = get_upload_manifest()
    for entry in manifest:
        if entry.get("file_hash") == file_hash:
            return True, entry
    return False, None

def load_historical_data():
    """Loads the approved original historical dataset safely."""
    if not os.path.exists(HISTORICAL_PATH):
        raise FileNotFoundError(f"Approved historical dataset missing at: {HISTORICAL_PATH}")
    df = pd.read_csv(HISTORICAL_PATH)
    return preprocess_data(df)

def load_accepted_uploads():
    """Loads all previously accepted and confirmed uploads from persistent storage."""
    ensure_upload_dir()
    manifest = get_upload_manifest()
    frames = []
    for entry in manifest:
        stored_filename = entry.get("stored_filename")
        if not stored_filename or os.path.basename(stored_filename) != stored_filename:
            raise RuntimeError(f"Invalid stored filename in upload manifest: {stored_filename!r}")
        file_path = os.path.join(UPLOAD_DIR, stored_filename)
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"Accepted upload listed in manifest is missing: {file_path}")
        try:
            sub_df = pd.read_csv(file_path)
        except Exception as exc:
            raise RuntimeError(f"Could not read accepted upload {stored_filename}: {exc}") from exc
        if sub_df.empty:
            raise RuntimeError(f"Accepted upload contains no rows: {stored_filename}")
        frames.append(sub_df)
    if frames:
        return pd.concat(frames, ignore_index=True)
    return pd.DataFrame()

def load_reporting_data():
    """
    Central shared pipeline providing consistent reporting data to all pages.
    Combines approved historical data and persistent accepted uploads.
    """
    historical_df = load_historical_data()
    uploads_df = load_accepted_uploads()

    if uploads_df.empty:
        return historical_df

    uploads_processed = preprocess_data(uploads_df)
    combined = pd.concat([historical_df, uploads_processed], ignore_index=True)

    # Do not drop identical-looking rows: separate real sale lines can share
    # the same values. Exact duplicate files are rejected by SHA-256 instead.
    return combined.reset_index(drop=True)

def validate_upload_content(file_bytes, filename):
    """
    Validates uploaded file syntax, headers, required retail columns, and checks for duplicates.
    """
    file_hash = compute_file_hash(file_bytes)
    is_dup, dup_entry = is_duplicate_file(file_hash)
    if is_dup:
        return {
            "valid": False,
            "error": f"Duplicate file detected: This file has already been imported as '{dup_entry.get('original_filename')}'.",
            "file_hash": file_hash,
            "df": None
        }

    if not file_bytes or len(file_bytes.strip()) == 0:
        return {
            "valid": False,
            "error": "The uploaded file is empty (contains no data).",
            "file_hash": file_hash,
            "df": None
        }

    raw_df = None
    read_error = None

    if filename.lower().endswith(".xlsx"):
        try:
            raw_df = pd.read_excel(io.BytesIO(file_bytes))
        except Exception as e:
            read_error = str(e)
    else:
        # Try UTF-8, then common fallbacks (latin-1, cp1252, utf-8-sig)
        for enc in ["utf-8", "utf-8-sig", "latin-1", "cp1252"]:
            try:
                raw_df = pd.read_csv(io.BytesIO(file_bytes), encoding=enc)
                read_error = None
                break
            except Exception as e:
                read_error = str(e)

    if raw_df is None or read_error is not None:
        return {
            "valid": False,
            "error": f"Failed to parse file: {read_error}. Please ensure the file is a well-formed CSV or Excel file.",
            "file_hash": file_hash,
            "df": None
        }

    if raw_df.empty or len(raw_df) == 0:
        return {
            "valid": False,
            "error": "The uploaded file is empty (contains no data rows).",
            "file_hash": file_hash,
            "df": None
        }

    processed_df = preprocess_data(raw_df.copy())
    is_valid, missing = validate_data(processed_df)

    if not is_valid:
        return {
            "valid": False,
            "error": f"Required retail fields missing: {', '.join(missing)}. Please verify column names.",
            "file_hash": file_hash,
            "missing_columns": missing,
            "df": None
        }

    # Data validation checks: Check for invalid negative units or prices
    if "Units" in processed_df.columns:
        invalid_units = (processed_df["Units"] < 0).sum()
        if invalid_units > 0:
            return {
                "valid": False,
                "error": f"Invalid data: Found {invalid_units:,} record(s) with negative Units.",
                "file_hash": file_hash,
                "df": None
            }

    if "Selling_Price" in processed_df.columns:
        invalid_prices = (processed_df["Selling_Price"] < 0).sum()
        if invalid_prices > 0:
            return {
                "valid": False,
                "error": f"Invalid data: Found {invalid_prices:,} record(s) with negative Selling Price.",
                "file_hash": file_hash,
                "df": None
            }

    # Date validation: If a date column was provided in the upload, ensure at least one date parsed successfully
    date_alias_norms = {normalize_column_name(a) for a in COLUMN_ALIASES.get("Invoice_Date", [])} | {"invoice_date"}
    has_date_input = any(normalize_column_name(c) in date_alias_norms for c in raw_df.columns)
    if has_date_input and "Invoice_Date" in processed_df.columns:
        valid_dates = processed_df["Invoice_Date"].notna().sum()
        if valid_dates == 0 and len(processed_df) > 0:
            return {
                "valid": False,
                "error": "Invalid date format: All dates in the date column could not be parsed. Expected format YYYY-MM-DD or standard datetime.",
                "file_hash": file_hash,
                "df": None
            }

    return {
        "valid": True,
        "error": None,
        "file_hash": file_hash,
        "row_count": len(processed_df),
        "raw_df": raw_df,
        "processed_df": processed_df
    }

def persist_accepted_upload(file_bytes, original_filename, file_hash, processed_df):
    """
    Atomically writes the accepted file to persistent storage and updates the manifest.
    """
    ensure_upload_dir()

    # Re-check during commit because validation and commit occur on separate reruns.
    actual_hash = compute_file_hash(file_bytes)
    if actual_hash != file_hash:
        raise ValueError("Uploaded file changed after validation. Please upload it again.")
    is_dup, dup_entry = is_duplicate_file(file_hash)
    if is_dup:
        raise ValueError(
            "This file has already been imported as "
            f"'{dup_entry.get('original_filename', 'another file')}'."
        )

    original_filename = os.path.basename(str(original_filename))
    stored_name = f"{file_hash[:12]}_{original_filename.replace(' ', '_')}"
    if not stored_name.endswith(".csv"):
        stored_name += ".csv"
    dest_path = os.path.join(UPLOAD_DIR, stored_name)
    temp_path = dest_path + ".tmp"

    processed_df.to_csv(temp_path, index=False)
    os.replace(temp_path, dest_path)

    manifest = get_upload_manifest()
    manifest.append({
        "original_filename": original_filename,
        "stored_filename": stored_name,
        "file_hash": file_hash,
        "rows": len(processed_df),
        "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    })
    save_upload_manifest(manifest)
    return stored_name


def delete_accepted_upload(stored_filename):
    """Remove one accepted upload from the manifest and disk.

    The approved historical dataset is never affected by this function.
    """
    ensure_upload_dir()
    safe_name = os.path.basename(str(stored_filename))
    if safe_name != stored_filename or not safe_name:
        raise ValueError("Invalid stored filename.")

    manifest = get_upload_manifest()
    matches = [entry for entry in manifest if entry.get("stored_filename") == safe_name]
    if not matches:
        raise FileNotFoundError(f"Upload not found in history: {safe_name}")

    # Update manifest first so the file is no longer included in reporting.
    new_manifest = [entry for entry in manifest if entry.get("stored_filename") != safe_name]
    save_upload_manifest(new_manifest)

    file_path = os.path.join(UPLOAD_DIR, safe_name)
    if os.path.isfile(file_path):
        os.remove(file_path)

    return matches[0]

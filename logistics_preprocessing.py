"""Week 2: Logistics data cleaning and preprocessing demonstration.

The input CSV is an illustrative teaching dataset. It is not company data and
is not a downloaded copy of the Olist dataset. It uses logistics-related fields
inspired by the public Olist schema to demonstrate a reproducible workflow.
"""
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

BASE_DIR = Path(__file__).resolve().parent
INPUT_FILE = BASE_DIR / "sample_logistics_data.csv"
OUTPUT_FILE = BASE_DIR / "cleaned_logistics_data.csv"
SUMMARY_FILE = BASE_DIR / "preprocessing_summary.txt"


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Could not find {INPUT_FILE.name}. Keep it in the same folder as this script."
        )

    df = pd.read_csv(INPUT_FILE)
    initial_rows = len(df)
    initial_columns = len(df.columns)
    summary = []
    summary.append("LOGISTICS DATA PREPROCESSING SUMMARY")
    summary.append(f"Input file: {INPUT_FILE.name}")
    summary.append(f"Initial shape: {initial_rows} rows x {initial_columns} columns")

    # 1. Standardise text: remove surrounding spaces and make status consistent.
    for col in ["order_status", "customer_state"]:
        df[col] = df[col].astype("string").str.strip()
    df["order_status"] = df["order_status"].str.lower()
    df["customer_state"] = df["customer_state"].str.upper()
    df["customer_state"] = df["customer_state"].replace({"": pd.NA, "NAN": pd.NA})

    # 2. Parse dates safely. Invalid strings become missing rather than crashing.
    date_cols = ["purchase_timestamp", "delivered_timestamp", "estimated_delivery_timestamp"]
    for col in date_cols:
        df[col] = pd.to_datetime(df[col], errors="coerce")

    # 3. Convert numeric fields and mark physically invalid negative values as missing.
    numeric_cols = ["distance_km", "shipping_cost_brl", "package_weight_kg", "order_value_brl", "review_score"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    invalid_rules = {
        "distance_km": df["distance_km"] <= 0,
        "shipping_cost_brl": df["shipping_cost_brl"] < 0,
        "package_weight_kg": df["package_weight_kg"] <= 0,
        "order_value_brl": df["order_value_brl"] < 0,
        "review_score": ~df["review_score"].between(1, 5),
    }
    invalid_count = 0
    for col, mask in invalid_rules.items():
        invalid_count += int(mask.sum())
        df.loc[mask, col] = np.nan

    # 4. Remove exact duplicate order records using the unique order identifier.
    before_dedup = len(df)
    df = df.drop_duplicates(subset="order_id", keep="first").copy()
    duplicates_removed = before_dedup - len(df)

    # 5. Derive delivery duration. Keep it missing when an order is not delivered
    # or its timestamps are incomplete/chronologically invalid.
    df["delivery_days"] = (df["delivered_timestamp"] - df["purchase_timestamp"]).dt.total_seconds() / 86400
    invalid_duration = df["delivery_days"] < 0
    invalid_duration_count = int(invalid_duration.sum())
    df.loc[invalid_duration, "delivery_days"] = np.nan
    df.loc[df["order_status"] != "delivered", "delivery_days"] = np.nan

    # 6. Fill missing numeric operational attributes with column medians.
    # Review scores are not imputed because an absent review is not a data error.
    impute_cols = ["distance_km", "shipping_cost_brl", "package_weight_kg", "order_value_brl"]
    missing_filled = {}
    for col in impute_cols:
        missing_before = int(df[col].isna().sum())
        median_value = df[col].median()
        df[col] = df[col].fillna(median_value)
        missing_filled[col] = missing_before
    # Fill missing customer state with an explicit category, preserving the fact
    # that it was not supplied rather than guessing a location.
    state_missing = int(df["customer_state"].isna().sum())
    df["customer_state"] = df["customer_state"].fillna("UNKNOWN")
    status_missing = int(df["order_status"].isna().sum())
    df["order_status"] = df["order_status"].fillna("unknown")

    # 7. Detect outliers with the IQR rule and cap extreme values at the fences.
    # This limits the influence of extreme records while retaining each order.
    outlier_cols = ["distance_km", "shipping_cost_brl", "package_weight_kg", "order_value_brl"]
    outlier_report = {}
    for col in outlier_cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower = max(0, q1 - 1.5 * iqr)
        upper = q3 + 1.5 * iqr
        count = int(((df[col] < lower) | (df[col] > upper)).sum())
        df[col] = df[col].clip(lower=lower, upper=upper)
        outlier_report[col] = (count, float(lower), float(upper))

    # 8. Min-max normalise selected numeric variables to the 0-to-1 range.
    scale_cols = ["distance_km", "shipping_cost_brl", "package_weight_kg", "order_value_brl"]
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df[scale_cols])
    scaled_cols = [f"{c}_normalised" for c in scale_cols]
    df[scaled_cols] = scaled

    # 9. Export the prepared data and a readable audit summary.
    df.to_csv(OUTPUT_FILE, index=False, date_format="%Y-%m-%d %H:%M:%S")
    summary.append(f"Exact duplicate rows removed: {duplicates_removed}")
    summary.append(f"Invalid numeric values converted to missing: {invalid_count}")
    summary.append(f"Negative delivery durations set to missing: {invalid_duration_count}")
    summary.append(f"Missing order_status entries filled as unknown: {status_missing}")
    summary.append(f"Missing customer_state entries labelled UNKNOWN: {state_missing}")
    summary.append("Missing values filled with medians:")
    for col, count in missing_filled.items():
        summary.append(f"  - {col}: {count}")
    summary.append("IQR outlier checks (count, lower fence, upper fence):")
    for col, values in outlier_report.items():
        summary.append(f"  - {col}: {values[0]}, {values[1]:.3f}, {values[2]:.3f}")
    summary.append(f"Final shape: {len(df)} rows x {len(df.columns)} columns")
    summary.append(f"Cleaned output: {OUTPUT_FILE.name}")
    summary.append(f"Remaining missing review scores: {int(df['review_score'].isna().sum())} (left missing because no review is a valid state)")
    summary.append(f"Remaining missing delivery_days: {int(df['delivery_days'].isna().sum())} (not delivered/incomplete/invalid timestamps)")
    SUMMARY_FILE.write_text("\n".join(summary), encoding="utf-8")
    print("\n".join(summary))


if __name__ == "__main__":
    main()

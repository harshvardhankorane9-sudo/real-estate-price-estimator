import re
from typing import Tuple
import numpy as np
import pandas as pd

def _parse_total_sqft(x) -> float:
    """Convert total_sqft strings to float (handles ranges and units)."""
    if pd.isna(x):
        return np.nan
    if isinstance(x, (int, float)):
        return float(x)

    s = str(x).strip().replace(',', '')

    # Range like "2100 - 2850"
    if '-' in s:
        parts = s.split('-')
        try:
            return (float(parts[0]) + float(parts[1])) / 2.0
        except:
            return np.nan

    # Unit-based values like "34.46Sq. Meter"
    unit_map = {
        "sq. meter": 10.7639,
        "sq. meters": 10.7639,
        "sq. yard": 9.0,
        "sq. yards": 9.0,
        "acre": 43560.0,
        "acres": 43560.0,
        "cents": 435.6,
        "guntha": 1089.0,
        "perch": 272.25,
        "ground": 2400.0,
        "grounds": 2400.0,
        "sq. ft": 1.0,
        "sqft": 1.0,
    }

    s_lower = s.lower()
    for unit, factor in unit_map.items():
        if unit in s_lower:
            num_str = re.sub(r"[^0-9.\-]", "", s)
            try:
                num = float(num_str)
                return num * factor
            except:
                return np.nan

    # Plain numeric string
    try:
        return float(s)
    except:
        return np.nan

def _remove_price_per_sqft_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Remove extreme outliers based on price_per_sqft per location."""
    def filter_group(g):
        m = g.price_per_sqft.mean()
        s = g.price_per_sqft.std()
        if pd.isna(s) or s == 0:
            return g
        return g[(g.price_per_sqft > (m - s)) & (g.price_per_sqft < (m + s))]
    return df.groupby('location', group_keys=False).apply(filter_group)

def clean_and_engineer_features(df: pd.DataFrame, min_loc_count: int = 10) -> pd.DataFrame:
    """Clean and engineer features for modeling."""
    df = df.copy()

    # Extract BHK from size column
    df["bhk"] = pd.to_numeric(df["size"].str.extract(r"(\d+)")[0], errors="coerce")

    # Convert total_sqft
    df["total_sqft"] = df["total_sqft"].apply(_parse_total_sqft)

    # Drop rows with nulls
    df = df.dropna(subset=["total_sqft", "bhk", "price", "location", "area_type", "bath"])

    # Clean and bin location
    df["location"] = df["location"].str.strip()
    df["location"] = df["location"].replace("", np.nan).fillna("unknown")
    loc_counts = df["location"].value_counts()
    rare_locs = loc_counts[loc_counts < min_loc_count].index
    df.loc[df["location"].isin(rare_locs), "location"] = "other"

    # Calculate price per sqft
    df["price_per_sqft"] = (df["price"] * 100000) / df["total_sqft"]

    # Further filtering
    df = df[df["total_sqft"] / df["bhk"] >= 300]
    df = _remove_price_per_sqft_outliers(df)
    df = df[df["bath"] <= df["bhk"] + 2]
    return df[["total_sqft", "bath", "bhk", "area_type", "location", "price"]].reset_index(drop=True)

def split_X_y(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Split DataFrame into features and target."""
    X = df.drop(columns=["price"])
    y = df["price"]
    return X, y

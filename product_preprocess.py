import pandas as pd
import os
import re

INPUT_XLSX = "data/raw/online_retail_II.xlsx"
SHEETS = ["Year 2009-2010", "Year 2010-2011"]

OUT_XLSX = "data/online_retail_II_cleaned.xlsx"
OUT_DIR_PARQUET = "parquet_cleaned"
os.makedirs(OUT_DIR_PARQUET, exist_ok=True)

def normalize_desc(s):
    if pd.isna(s): 
        return ""
    s = str(s).strip()
    s = re.sub(r"\s+", " ", s)
    return s

def is_blank(series: pd.Series) -> pd.Series:
    return series.isna() | (series.astype("string").str.strip() == "")

pairs = []
for sh in SHEETS:
    d = pd.read_excel(INPUT_XLSX, sheet_name=sh, engine="openpyxl", usecols=["StockCode", "Description"])
    d["StockCode"] = d["StockCode"].astype("string")
    d["Description"] = d["Description"].map(normalize_desc)
    d = d.loc[~is_blank(d["Description"]), ["StockCode", "Description"]]
    pairs.append(d)

all_pairs = pd.concat(pairs, ignore_index=True)

stock2desc = all_pairs.drop_duplicates(subset=["StockCode"], keep="first").set_index("StockCode")["Description"]

with pd.ExcelWriter(OUT_XLSX, engine="openpyxl") as writer:
    for sh in SHEETS:
        df = pd.read_excel(INPUT_XLSX, sheet_name=sh, engine="openpyxl")

        df["StockCode"] = df["StockCode"].astype("string")
        df["Description"] = df["Description"].map(normalize_desc)

        blank = is_blank(df["Description"])

        df.loc[blank, "Description"] = df.loc[blank, "StockCode"].map(stock2desc).fillna("")

        to_drop = is_blank(df["Description"])

        before = len(df)
        filled = int(blank.sum()) - int(to_drop.sum()) 
        dropped = int(to_drop.sum())

        df = df.loc[~to_drop].reset_index(drop=True)

        print(f"[{sh}] rows before={before}, filled_desc~={filled}, dropped_rows={dropped}, rows after={len(df)}")

        df.to_excel(writer, sheet_name=sh, index=False)

print(f"\nSaved cleaned xlsx: {OUT_XLSX}")
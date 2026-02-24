import pandas as pd
import os
import re

p1 = "data/Year_2009-2010.parquet"
p2 = "data/Year_2010-2011.parquet"

out_dir = "data"
os.makedirs(out_dir, exist_ok=True)

df1 = pd.read_parquet(p1)
df2 = pd.read_parquet(p2)

def norm_family_no_space(x: str) -> str:
    if pd.isna(x):
        return ""
    s = str(x).strip().upper()
    s = re.sub(r"\s+", "", s)
    return s

def norm_desc(x: str) -> str:
    if pd.isna(x):
        return ""
    s = str(x).strip()
    s = re.sub(r"\s+", " ", s)
    return s

for df in (df1, df2):
    df["StockCode"] = df["StockCode"].astype("string").str.strip()
    df["Description"] = df["Description"].map(norm_desc)
    df["product_family_name_norm"] = df["product_family_name"].map(norm_family_no_space)

all_df = pd.concat([df1[["StockCode", "product_family_name_norm"]],
                    df2[["StockCode", "product_family_name_norm"]]], ignore_index=True)

family_stock_n = all_df.groupby("product_family_name_norm")["StockCode"].nunique()

keep_fams = set(family_stock_n[family_stock_n >= 2].index)

def apply_rule(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    keep = df["product_family_name_norm"].isin(keep_fams)

    df.loc[keep, "product_family_name"] = df.loc[keep, "product_family_name_norm"]

    df.loc[~keep, "product_family_name"] = df.loc[~keep, "Description"]

    df = df.drop(columns=["product_family_name_norm"])
    return df

df1_out = apply_rule(df1)
df2_out = apply_rule(df2)

out1 = os.path.join(out_dir, "Year_2009-2010_post.parquet")
out2 = os.path.join(out_dir, "Year_2010-2011_post.parquet")

df1_out.to_parquet(out1, index=False)
df2_out.to_parquet(out2, index=False)

print("Saved:", out1)
print("Saved:", out2)

# u1 = df1_out["product_family_name"].replace("", pd.NA).dropna().nunique()
# u2 = df2_out["product_family_name"].replace("", pd.NA).dropna().nunique()

# print("df1_out unique product_family_name:", u1)
# print("df2_out unique product_family_name:", u2)

print("\n=== Sanity checks ===")
print(df1_out.head(100).to_string(index=False))

# print("kept families (>=2 stockcodes):", len(keep_fams))
# kept_rows_2009 = df1["product_family_name_norm"].isin(keep_fams).sum()
# kept_rows_2010 = df2["product_family_name_norm"].isin(keep_fams).sum()
# print("kept rows 2009-2010:", kept_rows_2009, "/", len(df1))
# print("kept rows 2010-2011:", kept_rows_2010, "/", len(df2))
# print(df1_out.columns.tolist())
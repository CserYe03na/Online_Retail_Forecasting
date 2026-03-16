import pandas as pd
import os

os.makedirs("data/forecasting", exist_ok=True)

train_df = pd.read_parquet("data/clustering/train_df.parquet")
test_df = pd.read_parquet("data/clustering/test_df.parquet")

clusters = pd.read_parquet("data/clustering/clusters_3models.parquet")
print(clusters.head())

clusters = clusters.reset_index()
cluster_map = clusters[['product_family_name', 'cluster_kmeans']].drop_duplicates()
train_df = train_df.merge(cluster_map, on='product_family_name', how='left')
test_df  = test_df.merge(cluster_map, on='product_family_name', how='left')

print(train_df.head())

# extract day
for df in [train_df, test_df]:
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    df['date'] = df['InvoiceDate'].dt.floor('D')

train_daily = (
    train_df
    .groupby(['date', 'product_family_name', 'cluster_kmeans'], as_index=False)['sales_amount']
    .sum()
    .rename(columns={'cluster_kmeans': 'cluster', 'sales_amount': 'total_sales'})
)

test_daily = (
    test_df
    .groupby(['date', 'product_family_name', 'cluster_kmeans'], as_index=False)['sales_amount']
    .sum()
    .rename(columns={'cluster_kmeans': 'cluster', 'sales_amount': 'total_sales'})
)

# save
train_daily.to_parquet("data/forecasting/train_daily.parquet", index=False)
test_daily.to_parquet("data/forecasting/test_daily.parquet", index=False)


print("Saved:")
print(" - data/forecasting/train_daily.parquet")
print(" - data/forecasting/test_daily.parquet")

print("\ntrain_daily preview:")
print(train_daily.head())


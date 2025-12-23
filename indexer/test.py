import polars as pl


df = pl.read_parquet("image_embeddings.parquet")

print(df)

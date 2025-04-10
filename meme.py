# import pandas as pd

# test = pd.read_csv("sitc_country_country_product_year_4_2020_2023.csv")
# test['product_sitc_code'] = test['product_sitc_code'].astype(str)

# test.to_parquet("sitc_country_country_product_year_4_2020_2023.parquet", engine='pyarrow')

import pyarrow.dataset as ds
import pyarrow as pa
import numpy as np

dataset = ds.dataset("/root/eel-6878-final-project/sitc_country_country_product_year_4_1962_1969.parquet", format="parquet")
table = dataset.to_table()
indices = np.random.choice(table.num_rows, 50000, replace=False)
sample = table.take(pa.array(indices))
df = sample.to_pandas()

print(df)
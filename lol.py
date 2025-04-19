import pandas as pd
import glob
import joblib

files = glob.glob("sitc_country_country_product_year_4_*.parquet")
num_years = 2023-1962+1
all_files = [None] * len(files)
c = 0
wanted_cols = ['country_iso3_code', 'partner_iso3_code', 'product_sitc_code', 'year', 'export_value', 'import_value']

for f in files:
    all_files[c] = pd.read_parquet(f)
    all_files[c] = all_files[c][wanted_cols]
    c = c + 1

### Creation of the product specific import and export dataset, from the original dataset
imex_ps = [None] * num_years
c = 0
for i in range(num_years):
    year = 1962 + i # Index 0 for all the lists created will be at year 1962, every index after that is 1962 + i
    if(year%10 == 0):
        c = c + 1 # Changing the parquet file read from based on year
    imex_ps[i] = all_files[c].loc[all_files[c]['year'] == year]
    imex_ps[i] = imex_ps[i].drop('year', axis=1)

### Creation of a general import and export dataset, not product specific, from the previous dataset
#imex_tt = [None] * num_years
#agg = {'country_iso3_code':'first', 'partner_iso3_code':'first', 'export_value':'sum', 'import_value':'sum'}
#for i in range(num_years):
#    imex_tt[i] = imex_ps[i].groupby(['country_iso3_code', 'partner_iso3_code']).aggregate(agg)

### Creation of a general total trade volume dataset from the previous dataset
#tv_tt = [None] * num_years
#tv_tt = imex_tt
#for i in  range(num_years):
#    tv_tt[i]['trade_volume'] = tv_tt[i]['import_value'] + tv_tt[i]['export_value']

joblib.dump(imex_ps, 'product_specific_import_export_values.joblib')
#joblib.dump(imex_tt, 'total_country_trade_import_export_values.joblib')
#joblib.dump(tv_tt, 'total_country_trade_volumes.joblib')

#test = joblib.load('product_specific_import_export_values.joblib')
#print(test[0]) #index [0] is 1962
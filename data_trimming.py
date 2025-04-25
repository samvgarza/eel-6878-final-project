import pandas as pd
import glob
import os

def save_trim_data(df_merged, start_year, end_year):
    #Saves data as a parquet file, name is based on the year range of the data
    if not os.path.exists("trim"):
        os.makedirs("trim")
    df_merged.to_parquet("trim/sitc_country_country_product_year_4_"+str(start_year)+"_"+str(end_year)+".parquet", engine='pyarrow')

def trim_data(df_merged, start_year, end_year, save_to_file, return_as_list):
    #Trims data based on what common tags there are in the country and product columns, in the range provided
    earliest_year = df_merged['year'].min()
    target_year = earliest_year
    latest_year = df_merged['year'].max()
    max_year = latest_year
    #Sets a max year the dataset will trim
    if (max_year > end_year):
        max_year = end_year
    #Sets a min year the dataset will trim
    if (target_year < start_year):
        target_year = start_year

    #Preparing a list of dataframes based on year, within the desired range
    data_years = latest_year - earliest_year + 1
    num_years = max_year - target_year + 1
    df_list = [None] * num_years
    c = 0
    for i in range(data_years):
        year = earliest_year + i
        if((year >= target_year) and (year <= max_year)):
            df_list[c] = df_merged[df_merged['year']==(earliest_year+i)]
            c = c + 1
    
    #Initialization of the common tags used to trim the data
    #Variable common_countries appears twice because we check both country and partner iso3 codes for uniqueness
    common_countries = set(df_list[0]['country_iso3_code'].unique())
    common_countries = set(df_list[0]['partner_iso3_code'].unique())
    common_products = set(df_list[0]['product_sitc_code'].unique())

    #Loop to find all common tags in the datasets
    for df in df_list:
        common_countries &= set(df['country_iso3_code'].unique())
        common_countries &= set(df['partner_iso3_code'].unique())
        common_products &= set(df['product_sitc_code'].unique())

    #print("Number of common countries in each year: ", len(common_countries))
    #print("Common Countries: ")
    #print(sorted(common_countries))
    #print("Number of common products in each year: ", len(common_products))
    #print("Common Products: ")
    #print(sorted(common_products))
    
    #Trimming the data based on common tags for countries and products
    for i in range(len(df_list)):
        df_list[i] = df_list[i][df_list[i]['country_iso3_code'].isin(common_countries)]
        df_list[i] = df_list[i][df_list[i]['partner_iso3_code'].isin(common_countries)]
        df_list[i] = df_list[i][df_list[i]['product_sitc_code'].isin(common_products)]

    #If desired, can be saved as a parquet file
    if save_to_file:
        save_trim_data(pd.concat(df_list, ignore_index=True), target_year, max_year)

    #If desired, can be returned as a list of dataframes, or one big dataframe
    if return_as_list:
        return df_list
    else:
        return pd.concat(df_list, ignore_index=True)

def gather_data():
    #Gathers all dataset files and merges it into one big dataframe
    files = glob.glob("sitc_country_country_product_year_4_*.parquet")
    all_files = [None] * len(files)
    c = 0
    wanted_cols = ['country_iso3_code', 'partner_iso3_code', 'product_sitc_code', 'year', 'export_value', 'import_value']
    for f in files:
        all_files[c] = pd.read_parquet(f)
        all_files[c]['product_sitc_code'] = all_files[c]['product_sitc_code'].astype(str)
        all_files[c] = all_files[c][wanted_cols]
        c = c + 1

    return pd.concat(all_files, ignore_index=True)

def main():
    start_year = 1962
    end_year = 1969
    save_to_file = True
    return_as_list = False
    print(trim_data(gather_data(), start_year, end_year, save_to_file, return_as_list))

if __name__ == "__main__":
    main()
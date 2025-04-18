import pandas as pd
import numpy as np
import torch
from torch_geometric.data import HeteroData

def build_yearly_graphs(df, years):
    """Build heterogeneous graphs for each year in the dataset with consistent node numbering."""
    # First pass: Collect all unique nodes across all years
    all_countries = set()
    all_products = set()

    for year in years:
        year_data = df[df['year'] == year]
        all_countries.update(year_data['country_id'].unique())
        all_countries.update(year_data['partner_country_id'].unique())
        all_products.update(year_data['product_id'].unique())

    # Create consistent mappings for all years
    country_mapping = {id_: i for i, id_ in enumerate(sorted(all_countries))}
    product_mapping = {id_: i for i, id_ in enumerate(sorted(all_products))}

    yearly_graphs = {}
    num_countries = len(country_mapping)
    num_products = len(product_mapping)

    for year in years:
        # Filter data for the current year
        year_data = df[df['year'] == year]

        # Create empty heterogeneous graph
        graph = HeteroData()

        # Add all country nodes (including inactive ones this year)
        graph['country'].x = torch.zeros((num_countries, num_countries))  # Start with zeros
        active_countries = pd.concat([
            year_data[['country_id']].drop_duplicates().rename(columns={'country_id': 'id'}),
            year_data[['partner_country_id']].drop_duplicates().rename(columns={'partner_country_id': 'id'})
        ]).drop_duplicates()

        # Set one-hot encoding only for active countries
        for country_id in active_countries['id']:
            idx = country_mapping[country_id]
            graph['country'].x[idx, idx] = 1  # Set diagonal for active countries

        # Add all product nodes (including inactive ones this year)
        graph['product'].x = torch.zeros((num_products, num_products))  # Start with zeros
        active_products = year_data[['product_id']].drop_duplicates()

        # Set one-hot encoding only for active products
        for product_id in active_products['product_id']:
            idx = product_mapping[product_id]
            graph['product'].x[idx, idx] = 1  # Set diagonal for active products

        # Add export edges
        exports = year_data[year_data['export_value'] > 0]
        if len(exports) > 0:
            src = [country_mapping[id_] for id_ in exports['country_id']]
            dst = [product_mapping[id_] for id_ in exports['product_id']]

            graph['country', 'exports', 'product'].edge_index = torch.tensor([src, dst])
            edge_values = exports['export_value'].values
            edge_values = np.log1p(edge_values)
            edge_values = (edge_values - edge_values.mean()) / edge_values.std()  # Normalize
            graph['country', 'exports', 'product'].edge_attr = torch.tensor(edge_values, dtype=torch.float).reshape(-1, 1)

        # Add import edges
        imports = year_data[year_data['import_value'] > 0]
        if len(imports) > 0:
            src = [product_mapping[id_] for id_ in imports['product_id']]
            dst = [country_mapping[id_] for id_ in imports['country_id']]

            graph['product', 'imports', 'country'].edge_index = torch.tensor([src, dst])
            edge_values = imports['import_value'].values
            edge_values = np.log1p(edge_values)
            edge_values = (edge_values - edge_values.mean()) / edge_values.std()  # Normalize
            graph['country', 'imports', 'product'].edge_attr = torch.tensor(edge_values, dtype=torch.float).reshape(-1, 1)

        yearly_graphs[year] = graph

    return yearly_graphs, country_mapping, product_mapping
import pandas as pd
import numpy as np
import torch
from torch_geometric.data import HeteroData
from torch.utils.data import Dataset

class TemporalGraphDataset(Dataset):
    def __init__(self, df, years, window_size=5):
        """initialize dataset with trade data"""

        # store years in sorted order
        self.years = sorted(years)

        # set window size for temporal sequences
        self.window_size = window_size

        # build mappings between ids and indices
        self.build_mappings(df)

        # create master graph with all nodes
        self.master_graph = self.build_master_graph()

        # build edges for each year
        self.year_to_edges = self.build_temporal_edges(df)

        # generate training samples
        self.samples = self.generate_samples()

    def build_mappings(self, df):
        """create mappings between country/product ids and indices"""

        # get all unique country ids
        all_countries = set(df['country_id'].unique())
        all_partner_countries = set(df['partner_country_id'].unique())
        all_countries = all_countries.union(all_partner_countries)

        # get all unique product ids
        all_products = set(df['product_id'].unique())

        # create mapping from id to index
        self.country_to_idx = {}
        country_list = sorted(all_countries)
        for i, country_id in enumerate(country_list):
            self.country_to_idx[country_id] = i

        self.product_to_idx = {}
        product_list = sorted(all_products)
        for i, product_id in enumerate(product_list):
            self.product_to_idx[product_id] = i

        # create reverse mappings
        self.idx_to_country = {}
        for country_id, idx in self.country_to_idx.items():
            self.idx_to_country[idx] = country_id

        self.idx_to_product = {}
        for product_id, idx in self.product_to_idx.items():
            self.idx_to_product[idx] = product_id

    def build_master_graph(self):
        """create graph structure with all nodes"""

        graph = HeteroData()

        # get number of countries and products
        num_countries = len(self.country_to_idx)
        num_products = len(self.product_to_idx)

        # initialize node features with one-hot encoding
        graph['country'].x = torch.eye(num_countries)
        graph['product'].x = torch.eye(num_products)

        return graph

    def build_temporal_edges(self, df):
        """build edge data for each year"""

        year_to_edges = {}

        for year in self.years:
            # filter data for current year
            year_data = df[df['year'] == year]

            # initialize edge lists
            src_nodes = []
            dst_nodes = []
            edge_attrs = []

            # process export relationships
            exports = year_data[year_data['export_value'] > 0]
            if len(exports) > 0:
                for _, row in exports.iterrows():
                    # get source and destination indices
                    src = self.country_to_idx[row['country_id']]
                    dst = self.product_to_idx[row['product_id']]

                    # add to edge lists
                    src_nodes.append(src)
                    dst_nodes.append(dst)

                    # transform export value
                    edge_attrs.append(np.log1p(row['export_value']))

            # create edge index tensor
            edge_index = None
            if len(src_nodes) > 0:
                edge_index = torch.tensor([src_nodes, dst_nodes], dtype=torch.long)

            # create edge attribute tensor
            edge_attr = None
            if len(edge_attrs) > 0:
                edge_attr = torch.tensor(edge_attrs, dtype=torch.float).unsqueeze(1)

            # store edge data for this year
            year_to_edges[year] = {
                'edge_index': edge_index,
                'edge_attr': edge_attr
            }

        return year_to_edges

    def generate_samples(self):
        """create training samples using sliding window"""

        samples = []

        # iterate through possible time windows
        for i in range(len(self.years) - self.window_size):
            input_years = self.years[i:i+self.window_size]
            target_year = self.years[i+self.window_size]

            # get edges that exist in target year
            target_edges = set()
            if self.year_to_edges[target_year]['edge_index'] is not None:
                edge_pairs = self.year_to_edges[target_year]['edge_index'].t().tolist()
                for src, dst in edge_pairs:
                    target_edges.add((src, dst))

            # get edges that exist in last input year
            last_input_edges = set()
            if self.year_to_edges[input_years[-1]]['edge_index'] is not None:
                edge_pairs = self.year_to_edges[input_years[-1]]['edge_index'].t().tolist()
                for src, dst in edge_pairs:
                    last_input_edges.add((src, dst))

            # positive samples are new edges in target year
            new_edges = target_edges - last_input_edges

            # existing edges includes both target and last input edges
            existing_edges = target_edges.union(last_input_edges)

            # get number of nodes
            num_countries = len(self.country_to_idx)
            num_products = len(self.product_to_idx)

            # create positive samples
            for src, dst in new_edges:
                samples.append({
                    'input_years': input_years,
                    'country_idx': src,
                    'product_idx': dst,
                    'label': 1
                })

            # create negative samples
            neg_samples_needed = min(len(new_edges), 1000)
            neg_samples_added = 0
            attempts = 0
            max_attempts = neg_samples_needed * 10

            while neg_samples_added < neg_samples_needed and attempts < max_attempts:
                # generate random country-product pair
                src = torch.randint(0, num_countries, (1,)).item()
                dst = torch.randint(0, num_products, (1,)).item()

                # check if this pair doesn't exist
                if (src, dst) not in existing_edges:
                    samples.append({
                        'input_years': input_years,
                        'country_idx': src,
                        'product_idx': dst,
                        'label': 0
                    })
                    neg_samples_added += 1

                attempts += 1

        return samples

    def __len__(self):
        """get total number of samples"""
        return len(self.samples)

    def __getitem__(self, idx):
        """get sample by index"""

        sample = self.samples[idx]

        # build graph sequence for this sample
        graph_sequence = []

        for year in sample['input_years']:
            # create new graph with same node features
            g = HeteroData()
            g['country'].x = self.master_graph['country'].x
            g['product'].x = self.master_graph['product'].x

            # add edges for this year
            edges = self.year_to_edges[year]
            if edges['edge_index'] is not None:
                g['country', 'exports', 'product'].edge_index = edges['edge_index']
                if edges['edge_attr'] is not None:
                    g['country', 'exports', 'product'].edge_attr = edges['edge_attr']

            graph_sequence.append(g)

        return (
            graph_sequence,
            sample['country_idx'],
            sample['product_idx'],
            sample['label']
        )
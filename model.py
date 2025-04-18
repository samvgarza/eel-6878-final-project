import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv, GATConv
from torch_geometric.nn.conv import HGTConv

class TemporalHeteroGNN(nn.Module):
    def __init__(self, num_countries, num_products, hidden_dim=64, num_layers=2):
        super(TemporalHeteroGNN, self).__init__()

        # Node type-specific embeddings
        self.country_embedding = nn.Embedding(num_countries, hidden_dim)
        self.product_embedding = nn.Embedding(num_products, hidden_dim)

        nn.init.xavier_uniform_(self.country_embedding.weight)
        nn.init.xavier_uniform_(self.product_embedding.weight)

        # Heterogeneous Graph Transformer layers
        self.hgt_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.hgt_layers.append(HGTConv(
                in_channels={'country': hidden_dim, 'product': hidden_dim},
                out_channels=hidden_dim,
                metadata=(['country', 'product'],
                          [('country', 'exports', 'product'),
                           ('product', 'imports', 'country')]),
                heads=4
            ))

        # Temporal component (GRU)
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)

        # For the GRU
        for name, param in self.gru.named_parameters():
            if 'weight' in name:
                nn.init.orthogonal_(param)

        # Link prediction layers
        self.link_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def encode_graph(self, graph):
        # Get initial node embeddings
        country_x = self.country_embedding(torch.arange(graph['country'].num_nodes))
        product_x = self.product_embedding(torch.arange(graph['product'].num_nodes))

        # Dict of node features
        x_dict = {'country': country_x, 'product': product_x}

        # Apply HGT layers
        for layer in self.hgt_layers:
            # Modified line: Remove edge_attr_dict parameter
            x_dict = layer(x_dict, graph.edge_index_dict)

        return x_dict

    def forward(self, graph_sequence, target_country_idx, target_product_idx):
        # Encode each graph in the sequence
        sequence_embeddings = []
        for graph in graph_sequence:
            x_dict = self.encode_graph(graph)
            sequence_embeddings.append(x_dict)

        # Stack country embeddings across time steps
        country_seq = torch.stack([emb['country'] for emb in sequence_embeddings], dim=1)
        product_seq = torch.stack([emb['product'] for emb in sequence_embeddings], dim=1)

        # Apply GRU for temporal reasoning
        _, country_h = self.gru(country_seq)
        _, product_h = self.gru(product_seq)

        # Get final embeddings (squeeze to remove batch dimension)
        country_emb = country_h.squeeze(0)
        product_emb = product_h.squeeze(0)

        # Get specific country and product embeddings
        target_country_emb = country_emb[target_country_idx]
        target_product_emb = product_emb[target_product_idx]

        # Concatenate for link prediction
        pair_emb = torch.cat([target_country_emb, target_product_emb], dim=-1)

        # Predict link probability
        pred = self.link_predictor(pair_emb)
        return torch.sigmoid(pred)

    def predict(self, graph_sequence, all_country_indices, all_product_indices):
        """Generate predictions for all country-product pairs."""
        # Encode each graph in the sequence
        sequence_embeddings = []
        for graph in graph_sequence:
            x_dict = self.encode_graph(graph)
            sequence_embeddings.append(x_dict)

        # Apply GRU for temporal reasoning
        country_seq = torch.stack([emb['country'] for emb in sequence_embeddings], dim=1)
        product_seq = torch.stack([emb['product'] for emb in sequence_embeddings], dim=1)

        _, country_h = self.gru(country_seq)
        _, product_h = self.gru(product_seq)

        # Get final embeddings
        country_emb = country_h.squeeze(0)
        product_emb = product_h.squeeze(0)

        # Get predictions for all pairs
        predictions = {}
        for c_idx in all_country_indices:
            for p_idx in all_product_indices:
                pair_emb = torch.cat([country_emb[c_idx], product_emb[p_idx]], dim=-1)
                pred = torch.sigmoid(self.link_predictor(pair_emb))
                predictions[(c_idx, p_idx)] = pred.item()

        return predictions
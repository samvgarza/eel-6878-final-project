import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HGTConv


class TemporalAttention(nn.Module):
    def __init__(self, hidden_dim):
        super(TemporalAttention, self).__init__()
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)

        # Scaling factor for dot product attention
        self.scale = torch.sqrt(torch.FloatTensor([hidden_dim]))

        # Layer normalization and feed-forward network
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )

        # Initialize weights
        nn.init.xavier_uniform_(self.query.weight)
        nn.init.xavier_uniform_(self.key.weight)
        nn.init.xavier_uniform_(self.value.weight)

    def forward(self, x):
        # x shape: [batch_size, seq_len, hidden_dim]
        batch_size, seq_len, hidden_dim = x.size()

        # Create residual connection
        residual = x

        # Linear projections for Q, K, V
        Q = self.query(x)  # [batch_size, seq_len, hidden_dim]
        K = self.key(x)    # [batch_size, seq_len, hidden_dim]
        V = self.value(x)  # [batch_size, seq_len, hidden_dim]

        # Calculate attention scores
        attention = torch.matmul(Q, K.transpose(-2, -1)) / self.scale.to(x.device)

        # Apply softmax to get attention weights
        attention_weights = F.softmax(attention, dim=-1)

        # Apply attention weights to values
        context = torch.matmul(attention_weights, V)

        # Apply first residual connection and normalization
        output = self.layer_norm1(residual + context)

        # Feed-forward network
        residual = output
        output = self.ffn(output)

        # Apply second residual connection and normalization
        output = self.layer_norm2(residual + output)

        # Return final output and attention weights for visualization if needed
        return output, attention_weights


class TemporalHeteroGNN(nn.Module):
    def __init__(self, num_countries, num_products, hidden_dim=64, num_layers=2):
        super(TemporalHeteroGNN, self).__init__()

        # store the dimensions we'll be using
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # create embedding layers for countries and products
        self.country_embedding = nn.Embedding(num_countries, hidden_dim)
        self.product_embedding = nn.Embedding(num_products, hidden_dim)

        # initialize the embedding weights properly
        nn.init.xavier_uniform_(self.country_embedding.weight)
        nn.init.xavier_uniform_(self.product_embedding.weight)

        # create list to hold the HGT layers
        self.hgt_layers = nn.ModuleList()

        # add the specified number of HGT layers
        for layer_num in range(num_layers):
            hgt_layer = HGTConv(
                in_channels={
                    'country': hidden_dim,
                    'product': hidden_dim
                },
                out_channels=hidden_dim,
                metadata=(
                    ['country', 'product'],  # node types
                    [  # edge types
                        ('country', 'exports', 'product'),
                        ('product', 'imports', 'country')
                    ]
                ),
                heads=4  # number of attention heads
            )
            self.hgt_layers.append(hgt_layer)

        # Replace GRU with Temporal Attention
        self.temporal_attn_country = TemporalAttention(hidden_dim)
        self.temporal_attn_product = TemporalAttention(hidden_dim)

        # Create position embeddings for temporal attention
        self.max_seq_len = 50  # Adjust based on your expected maximum sequence length
        self.pos_embedding = nn.Parameter(torch.zeros(1, self.max_seq_len, hidden_dim))
        nn.init.normal_(self.pos_embedding, std=0.02)

        # create the link prediction network
        self.link_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # takes concatenated embeddings
            nn.ReLU(),  # activation function
            nn.Linear(hidden_dim, 1)  # final prediction
        )

    def encode_graph(self, graph):
        # get the number of countries and products in this graph
        num_countries = graph['country'].num_nodes
        num_products = graph['product'].num_nodes

        # create initial embeddings for all countries
        country_indices = torch.arange(num_countries)
        country_embeddings = self.country_embedding(country_indices)

        # create initial embeddings for all products
        product_indices = torch.arange(num_products)
        product_embeddings = self.product_embedding(product_indices)

        # store embeddings in a dictionary
        node_embeddings = {
            'country': country_embeddings,
            'product': product_embeddings
        }

        # apply each HGT layer to the graph
        for hgt_layer in self.hgt_layers:
            node_embeddings = hgt_layer(node_embeddings, graph.edge_index_dict)

        return node_embeddings

    def forward(self, graph_sequence, target_country_idx, target_product_idx):
        # this list will store embeddings for each graph in the sequence
        sequence_embeddings = []

        # process each graph in the sequence
        for graph in graph_sequence:
            # get embeddings for this graph
            graph_embeddings = self.encode_graph(graph)
            sequence_embeddings.append(graph_embeddings)

        # collect all country embeddings across time
        country_embeddings = []
        for emb in sequence_embeddings:
            country_embeddings.append(emb['country'])
        # stack them along the time dimension
        country_sequence = torch.stack(country_embeddings, dim=1)  # [num_countries, seq_len, hidden_dim]

        # collect all product embeddings across time
        product_embeddings = []
        for emb in sequence_embeddings:
            product_embeddings.append(emb['product'])
        # stack them along the time dimension
        product_sequence = torch.stack(product_embeddings, dim=1)  # [num_products, seq_len, hidden_dim]

        # Add positional embeddings
        seq_len = country_sequence.size(1)
        country_sequence = country_sequence + self.pos_embedding[:, :seq_len, :]
        product_sequence = product_sequence + self.pos_embedding[:, :seq_len, :]

        # Apply temporal attention
        country_attn_output, country_attn_weights = self.temporal_attn_country(country_sequence)
        product_attn_output, product_attn_weights = self.temporal_attn_product(product_sequence)

        # Use the last time step as the final representation
        country_final = country_attn_output[:, -1, :]  # [num_countries, hidden_dim]
        product_final = product_attn_output[:, -1, :]  # [num_products, hidden_dim]

        # get the specific embeddings we want to predict for
        target_country_embedding = country_final[target_country_idx]
        target_product_embedding = product_final[target_product_idx]

        # combine the embeddings for prediction
        combined_embedding = torch.cat(
            [target_country_embedding, target_product_embedding],
            dim=-1
        )

        # make the prediction
        raw_prediction = self.link_predictor(combined_embedding)
        # apply sigmoid to get probability between 0 and 1
        prediction_probability = torch.sigmoid(raw_prediction)

        return prediction_probability

    def predict(self, graph_sequence, all_country_indices, all_product_indices):
        # this will store all our predictions
        predictions = {}

        # first get embeddings for each graph in the sequence
        sequence_embeddings = []
        for graph in graph_sequence:
            graph_embeddings = self.encode_graph(graph)
            sequence_embeddings.append(graph_embeddings)

        # process country embeddings through temporal attention
        country_embeddings = []
        for emb in sequence_embeddings:
            country_embeddings.append(emb['country'])
        country_sequence = torch.stack(country_embeddings, dim=1)

        # process product embeddings through temporal attention
        product_embeddings = []
        for emb in sequence_embeddings:
            product_embeddings.append(emb['product'])
        product_sequence = torch.stack(product_embeddings, dim=1)

        # Add positional embeddings
        seq_len = country_sequence.size(1)
        country_sequence = country_sequence + self.pos_embedding[:, :seq_len, :]
        product_sequence = product_sequence + self.pos_embedding[:, :seq_len, :]

        # Apply temporal attention
        country_attn_output, _ = self.temporal_attn_country(country_sequence)
        product_attn_output, _ = self.temporal_attn_product(product_sequence)

        # Use the last time step as the final representation
        country_final = country_attn_output[:, -1, :]
        product_final = product_attn_output[:, -1, :]

        # make predictions for all requested country-product pairs
        for country_idx in all_country_indices:
            for product_idx in all_product_indices:
                # get the embeddings for this pair
                country_emb = country_final[country_idx]
                product_emb = product_final[product_idx]

                # combine the embeddings
                combined = torch.cat([country_emb, product_emb], dim=-1)

                # make prediction
                raw_pred = self.link_predictor(combined)
                pred_prob = torch.sigmoid(raw_pred).item()

                # store the prediction
                predictions[(country_idx, product_idx)] = pred_prob

        return predictions
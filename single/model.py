import torch
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import SAGEConv, to_hetero
from torch_geometric.data import HeteroData

# gnn model
class GNN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.conv1 = SAGEConv(hidden_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x

# link classifier
class Classifier(torch.nn.Module):
    def forward(self, x_country: Tensor, x_product: Tensor, edge_label_index: Tensor) -> Tensor:
        # get edge-level features
        edge_feat_country = x_country[edge_label_index[0]]
        edge_feat_product = x_product[edge_label_index[1]]
        # dot product for edge prediction
        return (edge_feat_country * edge_feat_product).sum(dim=-1)

# main model
class Model(torch.nn.Module):
    def __init__(self, hidden_channels, num_countries, num_products, metadata=None):
        super().__init__()
        # linear layers for initial features
        self.country_lin = torch.nn.Linear(num_countries, hidden_channels)
        self.product_lin = torch.nn.Linear(num_products, hidden_channels)

        # embedding layers
        self.country_emb = torch.nn.Embedding(num_countries, hidden_channels)
        self.product_emb = torch.nn.Embedding(num_products, hidden_channels)

        # gnn converted to heterogeneous
        self.gnn = GNN(hidden_channels)
        if metadata:
            self.gnn = to_hetero(self.gnn, metadata=metadata)

        self.classifier = Classifier()

        # direct prediction for country-product pairs
        self.predictor = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels * 2, hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_channels, 1),
            torch.nn.Sigmoid()
        )

    def forward(self, data: HeteroData) -> Tensor:
        x_dict = {
            "country": self.country_lin(data["country"].x) + self.country_emb(data["country"].node_id),
            "product": self.product_lin(data["product"].x) + self.product_emb(data["product"].node_id),
        }

        # pass through gnn
        x_dict = self.gnn(x_dict, data.edge_index_dict)

        # predict links
        pred = self.classifier(
            x_dict["country"],
            x_dict["product"],
            data["country", "exports", "product"].edge_label_index,
        )
        return pred

    def predict_pair(self, country_idx, product_idx):

        # get embeddings
        country_emb = self.country_emb(country_idx)
        product_emb = self.product_emb(product_idx)

        # concat & predict
        x = torch.cat([country_emb, product_emb], dim=1)
        return self.predictor(x).squeeze()
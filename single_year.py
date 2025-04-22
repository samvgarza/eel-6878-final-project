import torch
import pandas as pd
import numpy as np
from torch import Tensor
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import SAGEConv, to_hetero
from torch_geometric.loader import LinkNeighborLoader
import torch_geometric.transforms as T
from sklearn.metrics import roc_auc_score
import tqdm

# heavily ripped from https://medium.com/@pytorch_geometric/link-prediction-on-heterogeneous-graphs-with-pyg-6d5c29677c70

# load 2023 data
df = pd.read_parquet('test/data/sitc_country_country_product_year_4_2020_2023.parquet')
df_2023 = df[df['year'] == 2023].reset_index(drop=True)

print("sample of the 2023 data:")
print("========================")
print(df_2023.head())

# create mappings for countries and products
unique_countries = sorted(df_2023['country_id'].unique())
unique_products = sorted(df_2023['product_id'].unique())

country_to_idx = {country: idx for idx, country in enumerate(unique_countries)}
product_to_idx = {product: idx for idx, product in enumerate(unique_products)}

# create export links (country exports product if export_value > 0)
export_links = df_2023[['country_id', 'product_id', 'export_value']]
export_links['exports'] = (export_links['export_value'] > 0).astype(int)
export_links = export_links[export_links['exports'] == 1][['country_id', 'product_id']]

# convert to mapped indices
export_links['country_idx'] = export_links['country_id'].map(country_to_idx)
export_links['product_idx'] = export_links['product_id'].map(product_to_idx)

print("\nsample of export links with mapped indices:")
print("==========================================")
print(export_links.head())

# create edge_index in COO format
country_indices = torch.tensor(export_links['country_idx'].values, dtype=torch.long)
product_indices = torch.tensor(export_links['product_idx'].values, dtype=torch.long)
edge_index_country_to_product = torch.stack([country_indices, product_indices], dim=0)

print(f"\nnumber of unique countries: {len(unique_countries)}")
print(f"number of unique products: {len(unique_products)}")
print(f"edge index shape: {edge_index_country_to_product.shape}")

# create set of positive edges for quick lookup
pos_edge_set = set()

for country_idx, product_idx in zip(country_indices, product_indices):
    # tensor to numbers
    country_num = country_idx.item()
    product_num = product_idx.item()

    pos_edge_set.add((country_num, product_num))

print(f"\ncountry features shape: {torch.eye(len(unique_countries)).size()}")
print(f"product features shape: {torch.eye(len(unique_products)).size()}")

# init heterodata object
data = HeteroData()

# add node ids and features
data["country"].node_id = torch.arange(len(unique_countries))
data["product"].node_id = torch.arange(len(unique_products))

# simple one-hot encodings # NOTE: can change this to be meaningful
data["country"].x = torch.eye(len(unique_countries)).float()
data["product"].x = torch.eye(len(unique_products)).float()

# add edge indices
data["country", "exports", "product"].edge_index = edge_index_country_to_product

# make graph undirected
data = T.ToUndirected()(data)
print("\nheterogeneous graph data:")
print("========================")
print(data)

# generate negative samples
def generate_negative_samples(pos_edge_set, num_countries, num_products, num_samples):
    neg_edges = []
    max_attempts = num_samples * 20  # limit attempts
    attempts = 0

    while len(neg_edges) < num_samples and attempts < max_attempts:
        # random country and product
        country_idx = torch.randint(0, num_countries, (1,)).item()
        product_idx = torch.randint(0, num_products, (1,)).item()

        # check if edge doesn't exist
        if (country_idx, product_idx) not in pos_edge_set:
            neg_edges.append((country_idx, product_idx))
            pos_edge_set.add((country_idx, product_idx))  # avoid duplicates

        attempts += 1

    if len(neg_edges) < num_samples:
        print(f"warning: only generated {len(neg_edges)} negative samples out of {num_samples}")

    neg_edge_index = torch.tensor(neg_edges, dtype=torch.long).t()
    return neg_edge_index

# split edges into train/val/test
num_edges = edge_index_country_to_product.size(1)
perm = torch.randperm(num_edges)

total_edges = num_edges
train_count = int(0.8 * total_edges)
val_count = int(0.1 * total_edges)

# grab the indices for each set
train_idx = perm[:train_count] # first 80%
val_idx = perm[train_count : train_count + val_count] # next 10%
test_idx = perm[train_count + val_count :] # last 10%

# create edge indices for splits
train_edge_index = edge_index_country_to_product[:, train_idx]
val_pos_edge_index = edge_index_country_to_product[:, val_idx]
test_pos_edge_index = edge_index_country_to_product[:, test_idx]

# generate negative edges for val and test
val_neg_edge_index = generate_negative_samples(
    pos_edge_set.copy(),  # copy to avoid modifying original
    len(unique_countries),
    len(unique_products),
    val_pos_edge_index.size(1)
)

test_neg_edge_index = generate_negative_samples(
    pos_edge_set.copy(),
    len(unique_countries),
    len(unique_products),
    test_pos_edge_index.size(1)
)

# split train edges into message passing and supervision
num_train = train_edge_index.size(1)
perm = torch.randperm(num_train)
train_mp_idx = perm[:int(0.7 * num_train)]  # 70% for message passing
train_sup_idx = perm[int(0.7 * num_train):]  # 30% for supervision

train_mp_edge_index = train_edge_index[:, train_mp_idx]
train_sup_pos_edge_index = train_edge_index[:, train_sup_idx]

# negative samples for training supervision
train_sup_neg_edge_index = generate_negative_samples(
    pos_edge_set.copy(),
    len(unique_countries),
    len(unique_products),
    train_sup_pos_edge_index.size(1)
)

# create data objects for splits
train_data = data.clone()
val_data = data.clone()
test_data = data.clone()

# set up train data
train_data["country", "exports", "product"].edge_index = train_mp_edge_index
train_data["country", "exports", "product"].edge_label_index = torch.cat([
    train_sup_pos_edge_index, train_sup_neg_edge_index
], dim=1)
train_data["country", "exports", "product"].edge_label = torch.cat([
    torch.ones(train_sup_pos_edge_index.size(1)),
    torch.zeros(train_sup_neg_edge_index.size(1))
])

# set up val data
val_data["country", "exports", "product"].edge_index = train_edge_index  # message passing
val_data["country", "exports", "product"].edge_label_index = torch.cat([
    val_pos_edge_index, val_neg_edge_index
], dim=1)
val_data["country", "exports", "product"].edge_label = torch.cat([
    torch.ones(val_pos_edge_index.size(1)),
    torch.zeros(val_neg_edge_index.size(1))
])

# set up test data
test_data["country", "exports", "product"].edge_index = torch.cat([
    train_edge_index, val_pos_edge_index
], dim=1)  # message passing

test_data["country", "exports", "product"].edge_label_index = torch.cat([
    test_pos_edge_index, test_neg_edge_index
], dim=1)
test_data["country", "exports", "product"].edge_label = torch.cat([
    torch.ones(test_pos_edge_index.size(1)),
    torch.zeros(test_neg_edge_index.size(1))
])

# mini-batch loaders
train_loader = LinkNeighborLoader(
    data=train_data,
    num_neighbors=[20, 10],
    batch_size=128,
    edge_label_index=(("country", "exports", "product"),
                      train_data["country", "exports", "product"].edge_label_index),
    edge_label=train_data["country", "exports", "product"].edge_label,
    shuffle=True,
)

val_loader = LinkNeighborLoader(
    data=val_data,
    num_neighbors=[20, 10],
    batch_size=128,
    edge_label_index=(("country", "exports", "product"),
                     val_data["country", "exports", "product"].edge_label_index),
    edge_label=val_data["country", "exports", "product"].edge_label,
    shuffle=False,
)

test_loader = LinkNeighborLoader(
    data=test_data,
    num_neighbors=[20, 10],
    batch_size=128,
    edge_label_index=(("country", "exports", "product"),
                      test_data["country", "exports", "product"].edge_label_index),
    edge_label=test_data["country", "exports", "product"].edge_label,
    shuffle=False,
)

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
    def __init__(self, hidden_channels, num_countries, num_products):
        super().__init__()
        # linear layers for initial features
        self.country_lin = torch.nn.Linear(num_countries, hidden_channels)
        self.product_lin = torch.nn.Linear(num_products, hidden_channels)

        # embedding layers
        self.country_emb = torch.nn.Embedding(num_countries, hidden_channels)
        self.product_emb = torch.nn.Embedding(num_products, hidden_channels)

        # gnn converted to heterogeneous
        self.gnn = GNN(hidden_channels)
        self.gnn = to_hetero(self.gnn, metadata=data.metadata())

        self.classifier = Classifier()

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

# init model
hidden_channels = 4
model = Model(
    hidden_channels=hidden_channels,
    num_countries=len(unique_countries),
    num_products=len(unique_products)
)

# training setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\ndevice: '{device}'")

model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

print("\ntraining the model...")
for epoch in range(1, 6):
    model.train()
    total_loss = total_examples = 0
    for sampled_data in tqdm.tqdm(train_loader):
        optimizer.zero_grad()
        sampled_data = sampled_data.to(device)
        pred = model(sampled_data)
        ground_truth = sampled_data["country", "exports", "product"].edge_label
        loss = F.binary_cross_entropy_with_logits(pred, ground_truth)
        loss.backward()
        optimizer.step()
        total_loss += float(loss) * pred.numel()
        total_examples += pred.numel()
    print(f"epoch: {epoch:03d}, loss: {total_loss / total_examples:.4f}")

# validation
print("\nevaluating on validation set...")
model.eval()
preds = []
ground_truths = []
for sampled_data in tqdm.tqdm(val_loader):
    with torch.no_grad():
        sampled_data = sampled_data.to(device)
        preds.append(model(sampled_data))
        ground_truths.append(sampled_data["country", "exports", "product"].edge_label)

pred = torch.cat(preds, dim=0).cpu().numpy()
ground_truth = torch.cat(ground_truths, dim=0).cpu().numpy()
auc = roc_auc_score(ground_truth, pred)
print(f"\nvalidation auc: {auc:.4f}")

# test eval
print("\nevaluating on test set...")
test_preds = []
test_ground_truths = []
for sampled_data in tqdm.tqdm(test_loader):
    with torch.no_grad():
        sampled_data = sampled_data.to(device)
        test_preds.append(model(sampled_data))
        test_ground_truths.append(sampled_data["country", "exports", "product"].edge_label)

test_pred = torch.cat(test_preds, dim=0).cpu().numpy()
test_ground_truth = torch.cat(test_ground_truths, dim=0).cpu().numpy()
test_auc = roc_auc_score(test_ground_truth, test_pred)
print(f"\ntest auc: {test_auc:.4f}")
import torch
import pandas as pd
import numpy as np
import tqdm
from torch_geometric.data import HeteroData
from torch_geometric.loader import LinkNeighborLoader
from torch_geometric import transforms
from sklearn.metrics import roc_auc_score
import torch.nn.functional as F
import argparse

from model import Model

# NOTE: helpful source: https://medium.com/@pytorch_geometric/link-prediction-on-heterogeneous-graphs-with-pyg-6d5c29677c70

parser = argparse.ArgumentParser(description='Trade prediction using GNNs')

parser.add_argument('--model_year', required=True, help='year to train model on')
args = parser.parse_args()

# load model data
df = pd.read_parquet('../test/data/sitc_country_country_product_year_4_1980_1989.parquet')
df_model_year = df[df['year'] == int(args.model_year)].reset_index(drop=True)

print(f"sample of the {args.model_year} data:")
print("========================")
print(df_model_year.head())

# create mappings for countries and products
unique_countries = sorted(df_model_year['country_id'].unique())
unique_products = sorted(df_model_year['product_id'].unique())

country_to_idx = {country: idx for idx, country in enumerate(unique_countries)}
product_to_idx = {product: idx for idx, product in enumerate(unique_products)}

# create export links (country exports product if export_value > 0)
export_links = df_model_year[['country_id', 'product_id', 'export_value']]
export_links = export_links[(export_links['export_value'] > 0) == True]

# convert to mapped indices
export_links['country_idx'] = export_links['country_id'].map(country_to_idx)
export_links['product_idx'] = export_links['product_id'].map(product_to_idx)

print("\nsample of export links with mapped indices:")
print("==========================================")
print(export_links)

# create edge_index in COO format
country_indices = torch.tensor(export_links['country_idx'].values, dtype=torch.long)
product_indices = torch.tensor(export_links['product_idx'].values, dtype=torch.long)
edge_index_country_to_product = torch.stack([country_indices, product_indices], dim=0)

print(f"\nnumber of unique countries: {len(unique_countries)}")
print(f"number of unique products: {len(unique_products)}")
print(f"edge index shape: {edge_index_country_to_product.shape}")

# init heterodata object
data = HeteroData()

# add node ids and features (maybe GDP??)
data["country"].node_id = torch.arange(len(unique_countries))
data["product"].node_id = torch.arange(len(unique_products))

# simple one-hot encodings
data["country"].x = torch.eye(len(unique_countries)).float()
data["product"].x = torch.eye(len(unique_products)).float()

# add edge indices
data["country", "exports", "product"].edge_index = edge_index_country_to_product

# make graph undirected
data = transforms.ToUndirected()(data)
print("\nheterogeneous graph data:")
print("========================")
print(data)

# create the set of positive edges
pos_edge_set = set()

for country_idx, product_idx in zip(country_indices, product_indices):
    # tensor to numbers
    country_num = country_idx.item()
    product_num = product_idx.item()

    pos_edge_set.add((country_num, product_num))


# NOTE: all of this to try and replicate the transforms.RandomLinkSplit that we couldn't get to work
# generate negative samples
def generate_negative_samples(pos_edge_set, num_countries, num_products, num_samples):
    neg_edges = []
    max_attempts = num_samples * 20  # limit attempts - this is crucial lol
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

    if len(neg_edges) < num_samples: # we never generate enough...
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

# mini-batch loaders (prob not needed for the smaller graphs but oh well)
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

# init model
hidden_channels = 4
model = Model(
    hidden_channels=hidden_channels,
    num_countries=len(unique_countries),
    num_products=len(unique_products),
    metadata=data.metadata()
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

# save trained model
torch.save({
    'model_state_dict': model.state_dict(),
    'country_to_idx': dict(country_to_idx),
    'product_to_idx': dict(product_to_idx),
    'hidden_channels': hidden_channels,
}, f'export_prediction_{args.model_year}.pth')
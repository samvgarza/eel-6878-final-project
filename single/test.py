import torch
import torch_geometric.transforms as T
from torch_geometric.data import HeteroData
import pandas as pd
import tqdm
from sklearn.metrics import roc_auc_score
import argparse

# import model class from model.py
from model import Model

parser = argparse.ArgumentParser(description='Trade prediction using graph neural networks')

parser.add_argument('--model_year', required=True, help='year model was trained on')
parser.add_argument('--test_year', required=True, help='year to test')
args = parser.parse_args()

# config settings
model_path = f"export_prediction_{args.model_year}.pth"

# load the trained model checkpoint
print(f"loading model from {model_path}...")
checkpoint = torch.load(model_path, weights_only=False)

# get model configuration from checkpoint
hidden_channels = checkpoint['hidden_channels']
country_to_idx = checkpoint['country_to_idx']
product_to_idx = checkpoint['product_to_idx']

# load test data for specified year
print(f"loading test data for year {args.test_year}...")
df_test = pd.read_parquet('../test/data/sitc_country_country_product_year_4_2020_2023.parquet')
df_test_year = df_test[df_test['year'] == int(args.test_year)].reset_index(drop=True)

# keep only countries and products that were in the training data
df_test_year = df_test_year[
    df_test_year['country_id'].isin(country_to_idx.keys()) &
    df_test_year['product_id'].isin(product_to_idx.keys())
].reset_index(drop=True)

print(f"\ntest dataset stats:")
print(f"- number of countries: {df_test_year['country_id'].nunique()}")
print(f"- number of products: {df_test_year['product_id'].nunique()}")

# NOTE: just building out graph so structure is the same
# init heterodata object
data = HeteroData()

# add node ids
data["country"].node_id = torch.zeros(len(country_to_idx), 1)
data["product"].node_id = torch.zeros(len(product_to_idx), 1)

# simple one-hot encodings
data["country"].x = torch.eye(len(country_to_idx)).float()
data["product"].x = torch.eye(len(product_to_idx)).float()

# add edge indices
data["country", "exports", "product"].edge_index = torch.zeros((2, 1), dtype=torch.long)  # Placeholder edge

# make graph undirected
data = T.ToUndirected()(data)

# load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Model(
    hidden_channels=hidden_channels,
    num_countries=len(country_to_idx),
    num_products=len(product_to_idx),
    metadata=data.metadata()
).to(device)

# load trained weights
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# prep all country-product pairs to test
country_ids = df_test_year['country_id'].unique()
product_ids = df_test_year['product_id'].unique()

# create dictionary of real export data
real_exports = {}
for _, row in df_test_year.iterrows():
    country_id = row['country_id']
    product_id = row['product_id']
    export_value = row['export_value']
    real_exports[(country_id, product_id)] = 1 if export_value > 0 else 0

# predict for all country-product pairs
print(f"predicting export links for {args.test_year}...")
results = []

with torch.no_grad():
    for country_id in tqdm.tqdm(country_ids):
        country_idx = torch.tensor([country_to_idx[country_id]], device=device)

        for product_id in product_ids:
            product_idx = torch.tensor([product_to_idx[product_id]], device=device)

            # predict probability of export link
            prob = model.predict_pair(country_idx, product_idx).item()

            # get actual export status
            actual = real_exports.get((country_id, product_id), 0)

            results.append({
                'country_id': country_id,
                'product_id': product_id,
                'predicted_probability': prob,
                'actual_export': actual
            })

# convert results to dataframe
results_df = pd.DataFrame(results)

# save predictions to csv
output_file = f"export_predictions_{args.test_year}.csv"
results_df.to_csv(output_file, index=False)
print(f"saved predictions to {output_file}")

predictions = results_df['predicted_probability'].values
actuals = results_df['actual_export'].values

# calculate accuracy
binary_preds = (predictions > 0.5).astype(int)
accuracy = (binary_preds == actuals).mean()
print(f"accuracy: {accuracy:.4f}")

# calculate auc
auc = roc_auc_score(actuals, predictions)
print(f"auc score: {auc:.4f}")
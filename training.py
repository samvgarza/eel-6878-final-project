import numpy as np
import torch
import torch.nn as nn

def prepare_training_data(yearly_graphs, years, window_size=5):
    """
    Prepare training data with a sliding window of graph snapshots.
    Each sample consists of window_size previous years and targets the next year.
    """
    X, y = [], []
    for i in range(len(years) - window_size):
        # Input: graph sequence for window_size years
        input_years = years[i:i+window_size]
        input_graphs = [yearly_graphs[year] for year in input_years]

        # Target: edges in the next year
        target_year = years[i+window_size]
        target_graph = yearly_graphs[target_year]

        # Get positive examples (edges that exist in target_year but not in the last input year)
        last_input_year = input_years[-1]
        last_input_graph = yearly_graphs[last_input_year]

        # Get edges that exist in target_year
        target_edges = []
        if ('country', 'exports', 'product') in target_graph.edge_index_dict:
            target_edges = target_graph[('country', 'exports', 'product')].edge_index.t().tolist()

        # Check which edges didn't exist in the last input year
        new_edges = []
        if ('country', 'exports', 'product') in last_input_graph.edge_index_dict:
            last_edges = set(map(tuple, last_input_graph[('country', 'exports', 'product')].edge_index.t().tolist()))
            new_edges = [(src, dst) for src, dst in target_edges if (src, dst) not in last_edges]
        else:
            new_edges = target_edges

        # Add positive examples
        for src, dst in new_edges:
            X.append((input_graphs, src, dst))
            y.append(1)

        # Add negative examples (randomly sample non-existent edges)
        num_countries = input_graphs[-1]['country'].num_nodes
        num_products = input_graphs[-1]['product'].num_nodes

        # Create sets for faster lookup
        target_edges_set = set(map(tuple, target_edges))

        last_edges_set = set()
        if ('country', 'exports', 'product') in last_input_graph.edge_index_dict:
            last_edges_set = set(map(tuple, last_input_graph[('country', 'exports', 'product')].edge_index.t().tolist()))

        # Match the number of positive examples with negative examples
        neg_samples_needed = len(new_edges)
        neg_samples = 0
        max_attempts = neg_samples_needed * 10  # Allow more attempts to find valid negatives
        attempts = 0

        while neg_samples < neg_samples_needed and attempts < max_attempts:
            neg_src = torch.randint(0, num_countries, (1,)).item()
            neg_dst = torch.randint(0, num_products, (1,)).item()

            # Check if this edge exists in the last input graph or target graph
            if (neg_src, neg_dst) not in last_edges_set and (neg_src, neg_dst) not in target_edges_set:
                X.append((input_graphs, neg_src, neg_dst))
                y.append(0)
                neg_samples += 1

            attempts += 1

        # If we couldn't find enough negative samples, just use what we have
        if neg_samples < neg_samples_needed:
            print(f"Warning: Could only find {neg_samples} negative samples out of {neg_samples_needed} needed")

    return X, y

def train_model(model, X, y, epochs=10, lr=0.0001, batch_size=32):
    """Train the temporal heterogeneous GNN model."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.BCELoss()

    # Convert labels to tensor
    y_tensor = torch.tensor(y, dtype=torch.float)

    # Training loop
    for epoch in range(epochs):
        model.train()
        total_loss = 0

        # Process in batches
        indices = torch.randperm(len(X))
        for i in range(0, len(X), batch_size):
            print("we are getting here")
            batch_indices = indices[i:i+batch_size]
            batch_loss = 0

            for idx in batch_indices:
                # Get sample
                graph_sequence, country_idx, product_idx = X[idx]
                target = y_tensor[idx]

                # Forward pass
                pred = model(graph_sequence, country_idx, product_idx)

                # Compute loss
                loss = criterion(pred, target.unsqueeze(0))
                batch_loss += loss
                print(batch_loss)
            # Backward pass
            optimizer.zero_grad()
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += batch_loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(X):.4f}")

    return model

def evaluate_model(model, X_test, y_test):
    """Evaluate the model performance."""
    model.eval()
    y_pred = []
    y_true = y_test

    with torch.no_grad():
        for graph_sequence, country_idx, product_idx in X_test:
            pred = model(graph_sequence, country_idx, product_idx)
            y_pred.append(pred.item())

    # Convert to numpy arrays
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)

    # Compute metrics
    from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score

    # Binary predictions (threshold = 0.5)
    y_binary = (y_pred > 0.5).astype(int)

    auc = roc_auc_score(y_true, y_pred)
    ap = average_precision_score(y_true, y_pred)
    acc = accuracy_score(y_true, y_binary)

    print(f"Test AUC: {auc:.4f}")
    print(f"Test AP: {ap:.4f}")
    print(f"Test Accuracy: {acc:.4f}")

    return auc, ap, acc
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np


def train_model(model, train_dataset, test_dataset=None, epochs=10, lr=0.0001, batch_size=32):
    """Train the temporal heterogeneous GNN model."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    exit()
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.BCELoss()

    # Create a simpler collate function that processes one batch at a time
    def collate_fn(batch):
        # Separate the batch components
        graph_sequences = [item[0] for item in batch]  # List of graph sequences
        country_indices = torch.tensor([item[1] for item in batch])
        product_indices = torch.tensor([item[2] for item in batch])
        labels = torch.tensor([item[3] for item in batch], dtype=torch.float)

        return graph_sequences, country_indices, product_indices, labels

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )

    # Create test loader if test dataset is provided
    test_loader = None
    if test_dataset is not None:
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn
        )

    # Training loop with progress reporting
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        batch_count = 0

        # Use tqdm for progress bar if available
        try:
            from tqdm import tqdm
            loader_iter = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        except ImportError:
            loader_iter = train_loader
            print(f"Starting Epoch {epoch+1}/{epochs}")

        for graph_sequences, country_indices, product_indices, labels in loader_iter:
            batch_size = len(graph_sequences)

            # Process batch
            predictions = []
            country_indices = country_indices.to(device)
            product_indices = product_indices.to(device)
            labels = labels.to(device)

            # Forward pass for each sample in the batch
            for i in range(batch_size):
                # Move graphs to device (can't batch heterogeneous graphs easily)
                graphs = [g.to(device) for g in graph_sequences[i]]

                # Get prediction for this sample
                pred = model(graphs, country_indices[i], product_indices[i])
                predictions.append(pred)

            # Stack predictions and compute loss for the entire batch at once
            predictions = torch.cat(predictions, dim=0)
            loss = criterion(predictions, labels)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            batch_count += 1

        avg_train_loss = total_loss / batch_count
        print(f"Epoch {epoch+1}/{epochs}, Training Loss: {avg_train_loss:.4f}")

        # Validation phase
        if test_loader:
            model.eval()
            total_test_loss = 0
            test_count = 0
            correct_preds = 0
            total_preds = 0

            with torch.no_grad():
                for graph_sequences, country_indices, product_indices, labels in test_loader:
                    test_batch_size = len(graph_sequences)

                    # Process test batch
                    predictions = []
                    country_indices = country_indices.to(device)
                    product_indices = product_indices.to(device)
                    labels = labels.to(device)

                    for i in range(test_batch_size):
                        graphs = [g.to(device) for g in graph_sequences[i]]
                        pred = model(graphs, country_indices[i], product_indices[i])
                        predictions.append(pred)

                    predictions = torch.cat(predictions, dim=0)
                    test_loss = criterion(predictions, labels)

                    # Calculate accuracy
                    binary_preds = (predictions > 0.5).float()
                    correct_preds += (binary_preds == labels).sum().item()
                    total_preds += labels.size(0)

                    total_test_loss += test_loss.item()
                    test_count += 1

            avg_test_loss = total_test_loss / test_count
            accuracy = correct_preds / total_preds
            print(f"Validation Loss: {avg_test_loss:.4f}, Accuracy: {accuracy:.4f}")

    return model

def evaluate_model(model, dataset):
    """Evaluate the model performance."""
    model.eval()
    y_pred = []
    y_true = []

    loader = DataLoader(
        dataset,
        batch_size=32,
        collate_fn=lambda batch: (
            [item[0] for item in batch],
            torch.tensor([item[1] for item in batch]),
            torch.tensor([item[2] for item in batch]),
            torch.tensor([item[3] for item in batch])
        )
    )

    with torch.no_grad():
        for graph_sequences, country_idxs, product_idxs, labels in loader:
            for i in range(len(graph_sequences)):
                pred = model(graph_sequences[i], country_idxs[i], product_idxs[i])
                y_pred.append(pred.item())
                y_true.append(labels[i].item())

    # Compute metrics
    from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)

    auc = roc_auc_score(y_true, y_pred)
    ap = average_precision_score(y_true, y_pred)
    acc = accuracy_score(y_true, (y_pred > 0.5).astype(int))

    return auc, ap, acc
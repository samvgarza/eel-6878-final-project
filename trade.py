import pandas as pd
import torch
import os
import argparse
from datetime import datetime

# Import functions from previous modules
# (assuming you've saved them into separate files)
from data_processing import build_yearly_graphs
from model import TemporalHeteroGNN
from training import prepare_training_data, train_model, evaluate_model
from analysis import (analyze_product_space, calculate_product_proximity,
                     visualize_product_network, analyze_country_development_paths,
                     analyze_regional_clusters, predict_product_adoption_likelihood,
                     identify_product_potential, visualize_product_adoption_potential)

def main():
    """Main execution function for the trade prediction project."""
    parser = argparse.ArgumentParser(description='Trade Network Analysis and Prediction')
    parser.add_argument('--data', required=True, help='Path to the trade data CSV file')
    parser.add_argument('--output', default='output', help='Output directory for results')
    parser.add_argument('--years', default='1962-1969', help='Year range to use (e.g., 1990-2023)')
    parser.add_argument('--window', type=int, default=5, help='Time window size for prediction')
    parser.add_argument('--hidden-dim', type=int, default=64, help='Hidden dimension size')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--no-train', action='store_true', help='Skip training (load model instead)')
    parser.add_argument('--model-path', default='model.pt', help='Path to save/load model')
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    # Set up logging
    log_file = os.path.join(args.output, 'erm.txt')
    def log(message):
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        with open(log_file, 'a') as f:
            f.write(f"[{timestamp}] {message}\n")
        print(f"[{timestamp}] {message}")

    log(f"Starting trade network analysis with args: {args}")

    # Parse year range
    start_year, end_year = map(int, args.years.split('-'))

    # Load data
    log(f"Loading data from {args.data}...")
    df = pd.read_parquet(args.data)

    # Filter by year range
    df = df[(df['year'] >= start_year) & (df['year'] <= end_year)]
    log(f"Data filtered to years {start_year}-{end_year}")

    # Build graphs
    log("Building yearly graphs...")
    years = sorted(df['year'].unique())

    yearly_graphs, country_mapping, product_mapping = build_yearly_graphs(df, years)
    log(f"Built {len(yearly_graphs)} yearly graphs")

    # Initialize model
    num_countries = max(country_mapping.values()) + 1
    num_products = max(product_mapping.values()) + 1
    log(f"Initializing model with {num_countries} countries and {num_products} products")

    model = TemporalHeteroGNN(num_countries, num_products, hidden_dim=args.hidden_dim)

    if not args.no_train:
        # Prepare training data
        log(f"Preparing training data with window size {args.window}...")
        X, y = prepare_training_data(yearly_graphs, years, window_size=args.window)
        log(f"Prepared {len(X)} training samples")

        # Split data
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        log(f"Split into {len(X_train)} training and {len(X_test)} testing samples")

        # Train model
        log(f"Training model for {args.epochs} epochs...")
        model = train_model(model, X_train, y_train, epochs=args.epochs)

        # Save model
        torch.save(model.state_dict(), os.path.join(args.output, args.model_path))
        log(f"Model saved to {os.path.join(args.output, args.model_path)}")

        # Evaluate model
        log("Evaluating model...")
        auc, ap, acc = evaluate_model(model, X_test, y_test)
        log(f"Test results - AUC: {auc:.4f}, AP: {ap:.4f}, Accuracy: {acc:.4f}")
    else:
        # Load model
        log(f"Loading model from {args.model_path}...")
        model.load_state_dict(torch.load(args.model_path))
        model.eval()

    # Run analyses
    log("Running analyses...")

    # 1. Product Space Analysis
    log("Analyzing product space...")
    latest_year = max(years)
    latest_graph = yearly_graphs[latest_year]
    product_embeddings, product_embeddings_2d, product_names = analyze_product_space(
        model, latest_graph, product_mapping, df)
    log("Product space analysis complete")

    # 2. Product Proximity Analysis
    log("Calculating product proximity...")
    proximity_matrix = calculate_product_proximity(model, latest_graph)
    log("Visualizing product network...")
    visualize_product_network(proximity_matrix, product_mapping, df, threshold=0.7)
    log("Product network visualization complete")

    # 3. Country Development Path Analysis
    log("Analyzing country development paths...")
    embeddings_2d_by_year, country_codes = analyze_country_development_paths(
        model, yearly_graphs, country_mapping, df)
    log("Country development analysis complete")

    # 4. Regional Cluster Analysis
    log("Analyzing regional clusters...")
    cluster_analysis, labels, country_embeddings_2d = analyze_regional_clusters(
        model, latest_graph, country_mapping, df)
    log("Regional cluster analysis complete")

    # 5. Product Adoption Prediction
    log("Predicting product adoption...")
    # Get the most recent window of years
    recent_years = years[-args.window:]
    input_graphs = [yearly_graphs[year] for year in recent_years]

    # Predict adoption likelihood
    country_predictions = predict_product_adoption_likelihood(
        model, input_graphs, country_mapping, product_mapping, df)

    # Save predictions to file
    import json
    with open(os.path.join(args.output, 'country_predictions.json'), 'w') as f:
        json.dump(country_predictions, f, indent=2)
    log(f"Saved country predictions to {os.path.join(args.output, 'country_predictions.json')}")

    # 6. Global Product Potential Analysis
    log("Analyzing global product potential...")
    top_products = identify_product_potential(model, input_graphs, country_mapping, product_mapping, df)

    # Save top products to file
    with open(os.path.join(args.output, 'top_products.json'), 'w') as f:
        json.dump(top_products, f, indent=2)
    log(f"Saved top products to {os.path.join(args.output, 'top_products.json')}")

    # Visualize product adoption potential
    visualize_product_adoption_potential(top_products)
    log("Product potential visualization complete")

    # Print summary
    log("Analysis complete!")
    log(f"All results saved to {args.output}")

    # Sample predictions for select countries
    log("\nSample Predictions:")
    selected_countries = ['USA', 'CHN', 'DEU', 'KOR', 'THA']
    for country in selected_countries:
        if country in country_predictions:
            log(f"\nTop 5 products for {country}:")
            for product in country_predictions[country][:5]:
                log(f"  SITC {product['sitc_code']}: {product['probability']:.4f} probability")

if __name__ == "__main__":
    main()
import pandas as pd
import torch
import os
import argparse
from datetime import datetime

# import our custom modules
from data_processing import TemporalGraphDataset
from model import TemporalHeteroGNN
from training import train_model, evaluate_model

def main():

    print(torch.cuda.is_available())
    exit()
    # set up command line arguments
    parser = argparse.ArgumentParser(description='Trade prediction using graph neural networks')

    # add all command line arguments
    parser.add_argument('--data', required=True, help='path to the trade data file')
    parser.add_argument('--output', default='output', help='directory to save results')
    parser.add_argument('--years', default='1962-1969', help='year range to analyze')
    parser.add_argument('--window', type=int, default=5, help='number of years to use for prediction')
    parser.add_argument('--hidden-dim', type=int, default=64, help='size of hidden layers')
    parser.add_argument('--epochs', type=int, default=10, help='number of training iterations')
    parser.add_argument('--no-train', action='store_true', help='skip training phase')
    parser.add_argument('--model-path', default='model.pt', help='file to save/load model')
    parser.add_argument('--batch-size', type=int, default=5, help='number of samples per training batch')

    # parse the command line arguments
    args = parser.parse_args()

    # create directory for output files
    if not os.path.exists(args.output):
        os.makedirs(args.output)

    # set up logging to file and console
    log_file_path = os.path.join(args.output, 'log.txt')

    def log_message(message):
        # get current time for logging
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        # write to log file
        with open(log_file_path, 'a') as log_file:
            log_file.write(f"[{current_time}] {message}\n")

        # also print to console
        print(f"[{current_time}] {message}")

    log_message(f"starting analysis with settings: {args}")

    # process the year range input
    year_parts = args.years.split('-')
    start_year = int(year_parts[0])
    end_year = int(year_parts[1])

    # load the trade data
    log_message(f"loading data from {args.data}")
    trade_data = pd.read_parquet(args.data)

    # filter data to only include selected years
    trade_data = trade_data[(trade_data['year'] >= start_year) & (trade_data['year'] <= end_year)]
    log_message(f"filtered data to years {start_year} through {end_year}")

    # get list of unique years in the data
    unique_years = sorted(trade_data['year'].unique())

    # create dataset object for training
    log_message("building dataset for training")
    dataset = TemporalGraphDataset(trade_data, years=unique_years, window_size=args.window)
    log_message(f"created dataset with {len(dataset)} training samples")

    # get the country and product mappings
    country_to_index = dataset.country_to_idx
    product_to_index = dataset.product_to_idx

    # get total number of countries and products
    total_countries = len(country_to_index)
    total_products = len(product_to_index)
    log_message(f"model will use {total_countries} countries and {total_products} products")

    # initialize the neural network model
    model = TemporalHeteroGNN(total_countries, total_products, hidden_dim=args.hidden_dim)

    # check if we should train or load existing model
    if not args.no_train:
        # split data into training and testing sets
        from sklearn.model_selection import train_test_split
        from torch.utils.data import Subset

        # create list of all sample indices
        all_indices = list(range(len(dataset)))

        # split indices into training and testing
        train_indices, test_indices = train_test_split(all_indices, test_size=0.2, random_state=42)

        # create subset datasets
        training_data = Subset(dataset, train_indices)
        testing_data = Subset(dataset, test_indices)
        log_message(f"split data into {len(training_data)} training and {len(testing_data)} testing samples")

        # train the model
        log_message(f"starting training for {args.epochs} epochs")
        model = train_model(model, training_data, epochs=args.epochs, batch_size=args.batch_size)

        # save the trained model
        model_save_path = os.path.join(args.output, args.model_path)
        torch.save(model.state_dict(), model_save_path)
        log_message(f"saved model to {model_save_path}")

        # evaluate model performance
        log_message("evaluating model on test data")
        auc_score, avg_precision, accuracy = evaluate_model(model, testing_data)
        log_message(f"test results - AUC: {auc_score:.4f}, Average Precision: {avg_precision:.4f}, Accuracy: {accuracy:.4f}")
    else:
        # load existing model
        model_load_path = os.path.join(args.output, args.model_path)
        model.load_state_dict(torch.load(model_load_path))
        model.eval()
        log_message(f"loaded existing model from {model_load_path}")

    # # Run analyses
    # log("Running analyses...")

    # # 1. Product Space Analysis
    # log("Analyzing product space...")
    # latest_year = max(years)
    # latest_graph = yearly_graphs[latest_year]
    # product_embeddings, product_embeddings_2d, product_names = analyze_product_space(
    #     model, latest_graph, product_mapping, df)
    # log("Product space analysis complete")

    # # 2. Product Proximity Analysis
    # log("Calculating product proximity...")
    # proximity_matrix = calculate_product_proximity(model, latest_graph)
    # log("Visualizing product network...")
    # visualize_product_network(proximity_matrix, product_mapping, df, threshold=0.7)
    # log("Product network visualization complete")

    # # 3. Country Development Path Analysis
    # log("Analyzing country development paths...")
    # embeddings_2d_by_year, country_codes = analyze_country_development_paths(
    #     model, yearly_graphs, country_mapping, df)
    # log("Country development analysis complete")

    # # 4. Regional Cluster Analysis
    # log("Analyzing regional clusters...")
    # cluster_analysis, labels, country_embeddings_2d = analyze_regional_clusters(
    #     model, latest_graph, country_mapping, df)
    # log("Regional cluster analysis complete")

    # # 5. Product Adoption Prediction
    # log("Predicting product adoption...")
    # # Get the most recent window of years
    # recent_years = years[-args.window:]
    # input_graphs = [yearly_graphs[year] for year in recent_years]

    # # Predict adoption likelihood
    # country_predictions = predict_product_adoption_likelihood(
    #     model, input_graphs, country_mapping, product_mapping, df)

    # # Save predictions to file
    # import json
    # with open(os.path.join(args.output, 'country_predictions.json'), 'w') as f:
    #     json.dump(country_predictions, f, indent=2)
    # log(f"Saved country predictions to {os.path.join(args.output, 'country_predictions.json')}")

    # # 6. Global Product Potential Analysis
    # log("Analyzing global product potential...")
    # top_products = identify_product_potential(model, input_graphs, country_mapping, product_mapping, df)

    # # Save top products to file
    # with open(os.path.join(args.output, 'top_products.json'), 'w') as f:
    #     json.dump(top_products, f, indent=2)
    # log(f"Saved top products to {os.path.join(args.output, 'top_products.json')}")

    # # Visualize product adoption potential
    # visualize_product_adoption_potential(top_products)
    # log("Product potential visualization complete")

    # # Print summary
    # log("Analysis complete!")
    # log(f"All results saved to {args.output}")

    # # Sample predictions for select countries
    # log("\nSample Predictions:")
    # selected_countries = ['USA', 'CHN', 'DEU', 'KOR', 'THA']
    # for country in selected_countries:
    #     if country in country_predictions:
    #         log(f"\nTop 5 products for {country}:")
    #         for product in country_predictions[country][:5]:
    #             log(f"  SITC {product['sitc_code']}: {product['probability']:.4f} probability")

if __name__ == "__main__":
    main()
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import pickle
import glob
import pyarrow.dataset as ds
import pyarrow as pa

def load_multi_decade_trade_data(data_dir, file_pattern="sitc_country_country_product_year_*.parquet", sample_per_file=None):
    """
    Load multiple SITC trade data files spanning different decades.

    Args:
        data_dir: Directory containing data files
        file_pattern: Pattern to match the parquet files
        sample_per_file: Optional number of rows to randomly sample from each file

    Returns:
        Combined DataFrame containing trade data
    """
    file_paths = glob.glob(os.path.join(data_dir, file_pattern))
    if not file_paths:
        raise FileNotFoundError(f"No files matching '{file_pattern}' found in {data_dir}")

    print(f"Found {len(file_paths)} data files to process")

    dfs = []
    for file_path in tqdm(file_paths, desc="Loading data files"):
        try:
            if sample_per_file:
                dataset = ds.dataset(file_path, format="parquet")
                table = dataset.to_table()
                num_rows = table.num_rows

                if num_rows < sample_per_file:
                    print(f"Warning: {file_path} has only {num_rows} rows; using all")
                    sample_idxs = np.arange(num_rows)
                else:
                    sample_idxs = np.random.choice(num_rows, sample_per_file, replace=False)
                    sample_idxs.sort()

                sample = table.take(pa.array(sample_idxs))
                df = sample.to_pandas()
            else:
                df = pd.read_parquet(file_path)

            df['source_file'] = os.path.basename(file_path)
            dfs.append(df)
            print(f"Loaded {file_path}: {df.shape[0]} rows")

        except Exception as e:
            print(f"Error loading {file_path}: {e}")

    if not dfs:
        raise ValueError("No data was successfully loaded")

    combined_df = pd.concat(dfs, ignore_index=True)

    expected_cols = ['country_id', 'country_iso3_code', 'partner_country_id',
                     'partner_iso3_code', 'product_id', 'product_sitc_code',
                     'year', 'export_value', 'import_value']

    missing_cols = [col for col in expected_cols if col not in combined_df.columns]
    if missing_cols:
        print(f"Warning: Missing expected columns: {missing_cols}")

    extra_cols = [col for col in combined_df.columns if col not in expected_cols + ['source_file']]
    if extra_cols:
        print(f"Note: Additional columns found: {extra_cols}")

    print(f"Combined dataset: {combined_df.shape[0]} rows and {combined_df.shape[1]} columns")
    print(f"Year range: {combined_df['year'].min()} - {combined_df['year'].max()}")

    return combined_df

def clean_data(df):
    """
    Perform basic data cleaning.

    Args:
        df: Raw DataFrame

    Returns:
        Cleaned DataFrame
    """
    # Remove rows with missing values in critical columns
    critical_cols = ['country_iso3_code', 'partner_iso3_code',
                     'year', 'export_value', 'import_value']
    df_clean = df.dropna(subset=critical_cols)

    # Convert year to integer
    df_clean['year'] = df_clean['year'].astype(int)

    # Handle zero or negative trade values
    df_clean = df_clean[(df_clean['export_value'] >= 0) &
                         (df_clean['import_value'] >= 0)]

    print(f"After cleaning: {df_clean.shape[0]} rows remaining")
    return df_clean

def create_yearly_trade_networks(df):
    """
    Create yearly trade networks where countries are nodes and
    trade flows are directed edges with weights.

    Args:
        df: Cleaned DataFrame

    Returns:
        Dictionary of NetworkX graphs by year
    """
    yearly_networks = {}
    years = sorted(df['year'].unique())

    for year in tqdm(years, desc="Creating yearly networks"):
        # Filter data for current year
        year_data = df[df['year'] == year]

        # Create directed graph
        G = nx.DiGraph(year=year)

        # Aggregate trade flows between country pairs
        country_pairs = year_data.groupby(
            ['country_iso3_code', 'partner_iso3_code']
        ).agg({'export_value': 'sum'}).reset_index()

        # Add countries as nodes
        unique_countries = set(country_pairs['country_iso3_code']) | set(country_pairs['partner_iso3_code'])
        G.add_nodes_from(unique_countries)

        # Add trade flows as weighted edges
        for _, row in country_pairs.iterrows():
            exporter = row['country_iso3_code']
            importer = row['partner_iso3_code']
            trade_value = row['export_value']

            if trade_value > 0:
                G.add_edge(exporter, importer, weight=trade_value)

        yearly_networks[year] = G

    return yearly_networks

def calculate_network_metrics(yearly_networks):
    """
    Calculate basic network metrics for each yearly network.

    Args:
        yearly_networks: Dictionary of NetworkX graphs by year

    Returns:
        DataFrame of network metrics
    """
    metrics_data = []

    for year, G in tqdm(yearly_networks.items(), desc="Calculating metrics"):
        metrics = {
            'year': year,
            'num_countries': G.number_of_nodes(),
            'num_trade_flows': G.number_of_edges(),
            'density': nx.density(G),
            'reciprocity': nx.reciprocity(G),
            'avg_clustering': nx.average_clustering(G.to_undirected()),
            'avg_degree': sum(dict(G.degree()).values()) / G.number_of_nodes(),
            'total_trade_volume': sum(nx.get_edge_attributes(G, 'weight').values())
        }

        # Identify top trading countries
        in_degree = {node: val for node, val in G.in_degree(weight='weight')}
        out_degree = {node: val for node, val in G.out_degree(weight='weight')}

        if in_degree:
            metrics['top_importer'] = max(in_degree.items(), key=lambda x: x[1])[0]
            metrics['top_exporter'] = max(out_degree.items(), key=lambda x: x[1])[0]

        metrics_data.append(metrics)

    metrics_df = pd.DataFrame(metrics_data)
    return metrics_df

def visualize_network(G, year, output_dir, top_n=50):
    """
    Visualize the trade network for a specific year.
    Optionally limit to top N nodes for clarity.

    Args:
        G: NetworkX graph
        year: Year of the data
        output_dir: Directory to save visualization
        top_n: Number of top countries to include (by total trade volume)
    """
    # Get total trade volume (in+out) for each country
    trade_volume = {}
    for node in G.nodes():
        in_vol = sum(d.get('weight', 0) for u, v, d in G.in_edges(node, data=True))
        out_vol = sum(d.get('weight', 0) for u, v, d in G.out_edges(node, data=True))
        trade_volume[node] = in_vol + out_vol

    # If requested, filter to top N countries
    if top_n and len(G) > top_n:
        top_countries = sorted(trade_volume.items(), key=lambda x: x[1], reverse=True)[:top_n]
        top_country_set = {country for country, _ in top_countries}
        G = G.subgraph(top_country_set)

    # Create the plot
    plt.figure(figsize=(12, 10))

    # Position nodes using force-directed layout
    pos = nx.spring_layout(G, k=0.3, iterations=50)

    # Get edge weights for sizing
    edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
    max_weight = max(edge_weights) if edge_weights else 1
    scaled_weights = [0.1 + (w / max_weight) * 3 for w in edge_weights]

    # Draw the network
    nx.draw_networkx_nodes(G, pos,
                          node_size=[50 + (trade_volume[node] / max(trade_volume.values()) * 500)
                                    for node in G.nodes()],
                          node_color='skyblue',
                          alpha=0.8)

    nx.draw_networkx_edges(G, pos, width=scaled_weights,
                          edge_color='gray', alpha=0.5,
                          arrows=True, arrowsize=10)

    nx.draw_networkx_labels(G, pos, font_size=8)

    plt.title(f"Global Trade Network {year} (Top {len(G)} Countries)")
    plt.axis('off')

    # Save the visualization
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f"trade_network_{year}.png"), dpi=300, bbox_inches='tight')
    plt.close()

def save_networks(yearly_networks, output_dir):
    """
    Save the yearly networks for future use.

    Args:
        yearly_networks: Dictionary of NetworkX graphs by year
        output_dir: Directory to save the networks
    """
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "yearly_trade_networks.pkl")

    print(f"Saving networks to {output_path}...")
    with open(output_path, 'wb') as f:
        pickle.dump(yearly_networks, f)
    print("Networks saved successfully.")

def create_network_evolution_plots(metrics_df, output_dir):
    """
    Create plots showing the evolution of network metrics over time.

    Args:
        metrics_df: DataFrame with network metrics
        output_dir: Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)

    # Plot key metrics over time
    metrics_to_plot = [
        'num_countries', 'num_trade_flows', 'density',
        'reciprocity', 'avg_clustering', 'total_trade_volume'
    ]

    for metric in metrics_to_plot:
        plt.figure(figsize=(10, 6))
        plt.plot(metrics_df['year'], metrics_df[metric], marker='o', linestyle='-')
        plt.title(f'Evolution of {metric.replace("_", " ").title()} (1962-2024)')
        plt.xlabel('Year')
        plt.ylabel(metric.replace("_", " ").title())
        plt.grid(True, alpha=0.3)

        # Add vertical lines for known economic shocks
        known_shocks = [1973, 1979, 1997, 2008]
        for shock_year in known_shocks:
            if shock_year in metrics_df['year'].values:
                plt.axvline(x=shock_year, color='r', linestyle='--', alpha=0.5)

        plt.savefig(os.path.join(output_dir, f"evolution_{metric}.png"), dpi=300, bbox_inches='tight')
        plt.close()

    # Create combined plot for overview
    plt.figure(figsize=(12, 10))

    # Normalize metrics for comparison
    normalized_metrics = metrics_df.copy()
    for metric in metrics_to_plot:
        max_val = normalized_metrics[metric].max()
        normalized_metrics[f'{metric}_norm'] = normalized_metrics[metric] / max_val

    for metric in metrics_to_plot:
        plt.plot(normalized_metrics['year'], normalized_metrics[f'{metric}_norm'],
                 marker='o', linestyle='-', label=metric.replace("_", " ").title())

    plt.title('Evolution of Trade Network Metrics (Normalized)')
    plt.xlabel('Year')
    plt.ylabel('Normalized Value')
    plt.grid(True, alpha=0.3)
    plt.legend()

    # Add vertical lines for known economic shocks
    for shock_year in known_shocks:
        if shock_year in metrics_df['year'].values:
            plt.axvline(x=shock_year, color='r', linestyle='--', alpha=0.5,
                       label=f'{shock_year} Shock' if shock_year == known_shocks[0] else "")

    plt.savefig(os.path.join(output_dir, "evolution_combined.png"), dpi=300, bbox_inches='tight')
    plt.close()

def sample_balanced_data(df, yearly_sample_size=None):
    """
    Create a balanced sample with equal representation from each year.

    Args:
        df: Full DataFrame
        yearly_sample_size: Number of rows to sample per year (None for all data)

    Returns:
        Balanced sample DataFrame
    """
    if yearly_sample_size is None:
        return df

    years = df['year'].unique()
    samples = []

    for year in tqdm(years, desc="Sampling by year"):
        year_data = df[df['year'] == year]

        # If we have fewer rows than the sample size, take all of them
        if len(year_data) <= yearly_sample_size:
            samples.append(year_data)
        else:
            samples.append(year_data.sample(yearly_sample_size, random_state=42))

    balanced_df = pd.concat(samples, ignore_index=True)
    print(f"Created balanced sample: {balanced_df.shape[0]} rows from {len(years)} years")

    return balanced_df

def get_data_summary(df):
    """
    Generate a summary of the dataset.

    Args:
        df: DataFrame to summarize

    Returns:
        Dictionary with summary statistics
    """
    summary = {
        "total_rows": len(df),
        "year_range": (df['year'].min(), df['year'].max()),
        "num_years": df['year'].nunique(),
        "num_countries": df['country_iso3_code'].nunique(),
        "num_partners": df['partner_iso3_code'].nunique(),
        "num_products": df['product_sitc_code'].nunique(),
        "total_export_value": df['export_value'].sum(),
        "total_import_value": df['import_value'].sum(),
    }

    # Get count of rows per year to see data distribution
    year_counts = df['year'].value_counts().sort_index()
    summary["rows_per_year"] = year_counts.to_dict()

    # Top trading countries by export value
    top_exporters = df.groupby('country_iso3_code')['export_value'].sum().sort_values(ascending=False).head(10)
    summary["top_exporters"] = top_exporters.to_dict()

    # Top products by trade volume
    top_products = df.groupby('product_sitc_code').agg({
        'export_value': 'sum',
        'import_value': 'sum'
    })
    top_products['total_value'] = top_products['export_value'] + top_products['import_value']
    top_products = top_products.sort_values('total_value', ascending=False).head(10)
    summary["top_products"] = top_products['total_value'].to_dict()

    return summary

def main():
    # Define paths
    data_dir = "/root/eel-final-project"  # Update with your actual path
    output_dir = "trade_network_analysis"

    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    viz_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(viz_dir, exist_ok=True)

    # 1. Load data from multiple files
    # For testing, use a smaller sample from each file
    df = load_multi_decade_trade_data(
        data_dir,
        file_pattern="sitc_country_country_product_year_*.parquet",
        sample_per_file=50000  # Adjust based on your memory constraints
    )

    # 2. Get a summary of the loaded data
    summary = get_data_summary(df)

    # Print key statistics
    print(f"Data spans {summary['year_range'][0]} to {summary['year_range'][1]}")
    print(f"Contains {summary['num_countries']} countries and {summary['num_products']} products")

    # Save summary for reference
    with open(os.path.join(output_dir, "data_summary.txt"), 'w') as f:
        for key, value in summary.items():
            f.write(f"{key}: {value}\\n")

    # 3. Clean data
    df_clean = clean_data(df)

    # 4. Create a balanced sample if needed
    # This helps ensure each year has similar representation
    sample_size_per_year = 10000  # Adjust based on your needs
    df_sample = sample_balanced_data(df_clean, yearly_sample_size=sample_size_per_year)

    # Save sample info for reference
    with open(os.path.join(output_dir, "sample_info.txt"), 'w') as f:
        f.write(f"Original data: {len(df_clean)} rows\n")
        f.write(f"Sampled data: {len(df_sample)} rows\n")
        f.write(f"Sample size per year: {sample_size_per_year}\\n")

    # 5. Create yearly trade networks
    yearly_networks = create_yearly_trade_networks(df_sample)
    print(f"Created {len(yearly_networks)} yearly networks")

    # 6. Save networks for future use
    save_networks(yearly_networks, output_dir)

    # 7. Calculate and save network metrics
    metrics_df = calculate_network_metrics(yearly_networks)
    metrics_df.to_csv(os.path.join(output_dir, "network_metrics.csv"), index=False)

    # 8. Create evolution plots
    create_network_evolution_plots(metrics_df, viz_dir)

    # 9. Visualize networks for select years
    # Choose representative years across the full time span
    min_year = metrics_df['year'].min()
    max_year = metrics_df['year'].max()
    num_visualizations = 10  # Adjust based on your needs

    # Select years evenly spaced across the time range
    sample_years = np.linspace(min_year, max_year, num_visualizations, dtype=int)

    for year in sample_years:
        if year in yearly_networks:
            print(f"Visualizing network for year {year}")
            visualize_network(yearly_networks[year], year, viz_dir)

    print("Analysis complete! Results saved to:", output_dir)

if __name__ == "__main__":
    main()
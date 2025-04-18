import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt

def analyze_product_space(model, latest_graph, product_mapping, df):
    """Analyze the product space based on learned embeddings."""
    model.eval()
    with torch.no_grad():
        # Get product embeddings
        x_dict = model.encode_graph(latest_graph)
        product_embeddings = x_dict['product'].numpy()

    # Reduce to 2D for visualization
    from sklearn.manifold import TSNE
    tsne = TSNE(n_components=2, random_state=42)
    product_embeddings_2d = tsne.fit_transform(product_embeddings)

    # Map product indices to SITC codes
    product_id_map = {v: k for k, v in product_mapping.items()}

    # Get product categories (first digit of SITC code indicates category)
    product_categories = {}
    product_names = {}

    for product_idx in range(len(product_embeddings)):
        product_id = product_id_map[product_idx]
        product_row = df[df['product_id'] == product_id].iloc[0]
        sitc_code = product_row['product_sitc_code']

        # Get category from first digit
        category = str(sitc_code)[0]
        product_categories[product_idx] = category
        product_names[product_idx] = f"SITC {sitc_code}"

    # Define colors for different categories
    import matplotlib.colors as mcolors
    colors = list(mcolors.TABLEAU_COLORS)
    category_colors = {str(i): colors[i % len(colors)] for i in range(10)}

    # Plot
    plt.figure(figsize=(14, 12))

    # Create a scatter plot for each category
    for category in set(product_categories.values()):
        indices = [idx for idx, cat in product_categories.items() if cat == category]
        x = [product_embeddings_2d[idx, 0] for idx in indices]
        y = [product_embeddings_2d[idx, 1] for idx in indices]
        plt.scatter(x, y, color=category_colors[category], label=f"Category {category}", alpha=0.7)

    # Add labels for some key products (add selectively to avoid cluttering)
    # Choose products with high diversity (exported by many countries)
    top_product_indices = sorted(range(len(product_embeddings)),
                               key=lambda idx: (product_embeddings[idx] ** 2).sum(),
                               reverse=True)[:30]

    for idx in top_product_indices:
        x, y = product_embeddings_2d[idx]
        plt.annotate(product_names[idx], (x, y), fontsize=8)

    plt.title('Product Space: Learned Embeddings')
    plt.xlabel('t-SNE dimension 1')
    plt.ylabel('t-SNE dimension 2')
    plt.legend()
    plt.tight_layout()
    plt.savefig('product_space.png')
    plt.show()

    return product_embeddings, product_embeddings_2d, product_names

def calculate_product_proximity(model, latest_graph):
    """Calculate proximity between products based on model embeddings."""
    model.eval()
    with torch.no_grad():
        # Get product embeddings
        x_dict = model.encode_graph(latest_graph)
        product_embeddings = x_dict['product']

    # Normalize embeddings
    product_embeddings_norm = F.normalize(product_embeddings, p=2, dim=1)

    # Calculate cosine similarity
    proximity_matrix = torch.mm(product_embeddings_norm, product_embeddings_norm.t())

    return proximity_matrix.numpy()

def visualize_product_network(proximity_matrix, product_mapping, df, threshold=0.7):
    """Visualize the product network based on proximity."""
    import networkx as nx

    # Create graph
    G = nx.Graph()

    # Add nodes
    product_id_map = {v: k for k, v in product_mapping.items()}
    for idx in range(proximity_matrix.shape[0]):
        product_id = product_id_map[idx]
        sitc_code = df[df['product_id'] == product_id]['product_sitc_code'].iloc[0]
        category = str(sitc_code)[0]
        G.add_node(idx, name=f"SITC {sitc_code}", category=category)

    # Add edges for products with proximity > threshold
    for i in range(proximity_matrix.shape[0]):
        for j in range(i+1, proximity_matrix.shape[0]):
            if proximity_matrix[i, j] > threshold:
                G.add_edge(i, j, weight=proximity_matrix[i, j])

    # Remove disconnected nodes
    isolated_nodes = list(nx.isolates(G))
    G.remove_nodes_from(isolated_nodes)

    if len(G.nodes()) == 0:
        print(f"No connections found with threshold {threshold}. Try lowering the threshold.")
        return

    # Position nodes using force-directed layout
    pos = nx.spring_layout(G, k=0.15, iterations=50)

    # Color nodes by category
    node_colors = [int(G.nodes[n]['category']) for n in G.nodes()]

    # Size nodes by degree
    node_sizes = [30 + 10 * G.degree(n) for n in G.nodes()]

    # Plot
    plt.figure(figsize=(16, 16))
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes,
                          cmap=plt.cm.tab10, alpha=0.8)

    # Draw edges with transparency based on weight
    edge_weights = [G[u][v]['weight'] * 2 for u, v in G.edges()]
    nx.draw_networkx_edges(G, pos, width=edge_weights, alpha=0.3)

    # Add labels for high-degree nodes
    high_degree_nodes = [n for n in G.nodes() if G.degree(n) > 5]
    labels = {n: G.nodes[n]['name'] for n in high_degree_nodes}
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=8)

    plt.title('Product Network: Similarity > {}'.format(threshold))
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('product_network.png')
    plt.show()

def analyze_country_development_paths(model, yearly_graphs, country_mapping, df):
    """Analyze how countries' export capabilities evolve over time."""
    # Get years in chronological order
    years = sorted(yearly_graphs.keys())

    # Calculate embeddings for each year
    country_embeddings_by_year = {}

    for year in years:
        graph = yearly_graphs[year]
        model.eval()
        with torch.no_grad():
            x_dict = model.encode_graph(graph)
            if 'country' in x_dict:
                country_embeddings_by_year[year] = x_dict['country'].numpy()

    # Get country codes
    country_id_map = {v: k for k, v in country_mapping.items()}
    country_codes = {}
    for idx, id_ in country_id_map.items():
        try:
            country_codes[idx] = df[df['country_id'] == id_]['country_iso3_code'].iloc[0]
        except IndexError:
            country_codes[idx] = f"Unknown-{id_}"

    # Perform dimensionality reduction across all years at once
    all_embeddings = np.vstack([emb for emb in country_embeddings_by_year.values()])

    from sklearn.manifold import TSNE
    tsne = TSNE(n_components=2, random_state=42)
    all_embeddings_2d = tsne.fit_transform(all_embeddings)

    # Split back by year
    embeddings_2d_by_year = {}
    start_idx = 0
    for year in years:
        if year in country_embeddings_by_year:
            num_countries = country_embeddings_by_year[year].shape[0]
            embeddings_2d_by_year[year] = all_embeddings_2d[start_idx:start_idx+num_countries]
            start_idx += num_countries

    # Plot development paths for selected countries
    selected_countries = ['USA', 'CHN', 'DEU', 'JPN', 'BRA', 'IND', 'KOR', 'THA', 'MEX']
    selected_indices = []

    for country_code in selected_countries:
        try:
            idx = [idx for idx, code in country_codes.items() if code == country_code][0]
            selected_indices.append(idx)
        except IndexError:
            print(f"Country {country_code} not found in the dataset")

    # Plot
    plt.figure(figsize=(16, 12))

    # Plot development paths
    for country_idx in selected_indices:
        country_code = country_codes[country_idx]

        # Get coordinates for each year
        x_coords = []
        y_coords = []
        valid_years = []

        for year in years:
            if year in embeddings_2d_by_year and country_idx < embeddings_2d_by_year[year].shape[0]:
                x, y = embeddings_2d_by_year[year][country_idx]
                x_coords.append(x)
                y_coords.append(y)
                valid_years.append(year)

        # Plot path
        plt.plot(x_coords, y_coords, '-o', label=country_code)

        # Add year labels at specific points
        for i, year in enumerate(valid_years):
            if year % 10 == 0:  # Label every decade
                plt.annotate(str(year), (x_coords[i], y_coords[i]), fontsize=8)

    plt.title('Country Development Paths (1962-2023)')
    plt.xlabel('t-SNE dimension 1')
    plt.ylabel('t-SNE dimension 2')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('country_development_paths.png')
    plt.show()

    return embeddings_2d_by_year, country_codes

def analyze_regional_clusters(model, latest_graph, country_mapping, df):
    """Analyze regional clusters of countries based on export similarity."""
    # Get country embeddings
    model.eval()
    with torch.no_grad():
        x_dict = model.encode_graph(latest_graph)
        country_embeddings = x_dict['country'].numpy()

    # Get country metadata
    country_id_map = {v: k for k, v in country_mapping.items()}
    country_info = {}

    for idx in range(len(country_embeddings)):
        if idx not in country_id_map:
            continue

        country_id = country_id_map[idx]
        try:
            country_row = df[df['country_id'] == country_id].iloc[0]
            country_info[idx] = {
                'code': country_row['country_iso3_code'],
                'id': country_id
            }
        except IndexError:
            continue

    # Perform clustering
    from sklearn.cluster import KMeans

    # Determine optimal number of clusters
    from sklearn.metrics import silhouette_score
    sil_scores = []
    max_clusters = min(10, len(country_embeddings) - 1)

    for k in range(2, max_clusters + 1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(country_embeddings)
        score = silhouette_score(country_embeddings, labels)
        sil_scores.append(score)

    best_k = sil_scores.index(max(sil_scores)) + 2  # +2 because we started from 2

    # Perform final clustering
    kmeans = KMeans(n_clusters=best_k, random_state=42)
    labels = kmeans.fit_predict(country_embeddings)

    # Reduce to 2D for visualization
    from sklearn.manifold import TSNE
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d = tsne.fit_transform(country_embeddings)

    # Plot clusters
    plt.figure(figsize=(14, 10))

    # Define colors for clusters
    import matplotlib.colors as mcolors
    colors = list(mcolors.TABLEAU_COLORS)

    for cluster_id in range(best_k):
        # Get indices of countries in this cluster
        indices = [i for i, label in enumerate(labels) if label == cluster_id and i in country_info]

        if not indices:
            continue

        # Plot points
        x = [embeddings_2d[i, 0] for i in indices]
        y = [embeddings_2d[i, 1] for i in indices]
        plt.scatter(x, y, color=colors[cluster_id % len(colors)],
                   label=f'Cluster {cluster_id+1}', alpha=0.7)

        # Add labels
        for i in indices:
            plt.annotate(country_info[i]['code'], (embeddings_2d[i, 0], embeddings_2d[i, 1]), fontsize=8)

    plt.title(f'Country Clusters Based on Export Similarity ({latest_graph.year})')
    plt.xlabel('t-SNE dimension 1')
    plt.ylabel('t-SNE dimension 2')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('country_clusters.png')
    plt.show()

    # Analyze cluster characteristics
    cluster_analysis = {}

    for cluster_id in range(best_k):
        # Get countries in this cluster
        countries = [country_info[i]['code'] for i in range(len(labels))
                    if labels[i] == cluster_id and i in country_info]

        if not countries:
            continue

        cluster_analysis[cluster_id] = {
            'countries': countries,
            'size': len(countries)
        }

        print(f"\nCluster {cluster_id+1} ({len(countries)} countries):")
        print(", ".join(countries))

    return cluster_analysis, labels, embeddings_2d

def predict_product_adoption_likelihood(model, input_graphs, country_mapping, product_mapping, df):
    """Predict the likelihood of countries adopting new products in the future."""
    # Get the latest graph (to check which edges already exist)
    latest_graph = input_graphs[-1]

    # Get country and product indices
    country_indices = range(latest_graph['country'].num_nodes)
    product_indices = range(latest_graph['product'].num_nodes)

    # Get existing export relationships
    existing_edges = set()
    if ('country', 'exports', 'product') in latest_graph.edge_index_dict:
        edge_index = latest_graph[('country', 'exports', 'product')].edge_index
        existing_edges = set([(edge_index[0, i].item(), edge_index[1, i].item())
                             for i in range(edge_index.shape[1])])

    # Generate predictions for all country-product pairs
    model.eval()
    predictions = {}

    with torch.no_grad():
        # Process in batches to avoid memory issues
        batch_size = 100
        for i in range(0, len(country_indices), batch_size):
            batch_countries = country_indices[i:i+batch_size]

            for country_idx in batch_countries:
                country_predictions = {}

                for product_idx in product_indices:
                    # Skip if the relationship already exists
                    if (country_idx, product_idx) in existing_edges:
                        continue

                    # Predict
                    pred = model(input_graphs, country_idx, product_idx)
                    country_predictions[product_idx] = pred.item()

                predictions[country_idx] = country_predictions

    # Convert indices to actual IDs
    country_id_map = {v: k for k, v in country_mapping.items()}
    product_id_map = {v: k for k, v in product_mapping.items()}

    # Create formatted output
    formatted_predictions = {}

    for country_idx, product_preds in predictions.items():
        if country_idx not in country_id_map:
            continue

        country_id = country_id_map[country_idx]
        try:
            country_code = df[df['country_id'] == country_id]['country_iso3_code'].iloc[0]
        except IndexError:
            country_code = f"Unknown-{country_id}"

        # Get top 10 product predictions
        top_products = sorted(product_preds.keys(), key=lambda p: product_preds[p], reverse=True)[:10]

        product_details = []
        for product_idx in top_products:
            if product_idx not in product_id_map:
                continue

            product_id = product_id_map[product_idx]
            try:
                product_row = df[df['product_id'] == product_id].iloc[0]
                product_code = product_row['product_sitc_code']

                product_details.append({
                    'product_id': product_id,
                    'sitc_code': product_code,
                    'probability': product_preds[product_idx]
                })
            except IndexError:
                continue

        formatted_predictions[country_code] = product_details

    return formatted_predictions

def identify_product_potential(model, input_graphs, country_mapping, product_mapping, df):
    """Identify products with high adoption potential globally."""
    # Get predictions for all country-product pairs
    all_predictions = predict_product_adoption_likelihood(model, input_graphs, country_mapping, product_mapping, df)

    # Aggregate by product
    product_potential = {}

    for country_code, products in all_predictions.items():
        for product in products:
            product_id = product['product_id']
            sitc_code = product['sitc_code']

            if sitc_code not in product_potential:
                product_potential[sitc_code] = {
                    'product_id': product_id,
                    'sitc_code': sitc_code,
                    'countries': [],
                    'avg_probability': 0.0
                }

            product_potential[sitc_code]['countries'].append({
                'country': country_code,
                'probability': product['probability']
            })

    # Calculate average probability
    for sitc_code, data in product_potential.items():
        probabilities = [country['probability'] for country in data['countries']]
        data['avg_probability'] = sum(probabilities) / len(probabilities)
        data['num_countries'] = len(probabilities)

    # Sort products by potential
    top_products = sorted(product_potential.values(), key=lambda p: p['avg_probability'], reverse=True)

    return top_products

def visualize_product_adoption_potential(top_products, num_products=20):
    """Visualize products with highest adoption potential."""
    # Select top N products
    top_n = top_products[:num_products]

    # Prepare data for plotting
    products = [f"SITC {p['sitc_code']}" for p in top_n]
    avg_probs = [p['avg_probability'] for p in top_n]
    num_countries = [p['num_countries'] for p in top_n]

    # Plot
    plt.figure(figsize=(14, 8))

    # Bar chart with color based on number of countries
    bars = plt.bar(products, avg_probs, color=plt.cm.viridis(np.array(num_countries) / max(num_countries)))

    plt.title('Top Products by Global Adoption Potential')
    plt.xlabel('Product SITC Code')
    plt.ylabel('Average Adoption Probability')
    plt.xticks(rotation=90)
    plt.grid(axis='y', alpha=0.3)

    # Add a colorbar
    sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=plt.Normalize(vmin=min(num_countries), vmax=max(num_countries)))
    sm.set_array([])
    cbar = plt.colorbar(sm)
    cbar.set_label('Number of Potential Adopting Countries')

    plt.tight_layout()
    plt.savefig('product_adoption_potential.png')
    plt.show()
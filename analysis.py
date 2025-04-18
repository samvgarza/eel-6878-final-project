import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt

def analyze_product_space(model, latest_graph, product_mapping, df):
    """analyze product space using learned embeddings"""

    # set model to evaluation mode
    model.eval()

    # get product embeddings without tracking gradients
    with torch.no_grad():
        x_dict = model.encode_graph(latest_graph)
        product_embeddings = x_dict['product'].numpy()

    # create tsne object for dimensionality reduction
    from sklearn.manifold import TSNE
    tsne = TSNE(n_components=2, random_state=42)

    # reduce embeddings to 2 dimensions
    product_embeddings_2d = tsne.fit_transform(product_embeddings)

    # create mapping from index to product id
    product_id_map = {}
    for k, v in product_mapping.items():
        product_id_map[v] = k

    # initialize dictionaries for product info
    product_categories = {}
    product_names = {}

    # populate product categories and names
    for product_idx in range(len(product_embeddings)):
        product_id = product_id_map[product_idx]

        # get product data from dataframe
        product_data = df[df['product_id'] == product_id].iloc[0]
        sitc_code = product_data['product_sitc_code']

        # get first digit of sitc code as category
        category = str(sitc_code)[0]
        product_categories[product_idx] = category
        product_names[product_idx] = "sitc " + str(sitc_code)

    # set up colors for plotting
    import matplotlib.colors as mcolors
    colors = list(mcolors.TABLEAU_COLORS)
    category_colors = {}
    for i in range(10):
        category_colors[str(i)] = colors[i % len(colors)]

    # create figure for plotting
    plt.figure(figsize=(14, 12))

    # plot each category separately
    for category in set(product_categories.values()):
        # get indices for this category
        indices = []
        for idx, cat in product_categories.items():
            if cat == category:
                indices.append(idx)

        # get x and y coordinates
        x_coords = []
        y_coords = []
        for idx in indices:
            x_coords.append(product_embeddings_2d[idx, 0])
            y_coords.append(product_embeddings_2d[idx, 1])

        # create scatter plot
        plt.scatter(
            x_coords,
            y_coords,
            color=category_colors[category],
            label="category " + str(category),
            alpha=0.7
        )

    # select top products to label
    top_product_indices = []
    for idx in range(len(product_embeddings)):
        embedding_magnitude = (product_embeddings[idx] ** 2).sum()
        top_product_indices.append((idx, embedding_magnitude))

    # sort by magnitude and take top 30
    top_product_indices.sort(key=lambda x: x[1], reverse=True)
    top_product_indices = [x[0] for x in top_product_indices[:30]]

    # add labels for top products
    for idx in top_product_indices:
        x = product_embeddings_2d[idx, 0]
        y = product_embeddings_2d[idx, 1]
        plt.annotate(product_names[idx], (x, y), fontsize=8)

    # configure plot
    plt.title('product space visualization')
    plt.xlabel('t-sne dimension 1')
    plt.ylabel('t-sne dimension 2')
    plt.legend()
    plt.tight_layout()

    # save and show plot
    plt.savefig('product_space.png')
    plt.show()

    return product_embeddings, product_embeddings_2d, product_names

def calculate_product_proximity(model, latest_graph):
    """calculate similarity between products"""

    # set model to evaluation mode
    model.eval()

    # get embeddings without tracking gradients
    with torch.no_grad():
        x_dict = model.encode_graph(latest_graph)
        product_embeddings = x_dict['product']

    # normalize embeddings
    squared_norms = torch.sum(product_embeddings ** 2, dim=1, keepdim=True)
    norms = torch.sqrt(squared_norms)
    product_embeddings_norm = product_embeddings / norms

    # calculate cosine similarity
    proximity_matrix = torch.mm(product_embeddings_norm, product_embeddings_norm.t())

    return proximity_matrix.numpy()

def visualize_product_network(proximity_matrix, product_mapping, df, threshold=0.7):
    """visualize product similarity network"""

    import networkx as nx

    # create empty graph
    G = nx.Graph()

    # create mapping from index to product id
    product_id_map = {}
    for k, v in product_mapping.items():
        product_id_map[v] = k

    # add nodes to graph
    for idx in range(proximity_matrix.shape[0]):
        product_id = product_id_map[idx]

        # get product data
        product_data = df[df['product_id'] == product_id].iloc[0]
        sitc_code = product_data['product_sitc_code']
        category = str(sitc_code)[0]

        # add node with attributes
        G.add_node(idx, name="sitc " + str(sitc_code), category=category)

    # add edges between similar products
    for i in range(proximity_matrix.shape[0]):
        for j in range(i+1, proximity_matrix.shape[0]):
            if proximity_matrix[i, j] > threshold:
                G.add_edge(i, j, weight=proximity_matrix[i, j])

    # remove isolated nodes
    isolated_nodes = []
    for node in G.nodes():
        if G.degree(node) == 0:
            isolated_nodes.append(node)
    G.remove_nodes_from(isolated_nodes)

    # check if graph has nodes
    if len(G.nodes()) == 0:
        print("no connections found with threshold", threshold)
        print("try lowering the threshold")
        return

    # calculate node positions
    pos = nx.spring_layout(G, k=0.15, iterations=50)

    # prepare node colors and sizes
    node_colors = []
    for n in G.nodes():
        node_colors.append(int(G.nodes[n]['category']))

    node_sizes = []
    for n in G.nodes():
        node_sizes.append(30 + 10 * G.degree(n))

    # create figure
    plt.figure(figsize=(16, 16))

    # draw nodes
    nx.draw_networkx_nodes(
        G,
        pos,
        node_color=node_colors,
        node_size=node_sizes,
        cmap=plt.cm.tab10,
        alpha=0.8
    )

    # draw edges
    edge_weights = []
    for u, v in G.edges():
        edge_weights.append(G[u][v]['weight'] * 2)
    nx.draw_networkx_edges(G, pos, width=edge_weights, alpha=0.3)

    # add labels for important nodes
    high_degree_nodes = []
    for n in G.nodes():
        if G.degree(n) > 5:
            high_degree_nodes.append(n)

    labels = {}
    for n in high_degree_nodes:
        labels[n] = G.nodes[n]['name']

    nx.draw_networkx_labels(G, pos, labels=labels, font_size=8)

    # configure plot
    plt.title('product network similarity > ' + str(threshold))
    plt.axis('off')
    plt.tight_layout()

    # save and show plot
    plt.savefig('product_network.png')
    plt.show()
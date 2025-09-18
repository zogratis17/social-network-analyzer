# src/graph_analysis.py
import pandas as pd
import networkx as nx
from config import DATA_PATH_PROCESSED

def build_graph(csv_file="technology_raw.csv"):
    df = pd.read_csv(f"{DATA_PATH_PROCESSED}{csv_file}")
    G = nx.Graph()
    for _, row in df.iterrows():
        author = row["author"]
        G.add_node(author)
    # Example: connect all authors (can be improved)
    authors = list(df["author"])
    for i in range(len(authors)):
        for j in range(i + 1, len(authors)):
            G.add_edge(authors[i], authors[j])
    return G

def detect_communities(G):
    # Simple community detection using connected components
    communities = [list(c) for c in nx.connected_components(G)]
    return communities

if __name__ == "__main__":
    G = build_graph()
    communities = detect_communities(G)
    print("Communities:", communities)

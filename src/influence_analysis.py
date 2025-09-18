# src/influence_analysis.py
import networkx as nx
from .graph_analysis import build_graph

from config import PAGE_RANK_ALPHA, MAX_ITER

def calculate_pagerank(csv_file="technology_raw.csv"):
    G = build_graph(csv_file)
    pr = nx.pagerank(G, alpha=PAGE_RANK_ALPHA, max_iter=MAX_ITER)
    return pr

if __name__ == "__main__":
    pr_scores = calculate_pagerank()
    sorted_nodes = sorted(pr_scores.items(), key=lambda x: x[1], reverse=True)
    print("Top 5 influential nodes:", sorted_nodes[:5])

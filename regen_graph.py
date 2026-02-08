#!/usr/bin/env python3
"""Quick script to regenerate network visualization with new colors"""
import json
import sys
import networkx as nx
from ai_sn_analysis_prototype import visualize_graph_plotly

subreddit = sys.argv[1] if len(sys.argv) > 1 else "samsung"
print(f"Regenerating {subreddit} graph visualization...")

with open(f"output/{subreddit}_graph.json", 'r', encoding='utf-8') as f:
    G = nx.node_link_graph(json.load(f))

visualize_graph_plotly(G, out_html=f"output/{subreddit}_graph.html")
print(f"✅ Done! Open output/{subreddit}_graph.html to view")

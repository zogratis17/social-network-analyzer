# dashboard/app.py
import streamlit as st
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import sys
import os

# Fix imports for src folder
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.graph_analysis import build_graph, detect_communities
from src.influence_analysis import calculate_pagerank
from src.ai_analysis import analyze_content_with_gemini, detect_trending_content
from config import DATA_PATH_PROCESSED, TREND_THRESHOLD

st.title("AI-Enhanced Social Network Analysis Dashboard")

uploaded_file = st.file_uploader("Upload Processed CSV", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("Data Preview")
    st.dataframe(df.head())

    # Build graph
    G = build_graph(uploaded_file.name)
    communities = detect_communities(G)
    st.subheader("Communities")
    st.write(communities)

    # Influential nodes
    pr_scores = calculate_pagerank(uploaded_file.name)
    top_nodes = sorted(pr_scores.items(), key=lambda x: x[1], reverse=True)[:5]
    st.subheader("Top Influential Nodes")
    st.write(top_nodes)

    # Trending posts (mock)
    posts = df["title"].tolist()
    trending = detect_trending_content(posts, TREND_THRESHOLD)
    st.subheader("Trending Posts")
    st.write(trending)

    # Visualize graph
    if st.button("Show Network Graph"):
        st.subheader("Network Graph")
        plt.figure(figsize=(10, 6))
        nx.draw(G, with_labels=True, node_color="skyblue", edge_color="gray", node_size=500)
        st.pyplot(plt)

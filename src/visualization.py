import matplotlib.pyplot as plt
import networkx as nx
from config import GRAPH_NODE_COLOR, GRAPH_EDGE_COLOR, FIGURE_SIZE

def plot_graph(G, title="Network Graph"):
    plt.figure(figsize=FIGURE_SIZE)
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_color=GRAPH_NODE_COLOR, edge_color=GRAPH_EDGE_COLOR)
    plt.title(title)
    plt.show()

if __name__ == "__main__":
    G = nx.karate_club_graph()
    plot_graph(G, "Karate Club Example")

import pandas as pd
import networkx as nx
import plotly.graph_objects as go
import matplotlib
import ast  # for safely parsing string-lists

# --- Step 1: Load CSVs ---
nodes_df = pd.read_csv("python_nodes.csv")
edges_df = pd.read_csv("python_edges.csv")

# --- Step 2: Build Graph ---
G = nx.Graph()

# Add nodes with attributes
for _, row in nodes_df.iterrows():
    G.add_node(row['user'], 
               pagerank=float(row['pagerank']),  # ensure numeric
               community=row['community_greedy'])

# Add edges with numeric interactions (count of list items)
for _, row in edges_df.iterrows():
    try:
        interactions_list = ast.literal_eval(row['interactions'])  # "['reply']" -> list
        interactions_count = len(interactions_list)
    except:
        interactions_count = 1  # fallback if parsing fails
    G.add_edge(row['source'], row['target'], interactions=interactions_count)

# --- Step 3: Layout ---
pos = nx.spring_layout(G, seed=42)  # force-directed layout

# --- Step 4: Node colors and sizes ---
communities = list(set(nx.get_node_attributes(G, 'community').values()))
cmap = matplotlib.colormaps['tab20']
colors_list = [matplotlib.colors.to_hex(cmap(i/len(communities))) for i in range(len(communities))]
community_color_map = {c: colors_list[i] for i, c in enumerate(communities)}

node_x = []
node_y = []
node_color = []
node_size = []

for node in G.nodes(data=True):
    x, y = pos[node[0]]
    node_x.append(x)
    node_y.append(y)
    node_color.append(community_color_map[node[1]['community']])
    node_size.append(max(5, node[1]['pagerank']*100))  # adjust scaling as needed

# --- Step 5: Edge traces ---
edge_x = []
edge_y = []
edge_width = []

for u, v, data in G.edges(data=True):
    x0, y0 = pos[u]
    x1, y1 = pos[v]
    edge_x.extend([x0, x1, None])
    edge_y.extend([y0, y1, None])
    edge_width.append(max(0.5, data['interactions']/2))  # scale thickness; adjust divisor for visuals

edge_trace = go.Scatter(
    x=edge_x, y=edge_y,
    line=dict(width=1, color='#888'),
    hoverinfo='none',
    mode='lines'
)

# --- Step 6: Node traces ---
node_trace = go.Scatter(
    x=node_x, y=node_y,
    mode='markers+text',
    text=list(G.nodes()),
    textposition="top center",
    hoverinfo='text',
    marker=dict(
        showscale=False,
        color=node_color,
        size=node_size,
        line_width=2
    )
)

# --- Step 7: Build figure ---
fig = go.Figure(data=[edge_trace, node_trace],
                layout=go.Layout(
                    title='AI Social Media Network',
                    title_x=0.5,
                    showlegend=False,
                    hovermode='closest',
                    margin=dict(b=20,l=5,r=5,t=40),
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                ))

# --- Step 8: Export to HTML ---
fig.write_html("python_graph.html")
print("python_graph.html generated successfully!")

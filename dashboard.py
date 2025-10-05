"""
Intelligent Social Network Analysis Dashboard
A comprehensive platform for network analysis with AI-powered insights
"""

import streamlit as st
import pandas as pd
import json
import os
import plotly.graph_objects as go
import plotly.express as px
import networkx as nx
from datetime import datetime
import sys
import time
from pathlib import Path

# Import the analysis module
from ai_sn_analysis_prototype import (
    RedditCollector,
    GraphBuilder,
    GeminiClient,
    detect_communities_union_find,
    detect_communities_louvain,
    compute_influence_pagerank,
    compute_mst,
    detect_trends,
    visualize_graph_plotly
)

# Page configuration
st.set_page_config(
    page_title="AI Social Network Analyzer",
    page_icon="🕸️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        font-size: 1.1rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="main-header">🕸️ AI Social Network Analyzer</div>', unsafe_allow_html=True)
st.markdown("---")

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/000000/neural-network.png", width=80)
    st.title("🎛️ Control Panel")
    
    # Analysis Mode Selection
    analysis_mode = st.radio(
        "📊 Mode",
        ["🔍 New Analysis", "📂 Load Existing"],
        help="Choose to analyze a new subreddit or load previous results"
    )
    
    if analysis_mode == "🔍 New Analysis":
        st.markdown("### 🎯 Analysis Configuration")
        
        # Subreddit input
        subreddit_name = st.text_input(
            "🔤 Subreddit Name",
            value="python",
            help="Enter subreddit name without 'r/' prefix"
        )
        
        # Number of posts
        num_posts = st.slider(
            "📝 Number of Posts",
            min_value=10,
            max_value=500,
            value=50,
            step=10,
            help="More posts = better insights but slower analysis"
        )
        
        # Time filter
        time_filter = st.selectbox(
            "⏰ Time Period",
            ["all", "year", "month", "week", "day"],
            index=0,
            help="Filter posts by time period"
        )
        
        # API credentials check
        st.markdown("---")
        st.markdown("### 🔑 API Status")
        
        reddit_id = os.getenv('REDDIT_CLIENT_ID')
        reddit_secret = os.getenv('REDDIT_CLIENT_SECRET')
        gemini_key = os.getenv('GEMINI_API_KEY')
        
        if reddit_id and reddit_secret:
            st.success("✅ Reddit API configured")
        else:
            st.error("❌ Reddit API not configured")
            st.info("Add credentials to .env file")
        
        if gemini_key:
            st.success("✅ Gemini AI configured")
        else:
            st.warning("⚠️ Gemini AI not configured (will use local analysis)")
        
        st.markdown("---")
        
        # Analysis button
        analyze_button = st.button(
            "🚀 Start Analysis",
            type="primary",
            use_container_width=True,
            disabled=not (reddit_id and reddit_secret)
        )
        
        selected_subreddit = None
        output_dir = "output"
        
    else:  # Load Existing mode
        st.markdown("### 📂 Existing Analyses")
        
        # Data source selection
        output_dir = st.text_input("📁 Output Directory", "output")
        
        # Available analysis files
        if os.path.exists(output_dir):
            files = os.listdir(output_dir)
            subreddits = list(set([f.split('_')[0] for f in files if f.endswith('.json')]))
            
            if subreddits:
                selected_subreddit = st.selectbox("📊 Select Analysis", subreddits)
            else:
                st.warning("No analysis found. Run an analysis first!")
                selected_subreddit = None
        else:
            st.error(f"Directory '{output_dir}' not found!")
            selected_subreddit = None
        
        analyze_button = False
    
    st.markdown("---")
    st.markdown("### 🔬 Features")
    st.markdown("""
    - 👥 Community Detection
    - 📈 Influence Analysis  
    - 🔥 Trend Prediction
    - 🎨 Network Visualization
    - 🧠 AI Content Analysis
    - 📊 User Behavior Analytics
    """)
    
    st.markdown("---")
    st.markdown("### 📖 About")
    st.info("""
    An intelligent platform combining graph algorithms with AI for 
    social network analysis.
    
    **Algorithms:**
    - Union-Find (Communities)
    - PageRank (Influence)
    - MST (Optimization)
    - Gemini AI (Content)
    """)

# Helper function to generate community names
def get_community_name(community_id, nodes_df, community_col, max_name_length=30):
    """
    Generate a meaningful name for a community based on its most influential members.
    Returns: "Community {id}: {top_user1}, {top_user2}, ... ({size} members)"
    """
    # Get all users in this community
    community_members = nodes_df[nodes_df[community_col] == community_id]
    
    if len(community_members) == 0:
        return f"Community {community_id} (0 members)"
    
    # Sort by PageRank if available, else by degree
    if 'pagerank' in community_members.columns:
        top_members = community_members.nlargest(3, 'pagerank')['user'].tolist()
    elif 'degree' in community_members.columns:
        top_members = community_members.nlargest(3, 'degree')['user'].tolist()
    else:
        top_members = community_members['user'].head(3).tolist()
    
    # Create name with top members
    member_str = ", ".join(top_members[:2])  # Show top 2 members
    if len(member_str) > max_name_length:
        member_str = member_str[:max_name_length-3] + "..."
    
    size = len(community_members)
    return f"Comm-{community_id}: {member_str} ({size})"

# Main content
if selected_subreddit:
    # Load data files
    @st.cache_data
    def load_data(subreddit, output_dir):
        data = {}
        try:
            # Load graph
            with open(f"{output_dir}/{subreddit}_graph.json", 'r') as f:
                data['graph'] = json.load(f)
            
            # Load trends
            with open(f"{output_dir}/{subreddit}_trends.json", 'r') as f:
                data['trends'] = json.load(f)
            
            # Load content analysis
            with open(f"{output_dir}/{subreddit}_content_analysis.json", 'r') as f:
                data['content'] = json.load(f)
            
            # Load nodes and edges
            data['nodes'] = pd.read_csv(f"{output_dir}/{subreddit}_nodes.csv")
            data['edges'] = pd.read_csv(f"{output_dir}/{subreddit}_edges.csv")
            
            # Load raw posts
            with open(f"{output_dir}/{subreddit}_raw_posts.json", 'r') as f:
                data['posts'] = json.load(f)
                
        except FileNotFoundError as e:
            st.error(f"Missing file: {e}")
            return None
        
        return data
    
    data = load_data(selected_subreddit, output_dir)
    
    if data:
        # Create tabs
        tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
            "📊 Overview", 
            "👥 Communities", 
            "⭐ Influencers", 
            "🔥 Trending Topics",
            "🕸️ Network Graph",
            "🧠 AI Insights",
            "📈 Analytics"
        ])
        
        # TAB 1: OVERVIEW
        with tab1:
            st.header("📊 Network Overview")
            
            # Key metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    label="Total Users",
                    value=f"{len(data['nodes']):,}",
                    delta="Active Nodes"
                )
            
            with col2:
                st.metric(
                    label="Interactions",
                    value=f"{len(data['edges']):,}",
                    delta="Total Edges"
                )
            
            with col3:
                num_communities = data['nodes']['community_greedy'].nunique() if 'community_greedy' in data['nodes'].columns else 0
                st.metric(
                    label="Communities",
                    value=f"{num_communities:,}",
                    delta="Detected"
                )
            
            with col4:
                st.metric(
                    label="Posts Analyzed",
                    value=f"{len(data['posts']):,}",
                    delta="r/" + selected_subreddit
                )
            
            st.markdown("---")
            
            # Network statistics
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("🔢 Network Statistics")
                
                # Calculate metrics
                avg_degree = data['edges']['weight'].mean() if 'weight' in data['edges'].columns else 0
                density = (len(data['edges']) * 2) / (len(data['nodes']) * (len(data['nodes']) - 1)) if len(data['nodes']) > 1 else 0
                
                stats_df = pd.DataFrame({
                    'Metric': [
                        'Average Degree',
                        'Network Density',
                        'Average Edge Weight',
                        'Max PageRank',
                        'Total Communities (UF)',
                        'Total Communities (Greedy)'
                    ],
                    'Value': [
                        f"{(len(data['edges']) / len(data['nodes'])) * 2:.2f}",
                        f"{density:.4f}",
                        f"{avg_degree:.2f}",
                        f"{data['nodes']['pagerank'].max():.6f}" if 'pagerank' in data['nodes'].columns else "N/A",
                        str(data['nodes']['community_uf'].nunique()) if 'community_uf' in data['nodes'].columns else "N/A",
                        str(num_communities)
                    ]
                })
                
                st.dataframe(stats_df, width='stretch', hide_index=True)
            
            with col2:
                st.subheader("📈 Degree Distribution")
                
                # Degree distribution
                if 'degree' in data['nodes'].columns:
                    fig = px.histogram(
                        data['nodes'], 
                        x='degree',
                        nbins=30,
                        title="User Connection Distribution",
                        labels={'degree': 'Number of Connections', 'count': 'Number of Users'}
                    )
                    fig.update_traces(showlegend=False)
                    st.plotly_chart(fig, width='stretch', config={'displayModeBar': False})
                else:
                    st.info("Degree data not available")
            
            # Recent posts
            st.markdown("---")
            st.subheader("📝 Recent Posts Overview")
            
            posts_df = pd.DataFrame([
                {
                    'Title': p['title'][:60] + '...' if len(p['title']) > 60 else p['title'],
                    'Author': p['author'] or '[deleted]',
                    'Score': p['score'],
                    'Comments': p['num_comments']
                }
                for p in data['posts'][:10]
            ])
            
            st.dataframe(posts_df, width='stretch', hide_index=True)
        
        # TAB 2: COMMUNITIES
        with tab2:
            st.header("👥 Community Detection Analysis")
            
            st.markdown("""
            Communities are detected using two advanced algorithms:
            - **Union-Find**: Fast connected component detection
            - **Greedy Modularity**: Optimized community structure
            """)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("🔵 Union-Find Communities")
                if 'community_uf' in data['nodes'].columns:
                    # Get community sizes and sort by size (descending)
                    community_sizes_uf = data['nodes']['community_uf'].value_counts().sort_values(ascending=False).head(10)
                    
                    # Debug info
                    total_uf = data['nodes']['community_uf'].nunique()
                    
                    if len(community_sizes_uf) > 0:
                        # Create community names
                        community_names = [get_community_name(cid, data['nodes'], 'community_uf', max_name_length=25) 
                                         for cid in community_sizes_uf.index]
                        
                        # Create bar chart with community names
                        fig = px.bar(
                            x=community_names,
                            y=community_sizes_uf.values,
                            labels={'x': 'Community', 'y': 'Members'},
                            title=f"Top {len(community_sizes_uf)} Largest Communities (Union-Find)",
                            color=community_sizes_uf.values,
                            color_continuous_scale='Blues'
                        )
                        fig.update_layout(
                            xaxis_tickangle=-45,
                            showlegend=False,
                            height=400
                        )
                        st.plotly_chart(fig, width='stretch', config={'displayModeBar': False})
                    else:
                        st.warning("No Union-Find communities detected")
                    
                    st.metric("Total Communities", f"{total_uf:,}")
            
            with col2:
                st.subheader("🟢 Greedy Modularity Communities")
                if 'community_greedy' in data['nodes'].columns:
                    community_sizes_greedy = data['nodes']['community_greedy'].value_counts().head(10)
                    
                    # Create community names
                    community_names = [get_community_name(cid, data['nodes'], 'community_greedy', max_name_length=25) 
                                     for cid in community_sizes_greedy.index]
                    
                    # Create bar chart with community names
                    fig = px.bar(
                        x=community_names,
                        y=community_sizes_greedy.values,
                        labels={'x': 'Community', 'y': 'Members'},
                        title="Top 10 Largest Communities (Greedy Modularity)",
                        color=community_sizes_greedy.values,
                        color_continuous_scale='Greens'
                    )
                    fig.update_layout(
                        xaxis_tickangle=-45,
                        showlegend=False,
                        height=400
                    )
                    st.plotly_chart(fig, width='stretch', config={'displayModeBar': False})
                    
                    total_greedy = data['nodes']['community_greedy'].nunique()
                    st.metric("Total Communities", f"{total_greedy:,}")
            
            # Detailed community insights
            st.markdown("---")
            st.subheader("� Algorithm Comparison")
            
            comparison_data = {
                'Metric': ['Total Communities', 'Avg Community Size', 'Largest Community', 'Smallest Community'],
                'Union-Find': [
                    str(data['nodes']['community_uf'].nunique()) if 'community_uf' in data['nodes'].columns else 'N/A',
                    f"{len(data['nodes']) / data['nodes']['community_uf'].nunique():.1f}" if 'community_uf' in data['nodes'].columns else 'N/A',
                    str(data['nodes']['community_uf'].value_counts().max()) if 'community_uf' in data['nodes'].columns else 'N/A',
                    str(data['nodes']['community_uf'].value_counts().min()) if 'community_uf' in data['nodes'].columns else 'N/A'
                ],
                'Greedy Modularity': [
                    str(data['nodes']['community_greedy'].nunique()) if 'community_greedy' in data['nodes'].columns else 'N/A',
                    f"{len(data['nodes']) / data['nodes']['community_greedy'].nunique():.1f}" if 'community_greedy' in data['nodes'].columns else 'N/A',
                    str(data['nodes']['community_greedy'].value_counts().max()) if 'community_greedy' in data['nodes'].columns else 'N/A',
                    str(data['nodes']['community_greedy'].value_counts().min()) if 'community_greedy' in data['nodes'].columns else 'N/A'
                ]
            }
            st.dataframe(pd.DataFrame(comparison_data), width='stretch', hide_index=True)
            
            # Community comparison
            st.markdown("---")
            st.subheader("📊 Community Size Distribution")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if 'community_uf' in data['nodes'].columns:
                    sizes = data['nodes']['community_uf'].value_counts()
                    size_dist = pd.DataFrame({
                        'Size Range': ['1-10', '11-50', '51-100', '101-500', '500+'],
                        'Count': [
                            len(sizes[sizes <= 10]),
                            len(sizes[(sizes > 10) & (sizes <= 50)]),
                            len(sizes[(sizes > 50) & (sizes <= 100)]),
                            len(sizes[(sizes > 100) & (sizes <= 500)]),
                            len(sizes[sizes > 500])
                        ]
                    })
                    
                    fig = px.pie(
                        size_dist, 
                        values='Count', 
                        names='Size Range',
                        title="Community Size Distribution (Union-Find)",
                        hole=0.3
                    )
                    fig.update_traces(textposition='inside', textinfo='percent+label')
                    st.plotly_chart(fig, width='stretch', config={'displayModeBar': False})
            
            with col2:
                if 'community_greedy' in data['nodes'].columns:
                    sizes = data['nodes']['community_greedy'].value_counts()
                    size_dist = pd.DataFrame({
                        'Size Range': ['1-10', '11-50', '51-100', '101-500', '500+'],
                        'Count': [
                            len(sizes[sizes <= 10]),
                            len(sizes[(sizes > 10) & (sizes <= 50)]),
                            len(sizes[(sizes > 50) & (sizes <= 100)]),
                            len(sizes[(sizes > 100) & (sizes <= 500)]),
                            len(sizes[sizes > 500])
                        ]
                    })
                    
                    fig = px.pie(
                        size_dist, 
                        values='Count', 
                        names='Size Range',
                        title="Community Size Distribution (Greedy)",
                        hole=0.3
                    )
                    fig.update_traces(textposition='inside', textinfo='percent+label')
                    st.plotly_chart(fig, width='stretch', config={'displayModeBar': False})
        
        # TAB 3: INFLUENCERS
        with tab3:
            st.header("⭐ Influence Analysis - Top Users")
            
            st.markdown("""
            Influence is measured using multiple centrality metrics:
            - **PageRank**: Overall influence in the network
            - **Betweenness**: Bridge between communities
            - **Closeness**: Central position in network
            - **Degree**: Direct connections
            """)
            
            # Top influencers by PageRank
            st.subheader("🏆 Top Influencers by PageRank")
            
            if 'pagerank' in data['nodes'].columns:
                top_influencers = data['nodes'].nlargest(20, 'pagerank')[[
                    'user', 'pagerank', 'degree', 'betweenness', 'closeness',
                    'community_greedy', 'posts', 'comments'
                ]].copy()
                
                top_influencers.columns = [
                    'User', 'PageRank', 'Connections', 'Betweenness', 
                    'Closeness', 'Community', 'Posts', 'Comments'
                ]
                
                # Format numeric columns
                # Format numeric columns as strings for display
                top_influencers['PageRank'] = top_influencers['PageRank'].apply(lambda x: f"{x:.6f}")
                top_influencers['Betweenness'] = top_influencers['Betweenness'].apply(lambda x: f"{x:.4f}")
                top_influencers['Closeness'] = top_influencers['Closeness'].apply(lambda x: f"{x:.4f}")
                top_influencers['Community'] = top_influencers['Community'].astype(str)
                
                st.dataframe(top_influencers, width='stretch', hide_index=True)
                
                # Visualization
                col1, col2 = st.columns(2)
                
                with col1:
                    top_10 = data['nodes'].nlargest(10, 'pagerank')
                    fig = px.bar(
                        top_10,
                        x='user',
                        y='pagerank',
                        title="Top 10 Users by PageRank",
                        labels={'user': 'User', 'pagerank': 'PageRank Score'},
                        color='pagerank',
                        color_continuous_scale='Viridis'
                    )
                    fig.update_layout(xaxis_tickangle=-45)

                    fig.update_traces(showlegend=False)
                    st.plotly_chart(fig, width='stretch', config={'displayModeBar': False})
                
                with col2:
                    fig = px.scatter(
                        data['nodes'].head(100),
                        x='degree',
                        y='pagerank',
                        size='betweenness',
                        hover_data=['user'],
                        title="Influence Metrics Correlation",
                        labels={'degree': 'Connections', 'pagerank': 'PageRank'},
                        color='betweenness',
                        color_continuous_scale='Plasma'
                    )
                    fig.update_traces(showlegend=False)
                    st.plotly_chart(fig, width='stretch', config={'displayModeBar': False})
            
            # User activity analysis
            st.markdown("---")
            st.subheader("📊 User Activity Patterns")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if 'posts' in data['nodes'].columns and 'comments' in data['nodes'].columns:
                    active_users = data['nodes'].nlargest(15, 'posts')[['user', 'posts', 'comments']]
                    
                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        name='Posts', 
                        x=active_users['user'], 
                        y=active_users['posts'],
                        marker_color='rgb(55, 83, 109)'
                    ))
                    fig.add_trace(go.Bar(
                        name='Comments', 
                        x=active_users['user'], 
                        y=active_users['comments'],
                        marker_color='rgb(26, 118, 255)'
                    ))
                    fig.update_layout(
                        title="Most Active Users (Posts & Comments)",
                        xaxis_tickangle=-45,
                        barmode='group',
                        xaxis_title="User",
                        yaxis_title="Count",
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                    )
                    st.plotly_chart(fig, width='stretch', config={'displayModeBar': False})
            
            with col2:
                if 'pagerank' in data['nodes'].columns and 'degree' in data['nodes'].columns:
                    # Categorize users
                    data['nodes']['influence_level'] = pd.cut(
                        data['nodes']['pagerank'],
                        bins=4,
                        labels=['Low', 'Medium', 'High', 'Very High']
                    )
                    
                    influence_dist = data['nodes']['influence_level'].value_counts()
                    
                    fig = px.pie(
                        values=influence_dist.values,
                        names=influence_dist.index,
                        title="Influence Distribution",
                        hole=0.3,
                        color_discrete_sequence=px.colors.sequential.RdBu
                    )
                    fig.update_traces(textposition='inside', textinfo='percent+label')
                    st.plotly_chart(fig, width='stretch', config={'displayModeBar': False})
        
        # TAB 4: TRENDING TOPICS
        with tab4:
            st.header("🔥 Trending Topics & Content Analysis")
            
            st.markdown("""
            Topics are extracted using advanced NLP and AI analysis:
            - **Gemini AI**: Contextual understanding
            - **TF-IDF**: Statistical relevance
            - **N-grams**: Multi-word phrases
            - **Velocity**: Recent vs historical activity
            """)
            
            # Top trending topics
            st.subheader("📈 Top Trending Topics")
            
            trends_df = pd.DataFrame([
                {
                    'Rank': i + 1,
                    'Topic': topic,
                    'Total Mentions': data['total'],
                    'Recent (7d)': data['recent'],
                    'Historical': data['older'],
                    'Velocity': f"{data['velocity']:.2f}",
                    'Importance': f"{data['importance']:.2f}"
                }
                for i, (topic, data) in enumerate(data['trends'][:15])
            ])
            
            st.dataframe(trends_df, width='stretch', hide_index=True)
            
            # Visualizations
            col1, col2 = st.columns(2)
            
            with col1:
                top_topics = trends_df.head(10)
                # Convert string back to float for plotting
                top_topics_plot = top_topics.copy()
                top_topics_plot['Importance'] = pd.to_numeric(top_topics_plot['Importance'])
                top_topics_plot['Velocity'] = pd.to_numeric(top_topics_plot['Velocity'])
                
                fig = px.bar(
                    top_topics_plot,
                    x='Topic',
                    y='Importance',
                    color='Velocity',
                    title="Top 10 Topics by Importance Score",
                    color_continuous_scale='Viridis',
                    labels={'Importance': 'Importance Score', 'Velocity': 'Growth Rate'}
                )
                fig.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig, width='stretch', config={'displayModeBar': False})
            
            with col2:
                # Convert for plotting
                trends_plot = trends_df.copy()
                trends_plot['Velocity'] = pd.to_numeric(trends_plot['Velocity'])
                trends_plot['Importance'] = pd.to_numeric(trends_plot['Importance'])
                
                fig = px.scatter(
                    trends_plot,
                    x='Total Mentions',
                    y='Velocity',
                    size='Importance',
                    hover_data=['Topic'],
                    title="Topic Momentum Analysis",
                    labels={'Total Mentions': 'Frequency', 'Velocity': 'Growth Rate'},
                    color='Importance',
                    color_continuous_scale='Turbo'
                )
                st.plotly_chart(fig, width='stretch', config={'displayModeBar': False})
            
            # Topic categories
            st.markdown("---")
            st.subheader("🏷️ Topic Categories")
            
            # Simple categorization
            tech_keywords = {'api', 'library', 'framework', 'package', 'module', 'tool'}
            ml_keywords = {'machine learning', 'deep learning', 'neural', 'model', 'ai', 'data science'}
            web_keywords = {'web', 'server', 'api', 'http', 'rest', 'websocket'}
            automation_keywords = {'automation', 'bot', 'scraping', 'crawler'}
            
            categories = {'Technical Tools': 0, 'ML/AI': 0, 'Web Development': 0, 'Automation': 0, 'Other': 0}
            
            for topic, _ in data['trends']:
                topic_lower = topic.lower()
                if any(kw in topic_lower for kw in ml_keywords):
                    categories['ML/AI'] += 1
                elif any(kw in topic_lower for kw in automation_keywords):
                    categories['Automation'] += 1
                elif any(kw in topic_lower for kw in web_keywords):
                    categories['Web Development'] += 1
                elif any(kw in topic_lower for kw in tech_keywords):
                    categories['Technical Tools'] += 1
                else:
                    categories['Other'] += 1
            
            fig = px.pie(
                values=list(categories.values()),
                names=list(categories.keys()),
                title="Topic Categories Distribution",
                hole=0.4,
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            fig.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig, width='stretch', config={'displayModeBar': False})
        
        # TAB 5: NETWORK GRAPH
        with tab5:
            st.header("🕸️ Interactive Network Visualization")
            
            st.markdown("""
            Explore the network structure interactively:
            - **Node Size**: Represents PageRank (influence)
            - **Node Color**: Represents community membership
            - **Hover**: View user details
            """)
            
            # Load and display the HTML graph
            html_path = f"{output_dir}/{selected_subreddit}_graph.html"
            
            if os.path.exists(html_path):
                with open(html_path, 'r', encoding='utf-8') as f:
                    html_content = f.read()
                
                st.components.v1.html(html_content, height=800, scrolling=True)
            else:
                st.warning("Interactive graph HTML not found. Generating simplified view...")
                
                # Create a simplified graph view
                G = nx.node_link_graph(data['graph'])
                
                # Sample for performance
                if len(G.nodes()) > 100:
                    st.info("Sampling 100 nodes for performance")
                    top_nodes = data['nodes'].nlargest(100, 'pagerank')['user'].tolist()
                    G = G.subgraph(top_nodes)
                
                pos = nx.spring_layout(G, k=0.5, iterations=50)
                
                edge_trace = go.Scatter(
                    x=[], y=[], mode='lines',
                    line=dict(width=0.5, color='#888'),
                    hoverinfo='none'
                )
                
                for edge in G.edges():
                    x0, y0 = pos[edge[0]]
                    x1, y1 = pos[edge[1]]
                    edge_trace['x'] += tuple([x0, x1, None])
                    edge_trace['y'] += tuple([y0, y1, None])
                
                node_trace = go.Scatter(
                    x=[], y=[], mode='markers+text',
                    hoverinfo='text',
                    marker=dict(
                        showscale=True,
                        colorscale='Viridis',
                        size=10,
                        colorbar=dict(title="Community")
                    )
                )
                
                for node in G.nodes():
                    x, y = pos[node]
                    node_trace['x'] += tuple([x])
                    node_trace['y'] += tuple([y])
                
                fig = go.Figure(data=[edge_trace, node_trace],
                               layout=go.Layout(
                                   showlegend=False,
                                   hovermode='closest',
                                   margin=dict(b=0,l=0,r=0,t=0),
                                   xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                   yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                               ))
                
                st.plotly_chart(fig, width='stretch', config={'displayModeBar': False})
        
        # TAB 6: AI INSIGHTS
        with tab6:
            st.header("🧠 AI-Powered Content Insights")
            
            st.markdown("""
            Content analyzed using Google Gemini AI for:
            - **Sentiment Analysis**: Positive, Neutral, Negative
            - **Topic Extraction**: Key themes and subjects
            - **Viral Prediction**: Content engagement potential
            """)
            
            # Sentiment distribution
            st.subheader("😊 Sentiment Analysis")
            
            sentiments = [v['sentiment'] for v in data['content'].values()]
            sentiment_counts = pd.Series(sentiments).value_counts()
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                fig = px.pie(
                    values=sentiment_counts.values,
                    names=sentiment_counts.index,
                    title="Overall Sentiment Distribution",
                    color_discrete_map={
                        'positive': '#00CC96',
                        'neutral': '#FFA15A',
                        'negative': '#EF553B'
                    },
                    hole=0.3
                )
                fig.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(fig, width='stretch', config={'displayModeBar': False})
            
            with col2:
                # Sentiment scores
                scores = [v['score'] for v in data['content'].values()]
                fig = px.histogram(
                    x=scores,
                    nbins=20,
                    title="Sentiment Score Distribution",
                    labels={'x': 'Sentiment Score (0=Negative, 1=Positive)', 'y': 'Count'},
                    color_discrete_sequence=['#636EFA']
                )
                fig.update_traces(showlegend=False)
                st.plotly_chart(fig, width='stretch', config={'displayModeBar': False})
            
            # Viral content prediction
            st.markdown("---")
            st.subheader("🚀 Viral Content Prediction")
            
            viral_scores = [(k, v['viral_score'], v['sentiment']) for k, v in data['content'].items()]
            viral_scores.sort(key=lambda x: x[1], reverse=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Top viral posts
                top_viral = []
                for post_id, viral_score, sentiment in viral_scores[:10]:
                    post = next((p for p in data['posts'] if p['id'] == post_id), None)
                    if post:
                        top_viral.append({
                            'Title': post['title'][:50] + '...',
                            'Viral Score': f"{viral_score:.3f}",
                            'Sentiment': sentiment,
                            'Actual Score': post['score']
                        })
                
                st.dataframe(pd.DataFrame(top_viral), width='stretch', hide_index=True)
            
            with col2:
                viral_scores_list = [v['viral_score'] for v in data['content'].values()]
                fig = px.box(
                    y=viral_scores_list,
                    title="Viral Score Distribution",
                    labels={'y': 'Viral Potential (0-1)'},
                    color_discrete_sequence=['#AB63FA']
                )
                fig.update_traces(showlegend=False)
                st.plotly_chart(fig, width='stretch', config={'displayModeBar': False})
            
            # Content topics overview
            st.markdown("---")
            st.subheader("📚 Content Topics Overview")
            
            all_topics = []
            for v in data['content'].values():
                all_topics.extend(v.get('topics', []))
            
            topic_freq = pd.Series(all_topics).value_counts().head(20)
            
            fig = px.bar(
                x=topic_freq.index,
                y=topic_freq.values,
                title="Most Common Topics in Content",
                labels={'x': 'Topic', 'y': 'Frequency'},
                color=topic_freq.values,
                color_continuous_scale='Teal'
            )
            fig.update_layout(xaxis_tickangle=-45)

            fig.update_traces(showlegend=False)
            st.plotly_chart(fig, width='stretch', config={'displayModeBar': False})
        
        # TAB 7: ANALYTICS
        with tab7:
            st.header("📈 Advanced Analytics & Insights")
            
            # Network health metrics
            st.subheader("💪 Network Health Metrics")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # Activity rate
                total_interactions = len(data['edges'])
                total_users = len(data['nodes'])
                activity_rate = (total_interactions / total_users) if total_users > 0 else 0
                
                st.metric(
                    "Activity Rate",
                    f"{activity_rate:.2f}",
                    "interactions/user"
                )
            
            with col2:
                # Average sentiment
                avg_sentiment = sum(v['score'] for v in data['content'].values()) / len(data['content']) if data['content'] else 0
                sentiment_label = "Positive" if avg_sentiment > 0.6 else "Negative" if avg_sentiment < 0.4 else "Neutral"
                
                st.metric(
                    "Average Sentiment",
                    sentiment_label,
                    f"{avg_sentiment:.2f}"
                )
            
            with col3:
                # Engagement quality
                avg_comments = sum(p['num_comments'] for p in data['posts']) / len(data['posts']) if data['posts'] else 0
                
                st.metric(
                    "Avg Comments/Post",
                    f"{avg_comments:.1f}",
                    "engagement"
                )
            
            # Time series (if we had timestamps)
            st.markdown("---")
            st.subheader("📊 Key Insights")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 🎯 Top Findings")
                st.markdown(f"""
                - **Largest Community**: {data['nodes']['community_greedy'].value_counts().iloc[0] if len(data['nodes']) > 0 else 0} members
                - **Most Influential User**: {data['nodes'].nlargest(1, 'pagerank')['user'].iloc[0] if len(data['nodes']) > 0 else 'N/A'}
                - **Hottest Topic**: {data['trends'][0][0] if data['trends'] else 'N/A'}
                - **Network Cohesion**: {'High' if len(data['nodes']) / num_communities < 50 else 'Medium' if len(data['nodes']) / num_communities < 200 else 'Low'}
                """)
            
            with col2:
                st.markdown("### 💡 Recommendations")
                
                # Generate recommendations based on analysis
                recommendations = []
                
                if num_communities > len(data['nodes']) * 0.1:
                    recommendations.append("🔸 High fragmentation detected - consider community engagement initiatives")
                
                if avg_sentiment < 0.5:
                    recommendations.append("🔸 Overall sentiment is low - monitor for potential issues")
                
                if activity_rate < 2:
                    recommendations.append("🔸 Low interaction rate - encourage more community participation")
                
                if len(recommendations) == 0:
                    recommendations.append("✅ Network appears healthy and active")
                
                for rec in recommendations:
                    st.markdown(rec)
            
            # Export options
            st.markdown("---")
            st.subheader("📥 Export Data")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                csv_nodes = data['nodes'].to_csv(index=False)
                st.download_button(
                    label="📄 Download Nodes CSV",
                    data=csv_nodes,
                    file_name=f"{selected_subreddit}_nodes.csv",
                    mime="text/csv"
                )
            
            with col2:
                csv_edges = data['edges'].to_csv(index=False)
                st.download_button(
                    label="📄 Download Edges CSV",
                    data=csv_edges,
                    file_name=f"{selected_subreddit}_edges.csv",
                    mime="text/csv"
                )
            
            with col3:
                trends_json = json.dumps(data['trends'], indent=2)
                st.download_button(
                    label="📄 Download Trends JSON",
                    data=trends_json,
                    file_name=f"{selected_subreddit}_trends.json",
                    mime="application/json"
                )

else:
    # No data available
    st.info("👈 Please select an analysis from the sidebar or run a new analysis")
    
    st.markdown("---")
    st.markdown("### 🚀 Get Started")
    st.markdown("""
    1. Run an analysis using the CLI:
       ```bash
       python ai_sn_analysis_prototype.py --subreddit python --posts 100
       ```
    
    2. Launch this dashboard:
       ```bash
       streamlit run dashboard.py
       ```
    
    3. Select your analysis from the sidebar
    
    4. Explore the insights!
    """)
    
    st.markdown("---")
    st.markdown("### 🔬 What This Platform Does")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Graph Algorithms:**
        - Union-Find for community detection
        - PageRank for influence measurement
        - Minimum Spanning Trees for optimization
        - Centrality metrics (betweenness, closeness)
        """)
    
    with col2:
        st.markdown("""
        **AI-Powered Analysis:**
        - Google Gemini for content understanding
        - Sentiment analysis
        - Topic extraction
        - Viral content prediction
        """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 1rem;'>
    <p>🕸️ <strong>AI Social Network Analyzer</strong> | Built with NetworkX, Gemini AI & Streamlit</p>
    <p>Advanced Graph Algorithms + AI-Powered Content Intelligence</p>
</div>
""", unsafe_allow_html=True)

"""
Intelligent Social Network Analysis Dashboard
A comprehensive platform for network analysis with AI-powered insights
"""

import os
import sys
import warnings

# Suppress warnings before other imports
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
os.environ['GRPC_VERBOSITY'] = 'ERROR'
os.environ['GLOG_minloglevel'] = '2'

import streamlit as st
import pandas as pd
import json
import plotly.graph_objects as go
import plotly.express as px
import networkx as nx
from datetime import datetime
import time
from pathlib import Path
import logging

# Suppress library logging
logging.getLogger("plotly").setLevel(logging.CRITICAL)
logging.getLogger("google").setLevel(logging.ERROR)
logging.getLogger("grpc").setLevel(logging.ERROR)
logging.getLogger("absl").setLevel(logging.ERROR)

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Import the analysis module
from ai_sn_analysis_prototype import (
    RedditCollector,
    GraphBuilder,
    GeminiClient,
    detect_communities_union_find,
    detect_communities_louvain,
    compute_pagerank,
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
            max_value=200,
            value=50,
            step=10,
            help="More posts = better insights but slower analysis"
        )
        
        # Performance warnings
        if num_posts > 100:
            st.warning("⚠️ >100 posts may take 3-5 minutes. Consider reducing for faster results.")
        
        # Time filter
        time_filter = st.selectbox(
            "⏰ Time Period",
            ["all", "year", "month", "week", "day"],
            index=0,
            help="Filter posts by time period"
        )
        
        # AI Analysis toggle
        use_gemini = st.checkbox(
            "🤖 Use Gemini AI for content analysis",
            value=False,
            help="⚠️ Free tier limited to 15 requests/min. Smart prioritization enabled."
        )
        
        if use_gemini:
            st.warning("🔥 **Free Tier Rate Limit: 15 requests/min**")
            
            max_ai_posts = st.slider(
                "🎯 Max AI-analyzed posts (rest use fast local)",
                min_value=5,
                max_value=50,
                value=20,
                step=5,
                help="Smart algorithm picks most important posts for AI analysis"
            )
            
            # Save to session state for use in analysis
            st.session_state['max_ai_requests'] = max_ai_posts
            
            # Calculate accurate time estimate
            ai_minutes = max_ai_posts / 14  # 14 RPM (safety margin under 15)
            st.info(f"⏱️ Estimated AI time: ~{ai_minutes:.1f} minutes for {max_ai_posts} posts + instant local for others")
        else:
            st.info("⚡ Using fast local analysis (instant, good enough for most cases)")
            st.session_state['max_ai_requests'] = 0
        
        # Advanced Analytics toggle
        st.markdown("---")
        st.markdown("### 🔬 Advanced Analytics")
        
        enable_advanced = st.checkbox(
            "🚀 Enable Advanced Features",
            value=False,
            help="Includes: Temporal Evolution, Sentiment Networks, ML Predictions, Validation Metrics"
        )
        
        if enable_advanced:
            st.success("✅ Advanced analytics enabled (+30-40 seconds)")
            
            # Data Append Mode
            append_mode = st.selectbox(
                "📊 Data Append Strategy",
                ["auto", "always", "never"],
                index=0,
                help="Auto: Append if >7 days old | Always: Cumulative | Never: Fresh"
            )
            
            append_help = {
                "auto": "🔄 Smart merge (appends if last analysis >7 days old)",
                "always": "📈 Cumulative analysis (always merge with existing data)",
                "never": "🆕 Fresh analysis (replace all existing data)"
            }
            st.info(append_help[append_mode])
        else:
            st.info("⚡ Standard analysis only (faster)")
            append_mode = "never"
        
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

# ========== REAL-TIME ANALYSIS EXECUTION ==========
def run_analysis(subreddit, num_posts, time_filter, output_dir, use_gemini_ai=False, enable_advanced=True, append_mode="auto"):
    """
    Run complete social network analysis pipeline with progress tracking.
    
    Args:
        use_gemini_ai: If True, use Gemini API for content analysis (slower but better).
                       If False, use local analysis (much faster).
        enable_advanced: If True, run advanced analytics (temporal, sentiment, ML, validation).
        append_mode: 'auto', 'always', or 'never' - determines data merging strategy.
    """
    try:
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Step 1: Initialize clients (10%)
        status_text.text("🔧 Initializing API clients...")
        progress_bar.progress(10)
        
        reddit_id = os.getenv('REDDIT_CLIENT_ID')
        reddit_secret = os.getenv('REDDIT_CLIENT_SECRET')
        reddit_agent = os.getenv('REDDIT_USER_AGENT', 'ai-sn-analysis/1.0')
        gemini_key = os.getenv('GEMINI_API_KEY')
        
        collector = RedditCollector(reddit_id, reddit_secret, reddit_agent)
        gemini_client = GeminiClient(gemini_key) if use_gemini_ai else GeminiClient(None)
        
        # Step 2: Fetch posts (30%)
        status_text.text(f"📥 Fetching {num_posts} posts from r/{subreddit}...")
        progress_bar.progress(20)
        
        posts = collector.fetch_subreddit_posts(subreddit, limit=num_posts, time_filter=time_filter)
        
        if not posts:
            st.error(f"❌ No posts found for r/{subreddit}. Check if the subreddit exists and is accessible.")
            progress_bar.empty()
            status_text.empty()
            return False
        
        progress_bar.progress(30)
        status_text.text(f"✅ Fetched {len(posts)} posts with {sum(len(p.get('comments', [])) for p in posts)} comments")
        
        # Save raw posts
        with open(f"{output_dir}/{subreddit}_raw_posts.json", 'w', encoding='utf-8') as f:
            json.dump(posts, f, indent=2, ensure_ascii=False)
        
        # Step 3: Build graph (40%)
        status_text.text("🕸️ Building social network graph...")
        progress_bar.progress(35)
        
        builder = GraphBuilder()
        G = builder.build_from_posts(posts)
        
        if G.number_of_nodes() == 0:
            st.error("❌ No users found in the posts. The subreddit might be too small or have no interactions.")
            progress_bar.empty()
            status_text.empty()
            return False
        
        progress_bar.progress(40)
        status_text.text(f"✅ Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        
        # Step 4: Community detection (50%)
        status_text.text("👥 Detecting communities...")
        progress_bar.progress(45)
        
        detect_communities_union_find(G)
        detect_communities_louvain(G)
        
        progress_bar.progress(50)
        status_text.text("✅ Communities detected")
        
        # Step 5: Influence analysis (60%)
        status_text.text("⭐ Computing influence scores...")
        progress_bar.progress(55)
        
        compute_pagerank(G)
        
        progress_bar.progress(60)
        status_text.text("✅ PageRank computed")
        
        # Step 6: MST computation (65%)
        status_text.text("🌳 Computing minimum spanning tree...")
        progress_bar.progress(62)
        
        mst = compute_mst(G)
        
        # Clean MST for GraphML export (remove list/dict attributes)
        mst_clean = mst.copy()
        for node in mst_clean.nodes():
            attrs_to_remove = []
            for key, value in mst_clean.nodes[node].items():
                if isinstance(value, (list, dict)):
                    attrs_to_remove.append(key)
            for key in attrs_to_remove:
                del mst_clean.nodes[node][key]
        
        for u, v in mst_clean.edges():
            attrs_to_remove = []
            for key, value in mst_clean[u][v].items():
                if isinstance(value, (list, dict)):
                    attrs_to_remove.append(key)
            for key in attrs_to_remove:
                del mst_clean[u][v][key]
        
        nx.write_graphml(mst_clean, f"{output_dir}/{subreddit}_mst.graphml")
        
        progress_bar.progress(65)
        
        # Step 7: Content analysis (80%)
        if use_gemini_ai:
            status_text.text("🧠 Running AI content analysis (this may take a while)...")
        else:
            status_text.text("⚡ Running fast local content analysis...")
        progress_bar.progress(68)
        
        # Import smart analysis
        from ai_sn_analysis_prototype import smart_gemini_analysis
        
        # Get max AI requests from session state
        max_ai_requests = st.session_state.get('max_ai_requests', 20)
        
        # Progress callback for real-time updates
        def update_progress(current, total_count, message):
            progress = 68 + int((current / total_count) * 12)
            progress_bar.progress(min(progress, 80))
            status_text.text(message)
        
        # Use smart analysis with prioritization and rate limiting
        content_analysis = smart_gemini_analysis(
            posts,
            gemini_client,
            use_gemini=use_gemini_ai,
            max_requests=max_ai_requests,
            progress_callback=update_progress
        )
        
        progress_bar.progress(80)
        status_text.text("✅ Content analysis complete")
        
        # Save content analysis
        with open(f"{output_dir}/{subreddit}_content_analysis.json", 'w', encoding='utf-8') as f:
            json.dump(content_analysis, f, indent=2, ensure_ascii=False)
        
        # Step 8: Trend detection (90%)
        status_text.text("🔥 Detecting trending topics...")
        progress_bar.progress(85)
        
        trends = detect_trends(posts, content_analysis)
        
        with open(f"{output_dir}/{subreddit}_trends.json", 'w', encoding='utf-8') as f:
            json.dump(trends, f, indent=2, ensure_ascii=False)
        
        progress_bar.progress(90)
        status_text.text("✅ Trends detected")
        
        # Step 8.5: Advanced Analytics (if enabled)
        if enable_advanced:
            try:
                from advanced_analytics import (
                    validate_influence_metrics,
                    build_sentiment_weighted_graphs,
                    build_temporal_graphs,
                    detect_community_evolution,
                    build_multilayer_network,
                    train_viral_predictor
                )
                
                status_text.text("🔬 Running advanced analytics...")
                progress_bar.progress(91)
                
                # Validation metrics
                try:
                    validation_results = validate_influence_metrics(G, posts)
                    with open(f"{output_dir}/{subreddit}_validation.json", 'w', encoding='utf-8') as f:
                        json.dump(validation_results, f, indent=2)
                except Exception as e:
                    print(f"⚠️ Validation metrics failed: {e}")
                
                # Sentiment networks
                try:
                    sentiment_graphs = build_sentiment_weighted_graphs(posts, content_analysis)
                    with open(f"{output_dir}/{subreddit}_sentiment_networks.json", 'w', encoding='utf-8') as f:
                        json.dump({
                            'positive': nx.node_link_data(sentiment_graphs['positive']),
                            'negative': nx.node_link_data(sentiment_graphs['negative']),
                            'neutral': nx.node_link_data(sentiment_graphs['neutral']),
                            'stats': sentiment_graphs['stats']
                        }, f, indent=2)
                except Exception as e:
                    print(f"⚠️ Sentiment networks failed: {e}")
                
                # Temporal evolution
                try:
                    temporal_data = build_temporal_graphs(posts)
                    evolution_data = detect_community_evolution(temporal_data['graphs'])
                    with open(f"{output_dir}/{subreddit}_evolution.json", 'w', encoding='utf-8') as f:
                        json.dump({
                            'temporal': {k: nx.node_link_data(v) for k, v in temporal_data['graphs'].items()},
                            'timeline': temporal_data['timeline'],
                            'evolution': evolution_data
                        }, f, indent=2)
                except Exception as e:
                    print(f"⚠️ Temporal evolution failed: {e}")
                
                # Multi-layer network
                try:
                    layers = build_multilayer_network(posts)
                    with open(f"{output_dir}/{subreddit}_layers.json", 'w', encoding='utf-8') as f:
                        json.dump({
                            'reply': nx.node_link_data(layers['reply']),
                            'mention': nx.node_link_data(layers['mention']),
                            'topic': nx.node_link_data(layers['topic'])
                        }, f, indent=2)
                except Exception as e:
                    print(f"⚠️ Multi-layer network failed: {e}")
                
                # ML viral predictor
                try:
                    predictor_results = train_viral_predictor(posts, G, content_analysis)
                    with open(f"{output_dir}/{subreddit}_predictor.json", 'w', encoding='utf-8') as f:
                        json.dump(predictor_results, f, indent=2)
                except Exception as e:
                    print(f"⚠️ ML predictor failed: {e}")
                
                status_text.text("✅ Advanced analytics complete")
                
            except Exception as e:
                st.warning(f"⚠️ Advanced analytics partially failed: {str(e)}")
        
        # Step 9: Export data (95%)
        status_text.text("💾 Exporting results...")
        progress_bar.progress(92)
        
        # Export graph data (JSON - supports all types)
        graph_data = nx.node_link_data(G)
        with open(f"{output_dir}/{subreddit}_graph.json", 'w', encoding='utf-8') as f:
            json.dump(graph_data, f, indent=2)
        
        # Clean graph for GraphML export (remove list/dict attributes)
        G_clean = G.copy()
        for node in G_clean.nodes():
            attrs_to_remove = []
            for key, value in G_clean.nodes[node].items():
                if isinstance(value, (list, dict)):
                    attrs_to_remove.append(key)
            for key in attrs_to_remove:
                del G_clean.nodes[node][key]
        
        for u, v in G_clean.edges():
            attrs_to_remove = []
            for key, value in G_clean[u][v].items():
                if isinstance(value, (list, dict)):
                    attrs_to_remove.append(key)
            for key in attrs_to_remove:
                del G_clean[u][v][key]
        
        nx.write_graphml(G_clean, f"{output_dir}/{subreddit}_graph.graphml")
        
        # Export nodes and edges CSV
        nodes_data = []
        for n in G.nodes():
            node_attr = G.nodes[n]
            nodes_data.append({
                'user': n,
                'degree': G.degree(n),
                'pagerank': node_attr.get('pagerank', 0),
                'community_uf': node_attr.get('community_uf', -1),
                'community_greedy': node_attr.get('community_greedy', -1),
                'posts': node_attr.get('posts', 0),
                'comments': node_attr.get('comments', 0)
            })
        pd.DataFrame(nodes_data).to_csv(f"{output_dir}/{subreddit}_nodes.csv", index=False)
        
        edges_data = []
        for u, v, data in G.edges(data=True):
            edges_data.append({
                'source': u,
                'target': v,
                'weight': data.get('weight', 1),
                'interaction': data.get('interaction', 'unknown')
            })
        pd.DataFrame(edges_data).to_csv(f"{output_dir}/{subreddit}_edges.csv", index=False)
        
        progress_bar.progress(95)
        
        # Step 10: Generate visualization (100%)
        status_text.text("🎨 Generating interactive visualization...")
        progress_bar.progress(97)
        
        visualize_graph_plotly(G, out_html=f"{output_dir}/{subreddit}_graph.html")
        
        progress_bar.progress(100)
        status_text.text("✅ Analysis complete!")
        
        time.sleep(1)
        progress_bar.empty()
        status_text.empty()
        
        return True
        
    except Exception as e:
        st.error(f"❌ Analysis failed: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
        return False

# Execute analysis if button clicked
if analyze_button:
    st.markdown("---")
    st.markdown(f"## 🔬 Analyzing r/{subreddit_name}")
    
    with st.spinner("Running analysis..."):
        success = run_analysis(subreddit_name, num_posts, time_filter, output_dir, use_gemini, enable_advanced, append_mode)
    
    if success:
        st.success(f"✅ Analysis of r/{subreddit_name} completed successfully!")
        st.balloons()
        selected_subreddit = subreddit_name
        
        # Prompt to switch to load mode
        st.info("💡 Switch to 'Load Existing' mode in sidebar to view the results")
        st.stop()
    else:
        st.stop()

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
        """Load analysis results with better error handling."""
        data = {}
        required_files = [
            ('graph', f"{output_dir}/{subreddit}_graph.json"),
            ('trends', f"{output_dir}/{subreddit}_trends.json"),
            ('content', f"{output_dir}/{subreddit}_content_analysis.json"),
            ('nodes', f"{output_dir}/{subreddit}_nodes.csv"),
            ('edges', f"{output_dir}/{subreddit}_edges.csv"),
            ('posts', f"{output_dir}/{subreddit}_raw_posts.json")
        ]
        
        missing_files = []
        
        for key, filepath in required_files:
            if not os.path.exists(filepath):
                missing_files.append(os.path.basename(filepath))
        
        if missing_files:
            st.error(f"❌ Missing analysis files for r/{subreddit}:")
            for file in missing_files:
                st.write(f"   - {file}")
            
            st.info("💡 **Solution:** Run a new analysis for this subreddit:")
            st.code(f"""
1. Switch to "🔍 New Analysis" mode in sidebar
2. Enter "{subreddit}" as subreddit name
3. Click "🚀 Start Analysis"
            """)
            return None
        
        try:
            # Load graph
            with open(f"{output_dir}/{subreddit}_graph.json", 'r', encoding='utf-8') as f:
                data['graph'] = json.load(f)
            
            # Load trends
            with open(f"{output_dir}/{subreddit}_trends.json", 'r', encoding='utf-8') as f:
                data['trends'] = json.load(f)
            
            # Load content analysis
            with open(f"{output_dir}/{subreddit}_content_analysis.json", 'r', encoding='utf-8') as f:
                data['content'] = json.load(f)
            
            # Load nodes and edges
            data['nodes'] = pd.read_csv(f"{output_dir}/{subreddit}_nodes.csv")
            data['edges'] = pd.read_csv(f"{output_dir}/{subreddit}_edges.csv")
            
            # Load raw posts
            with open(f"{output_dir}/{subreddit}_raw_posts.json", 'r', encoding='utf-8') as f:
                data['posts'] = json.load(f)
            
            st.success(f"✅ Loaded analysis for r/{subreddit}")
            return data
                
        except Exception as e:
            st.error(f"❌ Error loading data: {str(e)}")
            import traceback
            with st.expander("Show error details"):
                st.code(traceback.format_exc())
            with st.expander("Show error details"):
                st.code(traceback.format_exc())
            return None
    
    # Only load data if a subreddit is selected
    if selected_subreddit:
        data = load_data(selected_subreddit, output_dir)
    else:
        data = None
        st.info("👈 Select an analysis from the sidebar or run a new analysis to get started!")
        st.markdown("""
        ### 🚀 Getting Started
        
        **Option 1: Analyze a New Subreddit**
        1. Switch to "🔍 New Analysis" mode in sidebar
        2. Enter a subreddit name (e.g., "python", "machinelearning")
        3. Choose number of posts and time filter
        4. Click "🚀 Start Analysis"
        
        **Option 2: Load Existing Results**
        1. Make sure you're in "📂 Load Existing" mode
        2. Select a subreddit from the dropdown
        3. View all the analysis results
        
        ---
        
        ### 📚 Example Subreddits to Try:
        - **r/python** - Python programming community
        - **r/machinelearning** - ML and AI discussions
        - **r/datascience** - Data science community
        - **r/learnprogramming** - Beginner-friendly programming
        - **r/technology** - Tech news and discussions
        """)
    
    if data:
        # Create tabs
        tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10, tab11 = st.tabs([
            "📊 Overview", 
            "👥 Communities", 
            "⭐ Influencers", 
            "🔥 Trending Topics",
            "🕸️ Network Graph",
            "🧠 AI Insights",
            "📈 Analytics",
            "⏰ Temporal Evolution",
            "✅ Validation Metrics",
            "🔀 Sentiment Networks",
            "🎯 ML Predictions"
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
            Influence is measured using centrality metrics:
            - **PageRank**: Overall influence in the network (Google's algorithm)
            - **Degree**: Direct connections with other users
            - **Posts & Comments**: Contribution activity levels
            """)
            
            # Top influencers by PageRank
            st.subheader("🏆 Top Influencers by PageRank")
            
            if 'pagerank' in data['nodes'].columns:
                # Get available columns
                available_cols = ['user', 'pagerank', 'degree']
                optional_cols = ['community_greedy', 'posts', 'comments']
                
                # Add optional columns if they exist
                for col in optional_cols:
                    if col in data['nodes'].columns:
                        available_cols.append(col)
                
                top_influencers = data['nodes'].nlargest(20, 'pagerank')[available_cols].copy()
                
                # Create column name mapping
                col_mapping = {
                    'user': 'User',
                    'pagerank': 'PageRank',
                    'degree': 'Connections',
                    'community_greedy': 'Community',
                    'posts': 'Posts',
                    'comments': 'Comments'
                }
                
                # Rename only available columns
                top_influencers.columns = [col_mapping.get(c, c) for c in top_influencers.columns]
                
                # Format numeric columns (only if they exist)
                if 'PageRank' in top_influencers.columns:
                    top_influencers['PageRank'] = top_influencers['PageRank'].apply(lambda x: f"{x:.6f}")
                if 'Community' in top_influencers.columns:
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
                        size='degree',
                        hover_data=['user'],
                        title="Influence Metrics Correlation",
                        labels={'degree': 'Connections', 'pagerank': 'PageRank'},
                        color='pagerank',
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
        
        # TAB 8: TEMPORAL EVOLUTION
        with tab8:
            st.header("⏰ Temporal Evolution Analysis")
            
            # Load temporal evolution data
            evolution_file = f"{output_dir}/{selected_subreddit}_evolution.json"
            
            if os.path.exists(evolution_file):
                with open(evolution_file, 'r') as f:
                    evolution_data = json.load(f)
                
                st.subheader("📈 Community Growth Over Time")
                
                # Timeline metrics
                if 'timeline' in evolution_data and evolution_data['timeline']:
                    timeline_df = pd.DataFrame(evolution_data['timeline'])
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Growth chart
                        fig_growth = px.line(
                            timeline_df,
                            x='period',
                            y='nodes',
                            title='Network Growth',
                            labels={'nodes': 'Number of Users', 'period': 'Time Period'}
                        )
                        st.plotly_chart(fig_growth, use_container_width=True)
                    
                    with col2:
                        # Engagement chart
                        fig_engagement = px.line(
                            timeline_df,
                            x='period',
                            y='edges',
                            title='Interaction Growth',
                            labels={'edges': 'Number of Interactions', 'period': 'Time Period'}
                        )
                        st.plotly_chart(fig_engagement, use_container_width=True)
                
                # Evolution metrics
                if 'evolution' in evolution_data:
                    st.subheader("🔄 Community Evolution")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        stability = evolution_data['evolution'].get('stability', 0)
                        st.metric(
                            "Community Stability",
                            f"{stability:.1%}",
                            "Consistency score"
                        )
                    
                    with col2:
                        churn = evolution_data['evolution'].get('churn_rate', 0)
                        st.metric(
                            "User Churn Rate",
                            f"{churn:.1%}",
                            "Lost users"
                        )
                    
                    with col3:
                        new_users = evolution_data['evolution'].get('new_users', 0)
                        st.metric(
                            "New Users",
                            f"{new_users}",
                            "Recent period"
                        )
                    
                    # Changes over time
                    if 'changes' in evolution_data['evolution']:
                        st.markdown("### 📊 Community Changes")
                        changes_df = pd.DataFrame(evolution_data['evolution']['changes'])
                        st.dataframe(changes_df, use_container_width=True)
            else:
                st.info("⚠️ Temporal evolution data not available. Enable advanced analytics when running analysis.")
        
        # TAB 9: VALIDATION METRICS
        with tab9:
            st.header("✅ Validation & Accuracy Metrics")
            
            # Load validation data
            validation_file = f"{output_dir}/{selected_subreddit}_validation.json"
            
            if os.path.exists(validation_file):
                with open(validation_file, 'r') as f:
                    validation_data = json.load(f)
                
                st.subheader("🎯 PageRank Accuracy Assessment")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    correlation = validation_data.get('correlation', {}).get('spearman', 0)
                    accuracy_level = "High" if correlation > 0.7 else "Moderate" if correlation > 0.4 else "Low"
                    st.metric(
                        "Correlation Score",
                        f"{correlation:.3f}",
                        accuracy_level
                    )
                
                with col2:
                    pearson = validation_data.get('correlation', {}).get('pearson', 0)
                    st.metric(
                        "Pearson Correlation",
                        f"{pearson:.3f}",
                        "Linear relationship"
                    )
                
                with col3:
                    accuracy = validation_data.get('accuracy', 'Unknown')
                    st.metric(
                        "Overall Accuracy",
                        accuracy,
                        "PageRank vs Engagement"
                    )
                
                # Scatter plot: PageRank vs Actual Engagement
                if 'top_influencers' in validation_data:
                    st.subheader("📊 PageRank vs Actual Engagement")
                    
                    influencers_df = pd.DataFrame(validation_data['top_influencers'])
                    
                    fig_validation = px.scatter(
                        influencers_df,
                        x='pagerank',
                        y='actual_engagement',
                        size='actual_engagement',
                        hover_data=['user'],
                        title='PageRank Accuracy Validation',
                        labels={
                            'pagerank': 'Predicted Influence (PageRank)',
                            'actual_engagement': 'Actual Engagement (Posts + Comments)'
                        }
                    )
                    st.plotly_chart(fig_validation, use_container_width=True)
                    
                    st.markdown("### 🔬 Interpretation")
                    st.markdown(f"""
                    - **Correlation: {correlation:.3f}** - {accuracy_level} accuracy
                    - **r > 0.7**: PageRank accurately predicts influential users
                    - **0.4 < r < 0.7**: Moderate prediction accuracy
                    - **r < 0.4**: PageRank may not reflect true influence
                    
                    **Conclusion**: {'✅ PageRank is a reliable influence metric' if correlation > 0.7 else '⚠️ Consider additional metrics for influence assessment' if correlation > 0.4 else '❌ PageRank shows poor correlation with engagement'}
                    """)
            else:
                st.info("⚠️ Validation metrics not available. Enable advanced analytics when running analysis.")
        
        # TAB 10: SENTIMENT NETWORKS
        with tab10:
            st.header("🔀 Sentiment-Weighted Network Analysis")
            
            # Load sentiment network data
            sentiment_file = f"{output_dir}/{selected_subreddit}_sentiment_networks.json"
            
            if os.path.exists(sentiment_file):
                try:
                    with open(sentiment_file, 'r') as f:
                        content = f.read().strip()
                        if not content:
                            st.warning("⚠️ Sentiment networks file is empty. Run analysis with 'Enable Advanced Features' to generate sentiment networks.")
                        else:
                            sentiment_data = json.loads(content)
                            
                            st.subheader("😊 Sentiment Distribution")
                            
                            # Sentiment stats
                            stats = sentiment_data.get('stats', {})
                            
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                pos_count = stats.get('positive_count', 0)
                                st.metric(
                                    "😊 Positive Interactions",
                                    f"{pos_count}",
                                    f"{stats.get('positive_pct', 0):.1f}%"
                                )
                            
                            with col2:
                                neu_count = stats.get('neutral_count', 0)
                                st.metric(
                                    "😐 Neutral Interactions",
                                    f"{neu_count}",
                                    f"{stats.get('neutral_pct', 0):.1f}%"
                                )
                            
                            with col3:
                                neg_count = stats.get('negative_count', 0)
                                st.metric(
                                    "😞 Negative Interactions",
                                    f"{neg_count}",
                                    f"{stats.get('negative_pct', 0):.1f}%"
                                )
                            
                            # Pie chart
                            sentiment_counts = {
                                'Positive': pos_count,
                                'Neutral': neu_count,
                                'Negative': neg_count
                            }
                            
                            fig_pie = px.pie(
                                values=list(sentiment_counts.values()),
                                names=list(sentiment_counts.keys()),
                                title='Sentiment Distribution',
                                color_discrete_map={
                                    'Positive': '#00cc96',
                                    'Neutral': '#636efa',
                                    'Negative': '#ef553b'
                                }
                            )
                            st.plotly_chart(fig_pie, use_container_width=True)
                            
                            # Network comparison
                            st.subheader("🕸️ Sentiment Network Comparison")
                            
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.markdown("### 😊 Positive Network")
                                pos_graph = sentiment_data.get('positive', {})
                                st.metric("Nodes", len(pos_graph.get('nodes', [])))
                                st.metric("Edges", len(pos_graph.get('links', [])))
                            
                            with col2:
                                st.markdown("### 😐 Neutral Network")
                                neu_graph = sentiment_data.get('neutral', {})
                                st.metric("Nodes", len(neu_graph.get('nodes', [])))
                                st.metric("Edges", len(neu_graph.get('links', [])))
                            
                            with col3:
                                st.markdown("### 😞 Negative Network")
                                neg_graph = sentiment_data.get('negative', {})
                                st.metric("Nodes", len(neg_graph.get('nodes', [])))
                                st.metric("Edges", len(neg_graph.get('links', [])))
                            
                            st.markdown("### 💡 Insights")
                            
                            # Community health based on sentiment
                            pos_pct = stats.get('positive_pct', 0)
                            neg_pct = stats.get('negative_pct', 0)
                            
                            if pos_pct > 50:
                                health = "✅ **Healthy Community**: Majority of interactions are positive"
                            elif neg_pct > 50:
                                health = "⚠️ **Challenging Community**: High negative sentiment detected"
                            else:
                                health = "🔄 **Mixed Community**: Balanced sentiment distribution"
                            
                            st.markdown(health)
                            
                except json.JSONDecodeError as e:
                    st.error(f"❌ Error reading sentiment networks file: {e}")
                    st.info("💡 Try running a new analysis with 'Enable Advanced Features' turned ON.")
            else:
                st.warning("⚠️ No sentiment network data found. Run analysis with 'Enable Advanced Features' to generate sentiment networks.")
        
        # TAB 11: ML PREDICTIONS
        with tab11:
            st.header("🎯 Machine Learning Viral Content Prediction")
            
            # Load ML predictor data
            predictor_file = f"{output_dir}/{selected_subreddit}_predictor.json"
            
            if os.path.exists(predictor_file):
                with open(predictor_file, 'r') as f:
                    predictor_data = json.load(f)
                
                st.subheader("📊 Model Performance")
                
                # Model metrics
                metrics = predictor_data.get('metrics', {})
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    accuracy = metrics.get('accuracy', 0)
                    st.metric(
                        "Accuracy",
                        f"{accuracy:.1%}",
                        "Overall correctness"
                    )
                
                with col2:
                    precision = metrics.get('precision', 0)
                    st.metric(
                        "Precision",
                        f"{precision:.1%}",
                        "Viral prediction accuracy"
                    )
                
                with col3:
                    recall = metrics.get('recall', 0)
                    st.metric(
                        "Recall",
                        f"{recall:.1%}",
                        "Viral detection rate"
                    )
                
                with col4:
                    f1 = metrics.get('f1_score', 0)
                    st.metric(
                        "F1 Score",
                        f"{f1:.1%}",
                        "Balanced metric"
                    )
                
                # Feature importance
                if 'feature_importance' in predictor_data:
                    st.subheader("🔍 Feature Importance")
                    
                    features_df = pd.DataFrame(
                        list(predictor_data['feature_importance'].items()),
                        columns=['Feature', 'Importance']
                    ).sort_values('Importance', ascending=False)
                    
                    fig_features = px.bar(
                        features_df,
                        x='Importance',
                        y='Feature',
                        orientation='h',
                        title='Top Predictive Features',
                        color='Importance',
                        color_continuous_scale='Viridis'
                    )
                    st.plotly_chart(fig_features, use_container_width=True)
                    
                    st.markdown("### 💡 Key Findings")
                    top_feature = features_df.iloc[0]['Feature']
                    st.markdown(f"- **Most Important**: {top_feature} is the strongest predictor of viral content")
                    st.markdown(f"- **Model Accuracy**: {accuracy:.1%} success rate in predicting viral posts")
                    st.markdown(f"- **Reliability**: {'High' if accuracy > 0.8 else 'Moderate' if accuracy > 0.6 else 'Low'} - {'Use with confidence' if accuracy > 0.8 else 'Consider as one of multiple factors' if accuracy > 0.6 else 'Use cautiously'}")
                
                # Predictions
                if 'predictions' in predictor_data and predictor_data['predictions']:
                    st.subheader("🎯 Top Viral Predictions")
                    
                    predictions_df = pd.DataFrame(predictor_data['predictions'])
                    predictions_df = predictions_df.sort_values('viral_probability', ascending=False).head(10)
                    
                    st.dataframe(
                        predictions_df[['title', 'viral_probability', 'predicted_viral', 'actual_score']],
                        use_container_width=True
                    )
                    
                    st.markdown("### 📈 Prediction Distribution")
                    
                    fig_dist = px.histogram(
                        pd.DataFrame(predictor_data['predictions']),
                        x='viral_probability',
                        nbins=20,
                        title='Viral Probability Distribution',
                        labels={'viral_probability': 'Predicted Viral Probability'}
                    )
                    st.plotly_chart(fig_dist, use_container_width=True)
                
            else:
                st.info("⚠️ ML prediction data not available. Enable advanced analytics when running analysis.")

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
    
    3. Select your analysis from the side
    bar
    
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

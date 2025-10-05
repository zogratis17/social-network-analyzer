# Project Summary: AI Social Network Analysis for Reddit

## 📁 Project Structure

```
social-network-analyzer/
├── ai_sn_analysis_prototype.py  # Main application (540 lines)
├── setup.py                      # Automated setup script
├── example.py                    # Example usage script
├── requirements.txt              # Python dependencies
├── .env.example                  # Environment template
├── .env                          # Your credentials (not in git)
├── .gitignore                    # Git ignore rules
├── README.md                     # Full documentation
├── QUICKSTART.md                 # Quick start guide
└── output/                       # Analysis results
```

## ✅ What's Included

### Core Application (`ai_sn_analysis_prototype.py`)
- **RedditCollector**: Fetches posts and comments using PRAW
- **GraphBuilder**: Constructs directed graphs from user interactions
- **GeminiClient**: AI content analysis (with local fallback)
- **Community Detection**: Union-Find and Greedy Modularity algorithms
- **Influence Metrics**: PageRank, betweenness, closeness centrality
- **MST Computation**: Minimum spanning tree generation
- **Trend Detection**: Velocity-based trending topic identification
- **Visualization**: Interactive Plotly graphs

### Supporting Files
- **setup.py**: Automated installer for dependencies and configuration
- **example.py**: Programmatic usage examples
- **requirements.txt**: All Python dependencies
- **.env.example**: Template for API credentials
- **README.md**: Complete documentation with all features
- **QUICKSTART.md**: Step-by-step getting started guide
- **.gitignore**: Prevents committing sensitive data

## 🚀 How to Use

### First Time Setup
```bash
# 1. Run setup
python setup.py

# 2. Edit .env with your Reddit credentials
# Get them from: https://www.reddit.com/prefs/apps

# 3. Run analysis
python ai_sn_analysis_prototype.py --subreddit python --posts 100
```

### Command Line Usage
```bash
# Basic analysis
python ai_sn_analysis_prototype.py --subreddit datascience --posts 200

# Recent posts only
python ai_sn_analysis_prototype.py --subreddit learnprogramming --posts 100 --time-filter week

# Custom output directory
python ai_sn_analysis_prototype.py --subreddit python --posts 300 --outdir results/my_analysis
```

### Programmatic Usage
```python
from ai_sn_analysis_prototype import run_pipeline

results = run_pipeline(
    subreddit='python',
    posts_limit=100,
    outdir='output/python_analysis',
    time_filter='week'
)

# Access the graph
G = results['graph']
print(f"Nodes: {G.number_of_nodes()}")
print(f"Edges: {G.number_of_edges()}")

# Get trending topics
for topic, data in results['trends'][:5]:
    print(f"{topic}: velocity={data['velocity']}")
```

## 📊 Output Files

Each analysis generates:
- `{subreddit}_raw_posts.json` - Raw Reddit data
- `{subreddit}_graph.graphml` - Graph (GraphML format)
- `{subreddit}_graph.json` - Graph (JSON format)
- `{subreddit}_mst.graphml` - Minimum spanning tree
- `{subreddit}_content_analysis.json` - AI sentiment/topic analysis
- `{subreddit}_trends.json` - Trending topics
- `{subreddit}_nodes.csv` - Node attributes table
- `{subreddit}_edges.csv` - Edge attributes table
- `{subreddit}_graph.html` - **Interactive visualization** ⭐

## 🔧 Configuration

### Environment Variables (.env)
```bash
REDDIT_CLIENT_ID=your_id
REDDIT_CLIENT_SECRET=your_secret
REDDIT_USER_AGENT=ai-sn-analysis/0.1
GEMINI_API_KEY=optional_gemini_key  # For advanced AI analysis
```

### Defaults (in code)
```python
DEFAULTS = {
    'MIN_POSTS': 500,
    'PAGE_LIMIT': 100,
    'RATE_SLEEP': 1.0,  # Polite rate limiting
}
```

## 🎯 Key Features

### 1. Data Collection
- Fetches posts and comments from any subreddit
- Configurable limits and time filters
- Built-in rate limiting to respect Reddit API

### 2. Graph Analysis
- **Nodes**: Reddit users
- **Edges**: Interactions (replies, mentions)
- **Weights**: Number of interactions
- Directed graph structure

### 3. Community Detection
- **Union-Find**: Connected components
- **Greedy Modularity**: Optimized community detection

### 4. Influence Metrics
- **PageRank**: Overall influence score
- **Betweenness**: Bridge user identification
- **Closeness**: Network centrality
- **Degree**: Direct connections

### 5. Content Analysis
- Sentiment analysis (positive/neutral/negative)
- Topic extraction
- Viral score prediction
- Gemini API integration (optional)

### 6. Trend Detection
- Velocity-based trending (recent vs historical)
- Topic frequency analysis
- Time-windowed comparisons

### 7. Visualization
- Interactive HTML graphs (zoomable, draggable)
- Color-coded communities
- Size-scaled by influence (PageRank)
- Hover tooltips with user details

## 📦 Dependencies

All managed in `requirements.txt`:
- **praw**: Reddit API wrapper
- **networkx**: Graph algorithms
- **pandas**: Data manipulation
- **plotly**: Interactive visualizations
- **matplotlib**: Static plots
- **tqdm**: Progress bars
- **python-dotenv**: Environment variables (optional)

## 🔐 Security

- `.env` file in `.gitignore` (credentials not committed)
- `.env.example` template provided
- No hardcoded credentials

## 💡 Example Use Cases

1. **Community Analysis**: Identify sub-communities in a subreddit
2. **Influence Mapping**: Find the most influential users
3. **Trend Tracking**: Monitor emerging topics over time
4. **Network Dynamics**: Study how users interact
5. **Content Research**: Analyze sentiment and topics

## 🎓 Academic Applications

Perfect for:
- Social network analysis research
- Graph theory projects
- Data mining coursework
- Network visualization studies
- Community detection algorithms

## ✨ Next Steps / Extensions

Potential enhancements:
- [ ] Multi-platform support (Twitter, Facebook)
- [ ] Real-time streaming analysis
- [ ] Machine learning for user classification
- [ ] Temporal network evolution tracking
- [ ] Advanced graph algorithms (centrality, clustering)
- [ ] Database integration for large-scale analysis
- [ ] Web dashboard interface

## 📄 License

Educational/Research prototype. Respect Reddit's API terms and user privacy.

## 🆘 Support

See `README.md` for troubleshooting and detailed documentation.
See `QUICKSTART.md` for step-by-step setup instructions.

---

**Status**: ✅ **Complete and Ready to Use**

All components are implemented and tested. The app is fully functional for Reddit analysis.

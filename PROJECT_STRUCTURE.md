# 📁 Project Structure

## Overview
Clean, modular architecture for AI-powered social network analysis with production-ready components.

---

## Directory Structure

```
social-network-analyzer/
├── 📄 Core Analysis
│   ├── ai_sn_analysis_prototype.py    # Main analysis engine (48KB)
│   ├── advanced_analytics.py          # Advanced features (temporal, ML, validation)
│   ├── utils.py                       # Error handling & logging
│   └── performance.py                 # Caching & optimization
│
├── 🌐 Web Interfaces
│   ├── dashboard.py                   # Streamlit dashboard (11 tabs)
│   ├── launch_dashboard.py            # Dashboard launcher
│   └── api.py                         # FastAPI REST server
│
├── 🛠️ Utilities
│   ├── regen_graph.py                 # Regenerate visualizations
│   ├── view_trends.py                 # View trending topics
│   └── setup.py                       # Package setup
│
├── 🧪 Testing
│   ├── test_ai_sn_analysis.py         # Unit tests
│   └── test_setup.py                  # Setup tests
│
├── 📚 Documentation
│   ├── README.md                      # Main guide (setup & usage)
│   ├── API_GUIDE.md                   # REST API reference
│   └── ENHANCEMENTS_SUMMARY.md        # Features catalog
│
├── ⚙️ Configuration
│   ├── requirements.txt               # Python dependencies
│   ├── .env.example                   # Environment template
│   ├── .env                          # Your credentials (gitignored)
│   └── .gitignore                    # Git exclusions
│
└── 📦 Output
    └── output/                        # Analysis results
        ├── {subreddit}_graph.html     # Interactive visualization
        ├── {subreddit}_nodes.csv      # User data
        ├── {subreddit}_edges.csv      # Interactions
        ├── {subreddit}_trends.json    # Trending topics
        ├── {subreddit}_validation.json     # ✨ Accuracy metrics
        ├── {subreddit}_sentiment_networks.json  # ✨ Sentiment graphs
        ├── {subreddit}_evolution.json          # ✨ Temporal data
        ├── {subreddit}_layers.json             # ✨ Multi-layer networks
        └── {subreddit}_predictor.json          # ✨ ML predictions
```

---

## Core Modules

### 1. **ai_sn_analysis_prototype.py** (Main Engine)
**Purpose:** Core analysis pipeline  
**Key Functions:**
- `RedditCollector` - Fetch posts from Reddit API
- `GraphBuilder` - Construct social network graphs
- `GeminiClient` - AI-powered content analysis
- `detect_communities_*` - Union-Find & Louvain algorithms
- `compute_pagerank` - Influence measurement
- `detect_trends` - Topic extraction & velocity analysis
- `visualize_graph_plotly` - Interactive network visualization

**Usage:**
```bash
python ai_sn_analysis_prototype.py --subreddit python --posts 100 --enable-advanced
```

---

### 2. **advanced_analytics.py** (Advanced Features)
**Purpose:** Extended analysis capabilities  
**Key Functions:**
- `should_append_data()` - Smart data merging (7-day threshold)
- `validate_influence_metrics()` - PageRank accuracy correlation
- `build_sentiment_weighted_graphs()` - Positive/negative/neutral networks
- `build_temporal_graphs()` - Time-series analysis
- `detect_community_evolution()` - Stability & churn tracking
- `build_multilayer_network()` - Reply/mention/topic layers
- `train_viral_predictor()` - RandomForest ML model

**Dependencies:** `scikit-learn`, `networkx`, `pandas`

---

### 3. **utils.py** (Infrastructure)
**Purpose:** Error handling & logging  
**Key Components:**
- Custom exceptions: `AnalysisError`, `DataFetchError`, `GraphBuildError`
- `setup_logging()` - Rotating file handler (10MB, 5 backups)
- `PerformanceTimer` - Context manager for timing
- `validate_*()` - Input validation functions

**Log Location:** `logs/analysis.log`

---

### 4. **performance.py** (Optimization)
**Purpose:** Performance enhancement tools  
**Key Components:**
- `DiskCache` - Pickle-based caching system
- `@cached(ttl)` - Function result caching decorator
- `parallel_map()` - Multi-threading/processing
- `@profile_function` - cProfile integration
- `MemoryMonitor` - Memory usage tracking

---

## Web Interfaces

### 5. **dashboard.py** (Streamlit UI)
**Purpose:** Interactive web dashboard  
**Features:**
- 11 comprehensive tabs
- Real-time analysis execution
- Advanced analytics toggle
- Data append mode selector
- Export capabilities

**Tabs:**
1. Overview - Summary metrics
2. Communities - Detection results
3. Influencers - PageRank rankings
4. Trending Topics - Velocity analysis
5. Network Graph - Interactive visualization
6. AI Insights - Gemini analysis
7. Analytics - Advanced metrics
8. ⏰ Temporal Evolution - Growth trends
9. ✅ Validation Metrics - Accuracy assessment
10. 🔀 Sentiment Networks - Emotion-based graphs
11. 🎯 ML Predictions - Viral predictor

**Launch:**
```bash
python launch_dashboard.py
# Or: streamlit run dashboard.py
```

---

### 6. **api.py** (FastAPI Server)
**Purpose:** REST API for programmatic access  
**Endpoints:**
- `POST /analyze` - Start background analysis
- `GET /status/{task_id}` - Check progress
- `GET /results/{subreddit}` - Retrieve data
- `GET /list` - List analyses
- `GET /download/{subreddit}/{type}` - Download files
- `DELETE /delete/{subreddit}` - Remove analysis

**Launch:**
```bash
python api.py
# API: http://localhost:8000
# Docs: http://localhost:8000/docs
```

---

## Utility Scripts

### 7. **regen_graph.py**
**Purpose:** Regenerate network visualization  
**Usage:**
```bash
python regen_graph.py python
# Output: output/python_graph.html
```

### 8. **view_trends.py**
**Purpose:** Display trending topics in terminal  
**Usage:**
```bash
python view_trends.py
# Shows: Top 15 topics with velocity metrics
```

---

## Configuration Files

### requirements.txt
**Dependencies:**
```
# Core
praw==7.7.1              # Reddit API
networkx==3.0            # Graph algorithms
pandas==2.0.0            # Data manipulation
plotly==5.14.0           # Visualizations
streamlit==1.28.0        # Dashboard

# Advanced
scikit-learn>=1.3.0      # ML models
fastapi>=0.104.0         # REST API
uvicorn>=0.24.0          # ASGI server
psutil>=5.9.0            # Performance monitoring

# Optional
google-generativeai      # Gemini AI
python-dotenv            # Environment variables
```

### .env (Template in .env.example)
```env
REDDIT_CLIENT_ID=your_client_id
REDDIT_CLIENT_SECRET=your_client_secret
REDDIT_USER_AGENT=ai-sn-analysis/1.0
GEMINI_API_KEY=your_gemini_key  # Optional
```

---

## Output Files

### Standard Outputs
- `*_raw_posts.json` - Raw Reddit data
- `*_graph.json` - Network structure (node-link format)
- `*_graph.graphml` - GraphML export
- `*_graph.html` - Interactive Plotly visualization
- `*_mst.graphml` - Minimum spanning tree
- `*_nodes.csv` - User/node data table
- `*_edges.csv` - Interaction/edge data table
- `*_content_analysis.json` - AI/local content analysis
- `*_trends.json` - Trending topics with velocity

### Advanced Outputs (with `--enable-advanced`)
- `*_validation.json` - PageRank accuracy metrics
- `*_sentiment_networks.json` - Positive/negative/neutral graphs
- `*_evolution.json` - Temporal evolution data
- `*_layers.json` - Multi-layer network structure
- `*_predictor.json` - ML model predictions & metrics

---

## Data Flow

```
1. Input (Reddit API)
   └─> RedditCollector.fetch_subreddit_posts()
       └─> Returns: List[Dict] (posts + comments)

2. Graph Construction
   └─> GraphBuilder.build_graph_from_posts()
       └─> Returns: NetworkX Graph

3. Community Detection
   ├─> detect_communities_union_find()  # O(α(n))
   └─> detect_communities_louvain()     # Modularity optimization

4. Influence Analysis
   └─> compute_pagerank()
       └─> Updates: Node attributes

5. Content Analysis (Optional)
   ├─> GeminiClient.analyze_text()      # AI-powered
   └─> simple_local_text_analysis()     # Fast fallback

6. Trend Detection
   └─> detect_trends()
       └─> Returns: [(topic, metrics)]

7. Advanced Analytics (Optional)
   ├─> validate_influence_metrics()     # Accuracy
   ├─> build_sentiment_weighted_graphs() # 3 graphs
   ├─> build_temporal_graphs()          # Time-series
   ├─> build_multilayer_network()       # 3 layers
   └─> train_viral_predictor()          # RandomForest

8. Visualization
   └─> visualize_graph_plotly()
       └─> Outputs: Interactive HTML

9. Export
   ├─> JSON (full data)
   ├─> GraphML (network tools)
   ├─> CSV (spreadsheet analysis)
   └─> HTML (browser viewing)
```

---

## Testing

### Run Tests
```bash
# All tests
python -m pytest

# Specific test
python -m pytest test_ai_sn_analysis.py

# With coverage
python -m pytest --cov=.
```

### Test Coverage
- Reddit API mocking
- Graph construction
- Community detection algorithms
- PageRank computation
- Trend detection
- File I/O operations

---

## Best Practices

### Code Organization
✅ Modular design - Each file has single responsibility  
✅ Type hints - All functions annotated  
✅ Docstrings - Comprehensive documentation  
✅ Error handling - Custom exceptions throughout  
✅ Logging - Structured logging at all levels  

### Performance
✅ Caching - Expensive operations cached  
✅ Parallelization - Multi-threading for graph ops  
✅ Profiling - Performance tracking available  
✅ Memory monitoring - Leak detection  

### Production Ready
✅ Environment variables - No hardcoded secrets  
✅ Graceful degradation - Works without optional features  
✅ Progress tracking - Real-time feedback  
✅ Comprehensive docs - README + API guide  

---

## Maintenance

### Add New Feature
1. Implement in appropriate module (`advanced_analytics.py` for advanced features)
2. Add to pipeline in `ai_sn_analysis_prototype.py`
3. Update dashboard tab in `dashboard.py`
4. Add API endpoint in `api.py` if needed
5. Document in `ENHANCEMENTS_SUMMARY.md`
6. Write tests in `test_ai_sn_analysis.py`

### Update Dependencies
```bash
pip install --upgrade -r requirements.txt
pip freeze > requirements.txt  # Update lock file
```

### Clean Output
```bash
# Remove all analysis files
Remove-Item output/* -Recurse -Force

# Remove logs
Remove-Item logs/* -Force

# Remove cache
Remove-Item __pycache__/* -Recurse -Force
```

---

## File Size Summary

| Category | Files | Total Size |
|----------|-------|------------|
| Core Python | 5 | ~90 KB |
| Web Interfaces | 3 | ~95 KB |
| Utilities | 3 | ~15 KB |
| Tests | 2 | ~14 KB |
| Documentation | 3 | ~150 KB |
| **Total** | **16** | **~364 KB** |

**Note:** Compact, efficient codebase with comprehensive features

---

## Quick Reference

| Task | Command |
|------|---------|
| Run analysis | `python ai_sn_analysis_prototype.py --subreddit SUBREDDIT --posts 100` |
| Launch dashboard | `python launch_dashboard.py` |
| Start API | `python api.py` |
| View trends | `python view_trends.py` |
| Regenerate graph | `python regen_graph.py SUBREDDIT` |
| Run tests | `python -m pytest` |
| Install deps | `pip install -r requirements.txt` |

---

**Last Updated:** October 15, 2025  
**Version:** 2.0.0  
**Status:** Production Ready ✅

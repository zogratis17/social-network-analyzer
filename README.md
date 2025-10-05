# 🕸️ AI Social Network Analyzer

An intelligent network analysis platform that combines advanced graph algorithms with AI-powered content analysis to detect communities, identify influencers, predict trends, and analyze user behavior from social media data.

## ✨ Platform Features

### � Graph Algorithms
- **Union-Find**: Fast connected component detection for community identification
- **PageRank**: Google's algorithm adapted for social influence measurement
- **Greedy Modularity**: Optimized community structure detection
- **Minimum Spanning Trees**: Network optimization and hierarchical structure
- **Centrality Metrics**: Betweenness, Closeness, and Degree analysis

### 🤖 AI-Powered Analysis
- **Google Gemini Integration**: Advanced content understanding and categorization
- **Sentiment Analysis**: Positive/Neutral/Negative classification with confidence scores
- **Topic Extraction**: Multi-word phrase detection using TF-IDF and N-grams
- **Viral Prediction**: Machine learning-based engagement potential scoring
- **Trend Detection**: Temporal velocity analysis with AI filtering

### 📊 Interactive Dashboard
- **7 Comprehensive Tabs**: Overview, Communities, Influencers, Trends, Network Graph, AI Insights, Analytics
- **Real-time Visualizations**: Interactive Plotly charts and graphs
- **Multi-Analysis Support**: Compare different subreddits and time periods
- **Export Capabilities**: Download CSV, JSON, and GraphML files
- **Professional UI**: Built with Streamlit for a modern web experience

## Installation

### Prerequisites

- Python 3.8+
- Reddit API credentials (get them from [Reddit Apps](https://www.reddit.com/prefs/apps))

### Setup

1. **Clone or download this repository**

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment variables:**
   
   Copy the `.env.example` file to `.env`:
   ```bash
   cp .env.example .env
   ```
   
## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/social-network-analyzer.git
cd social-network-analyzer

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration

Create a `.env` file in the project root:

```env
# Reddit API (Required)
REDDIT_CLIENT_ID=your_client_id_here
REDDIT_CLIENT_SECRET=your_client_secret_here
REDDIT_USER_AGENT=ai-sn-analysis/1.0

# Google Gemini API (Optional, for AI features)
GEMINI_API_KEY=your_gemini_api_key_here
```

**Get API Keys:**
- Reddit: [https://www.reddit.com/prefs/apps](https://www.reddit.com/prefs/apps)
- Gemini: [https://ai.google.dev/](https://ai.google.dev/)

### 3. Run Analysis

```bash
# Analyze a subreddit
python ai_sn_analysis_prototype.py --subreddit python --posts 100
```

### 4. Launch Dashboard

```bash
# Quick launch
python launch_dashboard.py

# Or directly
streamlit run dashboard.py
```

The dashboard will open at: `http://localhost:8501`

## 💻 Usage

### Command-Line Analysis

```bash
python ai_sn_analysis_prototype.py [options]
```

**Options:**
- `--subreddit` (required): Subreddit name (e.g., `python`, `machinelearning`)
- `--posts`: Number of posts to fetch (default: 500)
- `--outdir`: Output directory (default: `output`)
- `--gemini-key`: Gemini API key (or use .env file)
- `--time-filter`: Time period - `all`, `year`, `month`, `week`, `day` (default: `all`)

**Examples:**

```bash
# Quick analysis
python ai_sn_analysis_prototype.py --subreddit python --posts 50

# Deep analysis with 500 posts
python ai_sn_analysis_prototype.py --subreddit datascience --posts 500

# Recent week's activity
python ai_sn_analysis_prototype.py --subreddit machinelearning --posts 200 --time-filter week

# Custom output location
python ai_sn_analysis_prototype.py --subreddit learnprogramming --posts 100 --outdir results/edu
```

### Dashboard Features

Launch the interactive dashboard to explore:

1. **📊 Overview**: Key metrics, network statistics, degree distribution
2. **👥 Communities**: Union-Find vs Greedy Modularity comparison
3. **⭐ Influencers**: PageRank leaders, activity patterns, influence levels
4. **🔥 Trending Topics**: AI-extracted topics with momentum analysis
5. **🕸️ Network Graph**: Interactive visualization with community colors
6. **🧠 AI Insights**: Sentiment analysis, viral prediction, topic extraction
7. **📈 Analytics**: Network health, recommendations, data export

## 📁 Output Files

The application generates several output files in the specified directory:

- `{subreddit}_raw_posts.json` - Raw Reddit data
- `{subreddit}_graph.graphml` - Graph in GraphML format
- `{subreddit}_graph.json` - Graph in JSON format
- `{subreddit}_mst.graphml` - Minimum Spanning Tree
- `{subreddit}_content_analysis.json` - AI content analysis results
- `{subreddit}_trends.json` - Detected trending topics
- `{subreddit}_nodes.csv` - Node attributes table
- `{subreddit}_edges.csv` - Edge attributes table
- `{subreddit}_graph.html` - Interactive visualization (open in browser)

## How It Works

1. **Data Collection**: Uses PRAW to fetch posts and comments from Reddit
2. **Graph Building**: Creates a directed graph where:
   - Nodes = Users
   - Edges = Interactions (replies, mentions)
   - Edge weights = Number of interactions
3. **Community Detection**: Applies algorithms to identify user communities
4. **Influence Metrics**: Calculates PageRank, betweenness, closeness centrality
5. **Content Analysis**: Analyzes text for sentiment and topics
6. **Trend Detection**: Identifies topics gaining velocity
7. **Visualization**: Generates interactive network graphs

## Configuration

Edit the `DEFAULTS` dictionary in the script to customize:
- `MIN_POSTS`: Minimum number of posts to collect
- `PAGE_LIMIT`: Page limit for Reddit API
- `RATE_SLEEP`: Delay between API requests (in seconds)

## API Keys

### Reddit API
1. Go to https://www.reddit.com/prefs/apps
2. Create a new app (select "script")
3. Copy the client ID and secret to your `.env` file

### Gemini API (Optional)
If you want advanced AI content analysis, add your Gemini API key to the `.env` file. Without it, the app uses a simple rule-based fallback analyzer.

## Troubleshooting

**"REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET must be set"**
- Make sure your `.env` file exists and contains valid credentials
- Load the environment variables before running the script

**Rate Limiting**
- Reddit API has rate limits. The app includes delays between requests
- For large analyses, consider increasing `RATE_SLEEP` value

**Graph visualization not showing**
- Open the generated `.html` file in a web browser
- Some nodes might overlap; zoom and pan to explore

## Dependencies

- **praw**: Reddit API wrapper
- **networkx**: Graph algorithms and analysis
- **pandas**: Data manipulation
- **plotly**: Interactive visualizations
- **matplotlib**: Static plotting
- **tqdm**: Progress bars

## License

This is a prototype for educational and research purposes.

## Contributing

Feel free to enhance this prototype with additional features:
- More social platforms (Twitter, Facebook, etc.)
- Advanced ML models for content classification
- Real-time streaming analysis
- Additional graph algorithms
- Enhanced visualizations

## Notes

- This is a prototype and may require adjustments for production use
- Respect Reddit's API terms of service and rate limits
- Be mindful of user privacy when analyzing social data

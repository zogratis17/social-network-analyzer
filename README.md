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
   
## 🚀 Quick Start Guide

### Prerequisites

Before you begin, ensure you have:
- **Python 3.8 or higher** installed ([Download Python](https://www.python.org/downloads/))
- **pip** package manager (comes with Python)
- **Reddit API credentials** (free, takes 2 minutes to get)
- **(Optional)** Google Gemini API key for AI features

---

### Step 1: Clone/Download the Project

```bash
# Option A: Clone with git
git clone https://github.com/zogratis17/social-network-analyzer.git
cd social-network-analyzer

# Option B: Download ZIP and extract, then navigate to folder
cd social-network-analyzer
```

---

### Step 2: Install Dependencies

```bash
# Install all required packages
pip install -r requirements.txt
```

**Note for Windows users:** If you get SSL errors, try:
```bash
python -m pip install --upgrade pip
pip install -r requirements.txt --trusted-host pypi.org --trusted-host files.pythonhosted.org
```

---

### Step 3: Get Reddit API Credentials (Required)

1. **Go to:** [https://www.reddit.com/prefs/apps](https://www.reddit.com/prefs/apps)
2. **Click** "Create App" or "Create Another App" at the bottom
3. **Fill in the form:**
   - **Name:** `My Social Network Analyzer` (or any name)
   - **App type:** Select **"script"**
   - **Description:** (leave blank or add description)
   - **About URL:** (leave blank)
   - **Redirect URI:** `http://localhost:8080` (required but not used)
4. **Click** "Create app"
5. **Copy your credentials:**
   - **Client ID:** The string under "personal use script" (looks like: `dQw4w9WgXcQ`)
   - **Client Secret:** The string next to "secret" (looks like: `dQw4w9WgXcQ-dQw4w9WgXcQ`)

---

### Step 4: Configure Environment Variables

Create a `.env` file in the project root directory:

**Windows (PowerShell):**
```powershell
Copy-Item .env.example .env
notepad .env
```

**macOS/Linux:**
```bash
cp .env.example .env
nano .env
```

**Edit the `.env` file with your credentials:**

```env
# Reddit API (Required)
REDDIT_CLIENT_ID=paste_your_client_id_here
REDDIT_CLIENT_SECRET=paste_your_client_secret_here
REDDIT_USER_AGENT=ai-sn-analysis/1.0

# Google Gemini API (Optional - for AI features)
GEMINI_API_KEY=paste_your_gemini_key_here
```

**⚠️ Important:** Replace `paste_your_client_id_here` and `paste_your_client_secret_here` with your actual Reddit credentials!

**Get Gemini API (Optional):**
- Visit: [https://makersuite.google.com/app/apikey](https://makersuite.google.com/app/apikey)
- Click "Create API Key"
- Copy and paste into `.env` file

---

### Step 5: Run the Project

You have **two options** to use the platform:

#### **Option A: Interactive Dashboard (Recommended)**

The easiest way to use the platform with a visual interface:

```bash
# Quick launch
python launch_dashboard.py

# Or use streamlit directly
streamlit run dashboard.py
```

**The dashboard will automatically open in your browser at:** `http://localhost:8501`

**Dashboard Features:**
- 🔍 **New Analysis**: Enter any subreddit name and analyze in real-time
- 📂 **Load Existing**: View previously analyzed subreddits
- 🎛️ **Customizable**: Adjust post count (10-500), time filter, AI toggle
- 📊 **7 Interactive Tabs**: Overview, Communities, Influencers, Trends, Network, AI Insights, Analytics

---

#### **Option B: Command-Line Analysis**

For automated analysis or scripting:

```bash
python ai_sn_analysis_prototype.py --subreddit SUBREDDIT_NAME --posts NUMBER_OF_POSTS
```

**Basic Examples:**

```bash
# Analyze r/python with 50 posts (quick test)
python ai_sn_analysis_prototype.py --subreddit python --posts 50

# Analyze r/machinelearning with 200 posts
python ai_sn_analysis_prototype.py --subreddit machinelearning --posts 200

# Analyze r/datascience with 500 posts (comprehensive)
python ai_sn_analysis_prototype.py --subreddit datascience --posts 500
```

**Advanced Options:**

```bash
# Use specific time filter
python ai_sn_analysis_prototype.py --subreddit python --posts 100 --time-filter week

# Custom output directory
python ai_sn_analysis_prototype.py --subreddit gaming --posts 200 --outdir my_results

# All options combined
python ai_sn_analysis_prototype.py \
  --subreddit learnprogramming \
  --posts 300 \
  --time-filter month \
  --outdir analysis_results
```

**Available Options:**
| Option | Description | Default | Example |
|--------|-------------|---------|---------|
| `--subreddit` | Subreddit name (required) | - | `python`, `gaming` |
| `--posts` | Number of posts to fetch | 500 | `50`, `100`, `500` |
| `--time-filter` | Time period | `all` | `day`, `week`, `month`, `year`, `all` |
| `--outdir` | Output directory | `output` | `results`, `my_analysis` |
| `--gemini-key` | Gemini API key | From `.env` | Your API key |

---

### Step 6: View Results

#### **If using Dashboard:**
- Results appear automatically in the browser
- Switch between tabs to explore different visualizations
- Use "Load Existing" to view past analyses

#### **If using Command-Line:**

After analysis completes, check the `output/` directory:

```bash
# View generated files
ls output/

# Open interactive network graph in browser
output/python_graph.html  # Double-click or open in browser
```

**Generated Files:**
```
output/
├── python_raw_posts.json          # Raw Reddit data
├── python_graph.json              # Graph structure
├── python_graph.graphml           # GraphML format
├── python_graph.html              # 🌟 Interactive visualization
├── python_mst.graphml             # Minimum spanning tree
├── python_nodes.csv               # User data table
├── python_edges.csv               # Interaction data
├── python_content_analysis.json   # AI analysis results
└── python_trends.json             # Trending topics
```

---

## 📖 How to Use the Dashboard

### **New Analysis Mode:**

1. **Select Mode:** Choose "🔍 New Analysis" in the sidebar
2. **Enter Subreddit:** Type subreddit name (without r/) - e.g., `python`, `gaming`
3. **Configure Settings:**
   - **Posts to Fetch:** Slide to select 10-500 posts
   - **Time Filter:** Choose time period (day/week/month/year/all)
   - **AI Toggle:** Enable/disable Google Gemini AI (faster without)
4. **Click** "🚀 Run Analysis" button
5. **Wait:** Progress bar shows analysis stages (10-60 seconds)
6. **Explore:** Navigate through 7 tabs to view results

### **Load Existing Mode:**

1. **Select Mode:** Choose "📂 Load Existing Results"
2. **Pick Analysis:** Select from dropdown of analyzed subreddits
3. **Explore:** All visualizations load instantly

### **Dashboard Tabs Explained:**

| Tab | What You'll See |
|-----|-----------------|
| 📊 **Overview** | Network stats, degree distribution, recent posts |
| 👥 **Communities** | Union-Find vs Greedy Modularity algorithms |
| ⭐ **Influencers** | Top 20 users by PageRank, influence metrics |
| 🔥 **Trending Topics** | AI-detected topics with velocity scores |
| 🕸️ **Network Graph** | Interactive visualization (zoom/hover) |
| 🧠 **AI Insights** | Sentiment analysis, viral prediction, topics |
| 📈 **Analytics** | Network health, export data options |

---

## 🎯 Quick Test (30 seconds)

Want to quickly test if everything works?

```bash
# 1. Quick analysis (takes ~30 seconds)
python ai_sn_analysis_prototype.py --subreddit python --posts 10

# 2. Launch dashboard
python launch_dashboard.py

# 3. In browser: Select "Load Existing" → Choose "python" → Explore!
```

---

## ❓ Troubleshooting

### **"ModuleNotFoundError: No module named 'X'"**
```bash
pip install -r requirements.txt --upgrade
```

### **"REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET must be set"**
- Ensure `.env` file exists in project root
- Check credentials are correctly pasted (no extra spaces)
- Verify file is named exactly `.env` (not `.env.txt`)

### **"HTTP Error 401: Unauthorized"**
- Reddit credentials are invalid
- Re-generate credentials at [Reddit Apps](https://www.reddit.com/prefs/apps)
- Make sure you selected "script" type when creating the app

### **"Rate limit exceeded"**
- Wait a few minutes before trying again
- Reduce number of posts: `--posts 50` instead of `--posts 500`

### **"Streamlit not opening in browser"**
- Manually navigate to: `http://localhost:8501`
- Check if port 8501 is already in use
- Try: `streamlit run dashboard.py --server.port 8502`

### **"Network graph looks all black"**
- This is normal for some subreddits with many communities
- Try the Python subreddit for colorful visualization
- Use the dashboard's "Load Existing" for pre-generated graphs

### **Gemini AI not working**
- Check your API key in `.env` file
- The app will automatically fall back to local analysis
- AI is optional - analysis works without it

---

## 💻 System Requirements

- **OS:** Windows 10/11, macOS 10.14+, Linux (Ubuntu 20.04+)
- **Python:** 3.8 or higher
- **RAM:** 4GB minimum (8GB recommended for 500+ posts)
- **Storage:** 500MB free space
- **Internet:** Required for Reddit API access

---

## 🎓 For Researchers/Students

### Example Analysis Workflow:

```bash
# 1. Analyze multiple related subreddits
python ai_sn_analysis_prototype.py --subreddit python --posts 200
python ai_sn_analysis_prototype.py --subreddit learnpython --posts 200
python ai_sn_analysis_prototype.py --subreddit django --posts 200

# 2. Launch dashboard to compare
python launch_dashboard.py

# 3. Export data for further analysis
# Use "Analytics" tab → "Export" buttons → Download CSV/JSON

# 4. Use exported files in your favorite tools
# - Import CSVs into Excel/Google Sheets
# - Load GraphML into Gephi/Cytoscape
# - Process JSON with Python/R scripts
```

### Suggested Subreddits for Testing:

| Subreddit | Size | Characteristics | Posts |
|-----------|------|-----------------|-------|
| `python` | Large | Technical, active | 200-500 |
| `learnprogramming` | Large | Questions/answers | 200-500 |
| `datascience` | Medium | Professional | 100-300 |
| `LocalLLaMA` | Medium | Enthusiast | 100-200 |
| `webdev` | Large | Project-focused | 200-500 |

---

# config.py
# ----------------------------
# Global configuration for AI-Enhanced Social Network Analysis Project
# ----------------------------

# ----------------------------
# Reddit API credentials
# ----------------------------
REDDIT_CLIENT_ID = 't98RMTVBhwbRQOPMKxMOWQ'
REDDIT_CLIENT_SECRET = 'PRX4oePCR_jzmFh9SqlzOhyNH3XIDA'
REDDIT_USER_AGENT = "AI_SNA_Project"

# ----------------------------
# Gemini AI API configuration
# ----------------------------
GEMINI_API_KEY = 'AIzaSyAbR-d3ZIVfznl1tvifQ1forcmB6dXV2LY'
GEMINI_API_ENDPOINT = "https://api.gemini.ai/v1/analyze"  # Example endpoint
GEMINI_MODEL = "gemini-text-analytics"

# ----------------------------
# Data storage paths
# ----------------------------
DATA_PATH_RAW = "data/raw/"
DATA_PATH_PROCESSED = "data/processed/"
DATA_PATH_RESULTS = "data/results/"

# ----------------------------
# Graph analysis parameters
# ----------------------------
PAGE_RANK_ALPHA = 0.85
MAX_ITER = 100
MIN_COMMUNITY_SIZE = 3

# ----------------------------
# AI analysis parameters
# ----------------------------
TREND_THRESHOLD = 0.7
SENTIMENT_ANALYSIS = True
VIRAL_SCORE_THRESHOLD = 0.8

# ----------------------------
# Visualization parameters
# ----------------------------
GRAPH_NODE_COLOR = "skyblue"
GRAPH_EDGE_COLOR = "gray"
GRAPH_LAYOUT = "spring"
FIGURE_SIZE = (8, 6)

# ----------------------------
# General project settings
# ----------------------------
LOGGING_ENABLED = True
MAX_POSTS_FETCH = 100
RANDOM_SEED = 42

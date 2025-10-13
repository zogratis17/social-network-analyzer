import os
import sys
import json
import time
import math
import argparse
import logging
from collections import defaultdict, Counter
from datetime import datetime, timezone
from typing import Dict, List, Tuple, Set, Optional, Any, Union
import warnings

# Suppress warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Suppress gRPC/ALTS warnings from Google libraries
os.environ['GRPC_VERBOSITY'] = 'ERROR'
os.environ['GLOG_minloglevel'] = '2'

import logging
logging.getLogger("plotly").setLevel(logging.CRITICAL)
logging.getLogger("google").setLevel(logging.ERROR)
logging.getLogger("grpc").setLevel(logging.ERROR)


# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv is optional

# Third-party libs
try:
    import praw
    import networkx as nx
    import pandas as pd
    import numpy as np
    import plotly.graph_objects as go
    import matplotlib.pyplot as plt
    from tqdm import tqdm
except Exception as e:
    print("Missing dependencies. Please install requirements. Error:", e)
    raise

# ----- Logging -----
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger('ai_sn_analysis')

# ----- Utilities -----

def safe_mkdir(path):
    os.makedirs(path, exist_ok=True)


def now_ts():
    return datetime.now(timezone.utc).isoformat()

# ----- Configurable parameters -----
DEFAULTS = {
    'MIN_POSTS': 500,
    'PAGE_LIMIT': 100,
    'RATE_SLEEP': 1.0,  # seconds between requests to be polite
}

# ----- Gemini Client (Google Generative AI) -----
class GeminiClient:
    """
    Wrapper for Google Gemini API calls using the official google-generativeai SDK.
    If GEMINI_API_KEY is provided, uses Gemini for advanced content analysis.
    Falls back to local analysis if API key is not provided or API calls fail.
    
    Features caching to avoid redundant API calls.
    """
    def __init__(self, api_key=None):
        self.api_key = api_key
        self.model = None
        self._cache = {}  # Cache for analyzed texts
        
        if self.api_key:
            try:
                import google.generativeai as genai
                genai.configure(api_key=self.api_key)
                # Try different model names (API keeps changing)
                try:
                    self.model = genai.GenerativeModel('gemini-2.5-flash')
                except:
                    try:
                        self.model = genai.GenerativeModel('gemini-1.5-flash')
                    except:
                        self.model = genai.GenerativeModel('models/gemini-pro')
                logger.info('✅ Gemini API initialized successfully')
            except ImportError:
                logger.warning('google-generativeai not installed. Install with: pip install google-generativeai')
                logger.warning('Falling back to local text analysis')
            except Exception as e:
                logger.warning('Failed to initialize Gemini API: %s - falling back to local analyzer', e)

    def analyze_text(self, text: str) -> dict:
        """Analyze text using Gemini API or fallback to local analysis.
        
        Args:
            text: Input text to analyze
            
        Returns:
            dict with keys: sentiment, score, topics, viral_score
            
        Note: Includes defensive parsing for API responses and caching.
        """
        if not text or not text.strip():
            return {'sentiment': 'neutral', 'score': 0.5, 'topics': [], 'viral_score': 0.0}
        
        # Check cache first
        text_hash = hash(text[:500])  # Hash first 500 chars for cache key
        if text_hash in self._cache:
            return self._cache[text_hash]

        if self.model:
            try:
                # Create a detailed prompt for Gemini with focus on extracting MEANINGFUL topics
                prompt = f"""Analyze this Reddit post and extract SPECIFIC, MEANINGFUL topics.

IMPORTANT RULES FOR TOPIC EXTRACTION:
- Extract only SPECIFIC technical terms, technologies, libraries, frameworks, concepts
- Avoid generic words like "code", "help", "question", "post", "python", "programming"
- Focus on: library names (pandas, numpy), technologies (Docker, AWS), specific concepts (machine learning, web scraping)
- Use 1-3 word phrases for multi-word topics (e.g., "machine learning", "data visualization")
- Return 3-7 topics maximum
- Return empty list if no meaningful topics found

Content:
{text[:2000]}

Respond ONLY with valid JSON:
{{"sentiment": "positive/negative/neutral", "score": 0.0-1.0, "topics": ["specific_topic1", "specific_topic2"], "viral_score": 0.0-1.0}}

Example good topics: ["pandas", "asyncio", "web scraping", "type hints", "fastapi", "docker", "pytest"]
Example bad topics: ["python", "code", "help", "question", "programming", "post", "this"]"""

                response = self.model.generate_content(prompt)
                result_text = response.text.strip()
                
                # DEFENSIVE PARSING: Handle various response formats
                # Remove markdown code blocks if present
                if result_text.startswith('```'):
                    parts = result_text.split('```')
                    if len(parts) >= 2:
                        result_text = parts[1]
                        if result_text.startswith('json'):
                            result_text = result_text[4:]
                result_text = result_text.strip()
                
                # Parse JSON with error handling
                try:
                    data = json.loads(result_text)
                except json.JSONDecodeError:
                    # Try to extract JSON if embedded in other text
                    import re
                    json_match = re.search(r'\{[^}]+\}', result_text)
                    if json_match:
                        data = json.loads(json_match.group(0))
                    else:
                        raise
                
                # Validate and filter topics
                topics = data.get('topics', [])
                if not isinstance(topics, list):
                    topics = []
                
                # Filter out generic/stopword topics
                filtered_topics = [
                    str(t).lower().strip() 
                    for t in topics 
                    if t and isinstance(t, (str, int, float)) and len(str(t)) > 2 
                    and str(t).lower() not in TOPIC_STOPWORDS
                ][:7]
                
                # Validate and normalize the response with bounds checking
                sentiment = data.get('sentiment', 'neutral')
                if sentiment not in ['positive', 'negative', 'neutral']:
                    sentiment = 'neutral'
                
                score = float(data.get('score', 0.5))
                score = max(0.0, min(1.0, score))  # Clamp to [0, 1]
                
                viral_score = float(data.get('viral_score', 0.0))
                viral_score = max(0.0, min(1.0, viral_score))  # Clamp to [0, 1]
                
                result = {
                    'sentiment': sentiment.lower(),
                    'score': score,
                    'topics': filtered_topics,
                    'viral_score': viral_score
                }
                
                # Cache the result
                self._cache[text_hash] = result
                return result
                
            except json.JSONDecodeError as e:
                logger.warning('Failed to parse Gemini JSON response: %s - Response: %s', e, result_text[:200])
            except Exception as e:
                logger.warning('Gemini API call failed: %s - falling back to local analyzer', e)

        # Fallback: simple rule-based sentiment + topic extraction
        result = simple_local_text_analysis(text)
        self._cache[text_hash] = result
        return result


# ----- Simple local text analysis fallback -----
POS_WORDS = {'good', 'great', 'awesome', 'love', 'like', 'amazing', 'happy', 'fun', 'win', 'success'}
NEG_WORDS = {'bad', 'terrible', 'hate', 'sad', 'angry', 'fail', 'losing', 'awful', 'worse'}

# Comprehensive stopwords list for better topic extraction
TOPIC_STOPWORDS = set([
    'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
    'from', 'as', 'is', 'was', 'are', 'were', 'been', 'be', 'have', 'has', 'had', 'do', 'does',
    'did', 'will', 'would', 'should', 'could', 'may', 'might', 'must', 'can', 'this', 'that',
    'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'them', 'their', 'what',
    'which', 'who', 'when', 'where', 'why', 'how', 'all', 'each', 'every', 'both', 'few',
    'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so',
    'than', 'too', 'very', 's', 't', 'just', 'don', 'now', 'get', 'got', 'like', 'know',
    'think', 'see', 'make', 'go', 'take', 'come', 'want', 'use', 'find', 'give', 'tell',
    'work', 'call', 'try', 'ask', 'need', 'feel', 'become', 'leave', 'put', 'mean', 'keep',
    'let', 'begin', 'seem', 'help', 'show', 'hear', 'play', 'run', 'move', 'live', 'believe',
    'bring', 'happen', 'write', 'sit', 'stand', 'lose', 'pay', 'meet', 'include', 'continue',
    'set', 'learn', 'change', 'lead', 'understand', 'watch', 'follow', 'stop', 'create', 'speak',
    'read', 'spend', 'grow', 'open', 'walk', 'win', 'teach', 'offer', 'remember', 'consider',
    'appear', 'buy', 'serve', 'die', 'send', 'build', 'stay', 'fall', 'cut', 'reach', 'kill',
    'raise', 'pass', 'sell', 'decide', 'return', 'explain', 'hope', 'develop', 'carry', 'break',
    've', 'll', 're', 'm', 'd', 'http', 'https', 'www', 'com', 'org', 'one', 'two', 'also',
    'really', 'even', 'still', 'way', 'well', 'back', 'through', 'much', 'before', 'right',
    'little', 'long', 'good', 'new', 'first', 'last', 'own', 'great', 'old', 'different',
    'small', 'large', 'next', 'early', 'young', 'important', 'public', 'bad', 'able', 'post',
    'comment', 'reddit', 'subreddit', 'thread', 'user', 'people', 'thing', 'time', 'year',
    'day', 'week', 'month', 'ago', 'today', 'yesterday', 'am', 'pm', 'edit', 'deleted',
    'removed', 'question', 'answer', 'thanks', 'thank', 'please', 'anyone', 'someone',
    'something', 'anything', 'everything', 'nothing', 'yes', 'yeah', 'yep', 'nope', 'lol',
    'lmao', 'tbh', 'imo', 'imho', 'btw', 'fyi', 'aka', 'etc', 'eg', 'ie', 'vs'
])


# Python-specific technical terms and libraries (will help identify meaningful topics)
PYTHON_TECH_KEYWORDS = {
    'django', 'flask', 'fastapi', 'pandas', 'numpy', 'scipy', 'matplotlib', 'seaborn',
    'tensorflow', 'pytorch', 'keras', 'scikit-learn', 'sklearn', 'opencv', 'pillow',
    'requests', 'beautifulsoup', 'selenium', 'scrapy', 'asyncio', 'multiprocessing',
    'threading', 'pytest', 'unittest', 'docker', 'kubernetes', 'aws', 'azure', 'gcp',
    'sqlalchemy', 'mongodb', 'postgresql', 'mysql', 'redis', 'celery', 'rabbitmq',
    'pydantic', 'typer', 'click', 'streamlit', 'gradio', 'jupyter', 'notebook',
    'vscode', 'pycharm', 'anaconda', 'conda', 'poetry', 'pipenv', 'virtualenv',
    'type', 'typing', 'async', 'await', 'decorator', 'generator', 'iterator',
    'dataclass', 'enum', 'protocol', 'abc', 'metaclass', 'descriptor', 'api', 'rest',
    'graphql', 'websocket', 'http', 'server', 'client', 'database', 'orm', 'sql'
}

def simple_local_text_analysis(text):
    import re
    tokens = [t.strip('.,!?:;"()[]').lower() for t in text.split() if t]
    pos = sum(1 for t in tokens if t in POS_WORDS)
    neg = sum(1 for t in tokens if t in NEG_WORDS)
    score = 0.5
    if pos + neg > 0:
        score = (pos + 0.5) / (pos + neg + 1.0)
    sentiment = 'neutral'
    if score > 0.6:
        sentiment = 'positive'
    elif score < 0.4:
        sentiment = 'negative'

    # Advanced topic extraction
    text_clean = re.sub(r'http\S+|www\.\S+', '', text)  # Remove URLs
    text_clean = re.sub(r'\[.*?\]|\(.*?\)', '', text_clean)  # Remove markdown links
    
    # Extract multi-word phrases and single words
    words = text_clean.split()
    topics = []
    
    # 1. Look for known Python technical terms
    for token in tokens:
        if token in PYTHON_TECH_KEYWORDS:
            topics.append(token)
    
    # 2. Extract capitalized words (libraries, frameworks, proper nouns)
    for w in words:
        clean = re.sub(r'[^a-zA-Z0-9]', '', w)
        if clean and len(clean) > 2 and clean[0].isupper() and clean.lower() not in TOPIC_STOPWORDS:
            # Check if it's a known library or framework pattern
            if clean.lower() in PYTHON_TECH_KEYWORDS or (len(clean) > 4 and clean.isalpha()):
                topics.append(clean.lower())
    
    # 3. Extract meaningful longer words
    for t in tokens:
        if (len(t) > 5 and t not in TOPIC_STOPWORDS and t.isalpha() 
            and not t.startswith(('http', 'www')) and t not in topics):
            topics.append(t)
    
    # 4. Extract 2-word technical phrases
    for i in range(len(tokens) - 1):
        if (len(tokens[i]) > 3 and len(tokens[i+1]) > 3 
            and tokens[i] not in TOPIC_STOPWORDS 
            and tokens[i+1] not in TOPIC_STOPWORDS):
            phrase = f"{tokens[i]} {tokens[i+1]}"
            if len(phrase) < 30:  # Reasonable phrase length
                topics.append(phrase)
    
    # Remove duplicates and filter
    unique_topics = list(dict.fromkeys(topics))  # Preserves order
    filtered = [t for t in unique_topics if len(t) > 2 and t not in TOPIC_STOPWORDS][:7]

    viral_score = min(1.0, (len(tokens) / 50.0) * (1.0 + 0.5 * (pos - neg)))
    viral_score = max(0.0, viral_score)

    return {'sentiment': sentiment, 'score': score, 'topics': filtered, 'viral_score': viral_score}

# ----- Reddit Data Collector -----
class RedditCollector:
    def __init__(self, client_id, client_secret, user_agent, rate_sleep=2.0):
        """Initialize Reddit API client with rate limiting.
        
        Args:
            rate_sleep: Base sleep time between requests (default 2.0s for API compliance)
        """
        self.reddit = praw.Reddit(client_id=client_id,
                                  client_secret=client_secret,
                                  user_agent=user_agent)
        self.rate_sleep = rate_sleep
        self.retry_count = 0
        self.max_retries = 3

    def fetch_subreddit_posts(self, subreddit_name, limit=500, time_filter='all'):
        """Fetch posts and comments with exponential backoff on rate limits.
        
        Returns list of post dicts with comments nested.
        """
        logger.info('Fetching up to %s posts from r/%s', limit, subreddit_name)
        try:
            subreddit = self.reddit.subreddit(subreddit_name)
        except Exception as e:
            logger.error(f'Failed to access r/{subreddit_name}: {e}')
            return []
        
        posts = []
        count = 0
        
        for submission in subreddit.top(time_filter=time_filter, limit=limit):
            try:
                # Fetch comments with retry logic
                retry_delay = self.rate_sleep
                comments_loaded = False
                
                for attempt in range(self.max_retries):
                    try:
                        submission.comments.replace_more(limit=0)  # limit=0 to avoid too many requests
                        comments_loaded = True
                        break  # Success, exit retry loop
                    except Exception as e:
                        if attempt < self.max_retries - 1:
                            logger.warning(f'Retry {attempt+1}/{self.max_retries} for post {submission.id}: {str(e)[:50]}')
                            time.sleep(retry_delay)
                            retry_delay *= 2  # Exponential backoff
                        else:
                            logger.error(f'Failed to fetch comments for post {submission.id} after {self.max_retries} attempts')
                
                # Build post data
                post = {
                    'id': submission.id,
                    'created_utc': submission.created_utc,
                    'title': submission.title,
                    'selftext': submission.selftext,
                    'author': getattr(submission.author, 'name', None),
                    'score': submission.score,
                    'num_comments': submission.num_comments,
                    'comments': []
                }
                
                # Add comments if loaded successfully
                if comments_loaded:
                    for c in submission.comments.list():
                        try:
                            post['comments'].append({
                                'id': c.id,
                                'parent_id': c.parent_id,
                                'created_utc': c.created_utc,
                                'body': getattr(c, 'body', ''),
                                'author': getattr(c.author, 'name', None),
                                'score': getattr(c, 'score', 0)
                            })
                        except Exception as e:
                            # Skip malformed comments
                            continue
                
                posts.append(post)
                count += 1
                
                if count % 10 == 0:
                    logger.info('Fetched %s/%s posts...', count, limit)
                
                time.sleep(self.rate_sleep)
                
            except Exception as e:
                logger.warning(f'Skipping post due to error: {str(e)[:100]}')
                continue
        
        logger.info('Finished fetching %s posts', len(posts))
        return posts

# ----- Graph Construction -----
class GraphBuilder:
    def __init__(self, directed=True):
        self.directed = directed
        self.G = nx.DiGraph() if directed else nx.Graph()

    def build_from_posts(self, posts):
        """Posts: list of post dicts with comments nested. Build nodes (users) and edges (interactions)
        Edge semantics: reply edge from commenter to parent author (comment->parent_author)
        Also create mention edges if '@username' appears in body (basic)
        
        OPTIMIZATION: Uses O(1) comment lookup via dictionary instead of O(n) linear search.
        """
        logger.info('Building graph from posts...')
        for post in posts:
            post_author = post.get('author')
            if post_author:
                self._add_node(post_author)
                self.G.nodes[post_author].setdefault('posts', 0)
                self.G.nodes[post_author]['posts'] += 1
            
            # OPTIMIZATION: Build comment ID -> comment dict for O(1) lookup (was O(n))
            comment_dict = {c['id']: c for c in post.get('comments', []) if 'id' in c}
            
            for c in post.get('comments', []):
                author = c.get('author')
                parent_id = c.get('parent_id')
                body = c.get('body', '')
                timestamp = c.get('created_utc')
                if author:
                    self._add_node(author)
                # parent can be t1_<commentid> or t3_<postid>
                parent_author = None
                if parent_id and parent_id.startswith('t1_'):
                    # O(1) lookup instead of O(n) next() scan
                    pid = parent_id.split('_', 1)[1]
                    parent_comment = comment_dict.get(pid)
                    if parent_comment:
                        parent_author = parent_comment.get('author')
                elif parent_id and parent_id.startswith('t3_'):
                    parent_author = post_author

                if author and parent_author and parent_author != author:
                    self._add_edge(author, parent_author, interaction='reply', timestamp=timestamp)

                # mentions
                mentions = self._extract_mentions(body)
                for m in mentions:
                    if m != author:
                        self._add_edge(author, m, interaction='mention', timestamp=timestamp)

                # add comment attributes to node
                if author:
                    self.G.nodes[author].setdefault('comments', 0)
                    self.G.nodes[author]['comments'] += 1
        logger.info('Graph built: %s nodes, %s edges', self.G.number_of_nodes(), self.G.number_of_edges())
        return self.G

    def _add_node(self, username):
        if username is None:
            return
        if username not in self.G:
            self.G.add_node(username, first_seen=now_ts())

    def _add_edge(self, a, b, interaction='reply', timestamp=None):
        if a is None or b is None:
            return
        if self.G.has_edge(a, b):
            self.G[a][b]['weight'] += 1
            self.G[a][b].setdefault('interactions', []).append(interaction)
        else:
            self.G.add_edge(a, b, weight=1, interactions=[interaction], first_seen=now_ts())
        if timestamp:
            self.G[a][b].setdefault('timestamps', []).append(timestamp)

    def _extract_mentions(self, text):
        # crude mention extraction: look for r"u/username" or @username patterns
        mentions = set()
        if not text:
            return mentions
        parts = text.split()
        for p in parts:
            p = p.strip('.,:;()[]"')
            if p.startswith('u/'):
                mentions.add(p.split('/', 1)[1])
            elif p.startswith('@'):
                mentions.add(p[1:])
        return mentions

    def export_graphml(self, path):
        # GraphML doesn't support list types, so convert them to strings
        G_copy = self.G.copy()
        for u, v, data in G_copy.edges(data=True):
            for key, value in list(data.items()):
                if isinstance(value, list):
                    data[key] = str(value)
        for node, data in G_copy.nodes(data=True):
            for key, value in list(data.items()):
                if isinstance(value, list):
                    data[key] = str(value)
        nx.write_graphml(G_copy, path)

    def export_json(self, path):
        data = nx.node_link_data(self.G)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)

# ----- Union-Find (Disjoint Set) -----
class UnionFind:
    """Union-Find (Disjoint Set) data structure for detecting connected components.
    
    Implements path compression and union by rank for optimal performance.
    """
    
    def __init__(self) -> None:
        self.parent: Dict[Any, Any] = {}
        self.rank: Dict[Any, int] = {}

    def find(self, x: Any) -> Any:
        """Find the root of the set containing x (with path compression)."""
        if self.parent.get(x, x) != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent.get(x, x)

    def union(self, x: Any, y: Any) -> None:
        """Unite the sets containing x and y (with union by rank)."""
        xroot = self.find(x)
        yroot = self.find(y)
        if xroot == yroot:
            return
        rx = self.rank.get(xroot, 0)
        ry = self.rank.get(yroot, 0)
        if rx < ry:
            self.parent[xroot] = yroot
        elif rx > ry:
            self.parent[yroot] = xroot
        else:
            self.parent[yroot] = xroot
            self.rank[xroot] = rx + 1
    
    def add(self, x: Any) -> None:
        """Add a new element to the Union-Find structure."""
        if x not in self.parent:
            self.parent[x] = x
            self.rank[x] = 0

    def build_from_graph(self, G: nx.Graph) -> Dict[Any, List[Any]]:
        """Build connected components from NetworkX graph.
        
        Returns:
            Dict mapping root node -> list of members in that component
        """
        for n in G.nodes():
            self.parent[n] = n
            self.rank[n] = 0
        for u, v in G.edges():
            self.union(u, v)
        # produce mapping
        groups = defaultdict(list)
        for n in G.nodes():
            groups[self.find(n)].append(n)
        return groups

# ----- Community Detection -----
def detect_communities_union_find(G):
    """Detect communities using Union-Find algorithm.
    
    NOTE: Union-Find finds CONNECTED COMPONENTS, not modular communities.
    In highly connected social networks, this often results in one large component.
    For meaningful sub-community detection, use Greedy Modularity instead.
    
    Use case: Identifying isolated clusters and network fragmentation.
    """
    logger.info('Detecting connected components via Union-Find')
    uf = UnionFind()
    groups = uf.build_from_graph(G.to_undirected())
    # assign community ids
    comm_assign = {}
    for cid, (root, members) in enumerate(groups.items()):
        for m in members:
            comm_assign[m] = cid
    nx.set_node_attributes(G, comm_assign, 'community_uf')  # Label: connected components
    logger.info('Detected %s connected components (Union-Find)', len(groups))
    return comm_assign


def detect_communities_louvain(G):
    logger.info('Detecting communities via greedy modularity (NetworkX)')
    try:
        from networkx.algorithms.community import greedy_modularity_communities
        communities = list(greedy_modularity_communities(G.to_undirected()))
        comm_assign = {}
        for cid, comm in enumerate(communities):
            for n in comm:
                comm_assign[n] = cid
        nx.set_node_attributes(G, comm_assign, 'community_greedy')
        logger.info('Detected %s communities (greedy modularity)', len(communities))
        return comm_assign
    except Exception as e:
        logger.warning('Greedy modularity failed: %s', e)
        return {}

# ----- Influence Analysis -----
def compute_pagerank(G, damping=0.85, max_iter=100, tol=1e-06):
    logger.info('Computing PageRank...')
    # PageRank on directed graph
    try:
        pr = nx.pagerank(G, alpha=damping, max_iter=max_iter, tol=tol)
    except Exception as e:
        logger.warning('NetworkX PageRank failed: %s - trying personalization-less fallback', e)
        pr = nx.pagerank(G.to_undirected(), alpha=damping)
    nx.set_node_attributes(G, pr, 'pagerank')
    # Store degree centrality
    degree = dict(G.degree())
    nx.set_node_attributes(G, degree, 'degree')
    logger.info('PageRank computed for %s nodes', len(pr))
    return pr

# ----- MST -----
def compute_mst(G):
    """Compute Minimum Spanning Tree using Kruskal's algorithm.
    
    NOTE: Edge weights are set as inverse of interaction count, so MST finds
    the strongest connections (most interactions = lowest weight).
    This makes MST meaningful for identifying core communication backbone.
    """
    logger.info('Computing MST (undirected)')
    U = G.to_undirected()
    
    # Set meaningful edge weights: inverse of interaction count
    # More interactions = lower weight = preferred in MST
    for u, v, data in U.edges(data=True):
        interaction_count = data.get('weight', 1)
        # Use inverse so MST prefers strong connections
        # Add small constant to avoid division by zero
        U[u][v]['mst_weight'] = 1.0 / (interaction_count + 0.1)
    
    mst = nx.minimum_spanning_tree(U, weight='mst_weight')
    logger.info('MST has %s nodes and %s edges (based on strongest interactions)', 
                mst.number_of_nodes(), mst.number_of_edges())
    return mst

# ----- Trend Detection -----
def detect_trends(posts, content_analysis_results, window_days=7, top_n=15, min_topic_threshold=10):
    """Advanced trending detection using multiple methods:
    1. AI topic extraction (from content_analysis_results)
    2. TF-IDF for important terms
    3. N-gram analysis for multi-word topics
    4. Velocity-based trending
    
    Args:
        posts: List of post dictionaries
        content_analysis_results: Dict mapping post_id -> analysis dict with topics
        window_days: Time window for trend calculation (default: 7 days)
        top_n: Number of top trends to return (default: 15)
        min_topic_threshold: Minimum number of topics needed for AI-based detection (default: 10)
    
    Returns:
        List of (topic, metrics_dict) tuples sorted by importance
    """
    logger.info('Detecting trends with advanced analysis...')
    
    # Words to filter out from trending (too generic or meta)
    FILTER_OUT = {'python', 'reddit', 'subreddit', 'upvote', 'upvotes', 'downvote', 
                  'post', 'posts', 'comment', 'comments', 'thats', 'im', 'dont', 'didnt',
                  'isnt', 'wasnt', 'havent', 'hasnt', 'youre', 'theyre', 'whats'}
    
    # Collect all topics from AI analysis
    gemini_topics = []
    for pid, analysis in content_analysis_results.items():
        topics = analysis.get('topics', [])
        gemini_topics.extend([t.lower().strip() for t in topics if t and len(t) > 2])
    
    # If we have sufficient AI topics, use them as primary source
    if gemini_topics and len(gemini_topics) > min_topic_threshold:
        logger.info(f'Using {len(gemini_topics)} topics extracted by AI')
        topic_times = defaultdict(list)
        
        # Map topics to timestamps
        for post in posts:
            ts = post.get('created_utc') or time.time()
            analysis = content_analysis_results.get(post['id'], {})
            for topic in analysis.get('topics', []):
                topic = topic.lower().strip()
                if topic and len(topic) > 2 and topic not in FILTER_OUT:
                    topic_times[topic].append(ts)
        
        # Calculate velocity
        now = time.time()
        window = window_days * 24 * 3600
        topic_scores = {}
        
        for topic, times in topic_times.items():
            recent = sum(1 for t in times if t >= now - window)
            older = sum(1 for t in times if t < now - window)
            total = len(times)
            
            # Velocity score: recent activity vs historical
            velocity = (recent + 1) / (older + 1) if older >= 0 else recent
            
            # Importance score: combines frequency and velocity
            importance = total * (1 + velocity * 0.5)
            
            topic_scores[topic] = {
                'recent': recent,
                'older': older,
                'total': total,
                'velocity': round(velocity, 2),
                'importance': round(importance, 2)
            }
        
        # Sort by importance (frequency + velocity)
        trending = sorted(topic_scores.items(), 
                         key=lambda x: (x[1]['importance'], x[1]['recent']), 
                         reverse=True)
        
        return trending[:top_n]
    
    # Fallback: Extract topics using TF-IDF and N-grams
    logger.info('Gemini topics not available, using TF-IDF and N-gram analysis')
    return extract_topics_tfidf(posts, window_days, top_n, FILTER_OUT)


def extract_topics_tfidf(posts, window_days=7, top_n=15, filter_out=None):
    """Extract trending topics using TF-IDF and N-gram analysis"""
    import re
    from collections import Counter
    
    if filter_out is None:
        filter_out = set()
    
    # Combine all text from posts
    all_texts = []
    post_times = []
    
    for post in posts:
        text = (post.get('title', '') or '') + '\n' + (post.get('selftext', '') or '')
        # Add top comments
        for c in post.get('comments', [])[:5]:
            text += '\n' + (c.get('body') or '')
        all_texts.append(text.lower())
        post_times.append(post.get('created_utc') or time.time())
    
    # Extract meaningful terms (2-3 word phrases + single important words)
    def extract_ngrams(text, n_range=(1, 3)):
        # Clean text
        text = re.sub(r'http\S+|www\.\S+', '', text)  # Remove URLs
        text = re.sub(r'[^a-z0-9\s]', ' ', text)  # Remove special chars
        words = [w for w in text.split() if len(w) > 2 and w not in TOPIC_STOPWORDS and w not in filter_out]
        
        ngrams = []
        # Unigrams (single words) - prioritize technical terms
        for w in words:
            if len(w) > 4 or w in PYTHON_TECH_KEYWORDS:
                ngrams.append(w)
        
        # Bigrams (2-word phrases)
        for i in range(len(words) - 1):
            bigram = f"{words[i]} {words[i+1]}"
            if len(bigram) > 8 and len(bigram) < 30:
                ngrams.append(bigram)
        
        # Trigrams (3-word phrases) - only if meaningful
        for i in range(len(words) - 2):
            trigram = f"{words[i]} {words[i+1]} {words[i+2]}"
            if len(trigram) > 15 and len(trigram) < 40:
                ngrams.append(trigram)
        
        return ngrams
    
    # Extract n-grams from all texts
    all_ngrams = []
    ngram_times = defaultdict(list)
    
    for text, timestamp in zip(all_texts, post_times):
        ngrams = extract_ngrams(text)
        all_ngrams.extend(ngrams)
        for ng in ngrams:
            ngram_times[ng].append(timestamp)
    
    # Calculate TF-IDF-like scores
    ngram_freq = Counter(all_ngrams)
    
    # Filter: must appear at least 2 times and not in filter list
    ngram_freq = {k: v for k, v in ngram_freq.items() if v >= 2 and k not in filter_out}
    
    # Calculate velocity and importance
    now = time.time()
    window = window_days * 24 * 3600
    topic_scores = {}
    
    for ngram, freq in ngram_freq.items():
        times = ngram_times[ngram]
        recent = sum(1 for t in times if t >= now - window)
        older = sum(1 for t in times if t < now - window)
        
        velocity = (recent + 1) / (older + 1)
        importance = freq * (1 + velocity * 0.5)
        
        topic_scores[ngram] = {
            'recent': recent,
            'older': older,
            'total': freq,
            'velocity': round(velocity, 2),
            'importance': round(importance, 2)
        }
    
    # Sort by importance
    trending = sorted(topic_scores.items(), 
                     key=lambda x: (x[1]['importance'], x[1]['total']), 
                     reverse=True)
    
    return trending[:top_n]

# ----- Visualization -----

def visualize_graph_plotly(G, community_attr='community_greedy', size_attr='pagerank', out_html=None):
    logger.info('Generating interactive Plotly visualization...')
    pos = nx.spring_layout(G, seed=42, k=None)
    node_x = []
    node_y = []
    node_text = []
    sizes = []
    colors = []
    # Get community attributes with fallback
    comms = nx.get_node_attributes(G, community_attr)
    if not comms:
        # Fallback to assigning all nodes to community 0
        comms = {n: 0 for n in G.nodes()}
    for n in G.nodes():
        x, y = pos[n]
        node_x.append(x)
        node_y.append(y)
        pr = G.nodes[n].get(size_attr, 0.0)
        if pr is None:
            pr = 0.0
        pr = float(pr)
        sizes.append(10 + pr * 100)
        c = comms.get(n, 0)
        colors.append(c)
        degree = G.nodes[n].get('degree', G.degree(n))
        text = f"{n}<br>pr={pr:.4f}<br>comm={c}<br>degree={degree}"
        node_text.append(text)

    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(x=edge_x, y=edge_y, mode='lines', line=dict(width=0.5, color='#888'), hoverinfo='none')
    node_trace = go.Scatter(x=node_x, y=node_y, mode='markers', hoverinfo='text', text=node_text,
                            marker=dict(showscale=True, 
                                      color=colors, 
                                      size=sizes, 
                                      colorscale='Plasma',  # Vibrant purple->pink->orange->yellow
                                      colorbar=dict(title='Community')))
    fig = go.Figure(data=[edge_trace, node_trace], layout=go.Layout(title='Social Network Graph', hovermode='closest'))
    if out_html:
        fig.write_html(out_html)
        logger.info('Saved interactive visualization to %s', out_html)
    return fig

# ----- Export Helpers -----

def export_node_table(G, path):
    rows = []
    for n, d in G.nodes(data=True):
        row = {'user': n}
        row.update(d)
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    logger.info('Exported node table to %s', path)


def export_edge_table(G, path):
    rows = []
    for u, v, d in G.edges(data=True):
        row = {'source': u, 'target': v}
        row.update(d)
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    logger.info('Exported edge table to %s', path)

# ----- End-to-end pipeline -----

def run_pipeline(subreddit, posts_limit, outdir, gemini_api_key=None, time_filter='all'):
    safe_mkdir(outdir)
    # 1) Collect
    collector = RedditCollector(client_id=os.environ.get('REDDIT_CLIENT_ID'),
                                client_secret=os.environ.get('REDDIT_CLIENT_SECRET'),
                                user_agent=os.environ.get('REDDIT_USER_AGENT', 'ai-sn-analysis/0.1'),
                                rate_sleep=DEFAULTS['RATE_SLEEP'])
    posts = collector.fetch_subreddit_posts(subreddit, limit=posts_limit, time_filter=time_filter)
    raw_path = os.path.join(outdir, f'{subreddit}_raw_posts.json')
    with open(raw_path, 'w', encoding='utf-8') as f:
        json.dump(posts, f, indent=2)
    logger.info('Saved raw posts to %s', raw_path)

    # 2) Build Graph
    gb = GraphBuilder(directed=True)
    G = gb.build_from_posts(posts)
    gb.export_graphml(os.path.join(outdir, f'{subreddit}_graph.graphml'))
    gb.export_json(os.path.join(outdir, f'{subreddit}_graph.json'))

    # 3) Community detection
    comm_uf = detect_communities_union_find(G)
    comm_greedy = detect_communities_louvain(G)

    # 4) Influence analysis
    pr = compute_pagerank(G)

    # 5) MST
    mst = compute_mst(G)
    if mst is not None:
        # Convert list attributes to strings for GraphML compatibility
        mst_copy = mst.copy()
        for u, v, data in mst_copy.edges(data=True):
            for key, value in list(data.items()):
                if isinstance(value, list):
                    data[key] = str(value)
        for node, data in mst_copy.nodes(data=True):
            for key, value in list(data.items()):
                if isinstance(value, list):
                    data[key] = str(value)
        nx.write_graphml(mst_copy, os.path.join(outdir, f'{subreddit}_mst.graphml'))

    # 6) AI Content Analysis
    gem = GeminiClient(api_key=gemini_api_key)
    content_results = {}
    logger.info('Analyzing content with Gemini fallback...')
    for p in tqdm(posts, desc='ContentAnalysis'):
        # analyze title + selftext + top comments
        text = (p.get('title','') or '') + '\n' + (p.get('selftext','') or '')
        # add top 3 comments
        for c in p.get('comments', [])[:3]:
            text += '\n' + (c.get('body') or '')
        res = gem.analyze_text(text)
        content_results[p['id']] = res
    # save content analysis
    with open(os.path.join(outdir, f'{subreddit}_content_analysis.json'), 'w', encoding='utf-8') as f:
        json.dump(content_results, f, indent=2)

    # 7) Trend detection (pass Gemini client for better topic extraction)
    trends = detect_trends(posts, content_results, gemini_client=gem)
    with open(os.path.join(outdir, f'{subreddit}_trends.json'), 'w', encoding='utf-8') as f:
        json.dump(trends, f, indent=2)

    # 8) Exports (nodes/edges)
    export_node_table(G, os.path.join(outdir, f'{subreddit}_nodes.csv'))
    export_edge_table(G, os.path.join(outdir, f'{subreddit}_edges.csv'))

    # 9) Visualize
    fig = visualize_graph_plotly(G, community_attr='community_greedy', size_attr='pagerank', out_html=os.path.join(outdir, f'{subreddit}_graph.html'))

    logger.info('Pipeline complete. Outputs in %s', outdir)
    return {
        'graph': G,
        'posts': posts,
        'content_analysis': content_results,
        'trends': trends,
        'fig': fig
    }

# ----- CLI -----

def parse_args():
    parser = argparse.ArgumentParser(description='AI-Enhanced Social Network Analysis Prototype')
    parser.add_argument('--subreddit', type=str, required=True)
    parser.add_argument('--posts', type=int, default=500)
    parser.add_argument('--outdir', type=str, default='output')
    parser.add_argument('--gemini-key', type=str, default=os.environ.get('GEMINI_API_KEY'))
    parser.add_argument('--time-filter', type=str, default='all', choices=['all','year','month','week','day'])
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    # basic env check
    if not os.environ.get('REDDIT_CLIENT_ID') or not os.environ.get('REDDIT_CLIENT_SECRET'):
        logger.error('REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET must be set in environment variables')
        sys.exit(1)
    outputs = run_pipeline(args.subreddit, args.posts, args.outdir, gemini_api_key=args.gemini_key, time_filter=args.time_filter)



# ----- End of file -----
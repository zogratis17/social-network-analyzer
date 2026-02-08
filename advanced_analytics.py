"""
Advanced analytics features for social network analysis.
Includes temporal analysis, sentiment-weighted networks, multi-layer graphs,
validation metrics, and predictive models.
"""
import os
import json
import logging
from datetime import datetime, timedelta, timezone
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Any
import re

import networkx as nx
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

logger = logging.getLogger('SocialNetworkAnalyzer')

# ===== DATA APPENDING STRATEGY =====

def should_append_data(subreddit, output_dir, time_threshold_days=7):
    """
    Decide whether to append or replace based on time and data quality.
    
    Args:
        subreddit: Subreddit name
        output_dir: Output directory path
        time_threshold_days: Days threshold for considering data "old"
    
    Returns:
        Tuple of (should_append: bool, reason: str)
    """
    existing_file = os.path.join(output_dir, f'{subreddit}_raw_posts.json')
    
    if not os.path.exists(existing_file):
        return False, "No existing data found"
    
    try:
        # Load existing data
        with open(existing_file, 'r', encoding='utf-8') as f:
            existing_posts = json.load(f)
        
        if not existing_posts:
            return False, "Existing data is empty"
        
        # Check time difference
        latest_timestamp = max(p['created_utc'] for p in existing_posts if 'created_utc' in p)
        latest_date = datetime.fromtimestamp(latest_timestamp, timezone.utc)
        days_old = (datetime.now(timezone.utc) - latest_date).days
        
        if days_old < time_threshold_days:
            return False, f"Recent analysis ({days_old} days old) - data too similar"
        
        return True, f"Old analysis ({days_old} days old) - safe to append"
    
    except Exception as e:
        logger.warning(f"Error checking existing data: {e}")
        return False, f"Error reading existing data: {e}"

def merge_posts_data(old_posts, new_posts):
    """
    Merge old and new posts, avoiding duplicates.
    
    Args:
        old_posts: List of existing posts
        new_posts: List of newly fetched posts
    
    Returns:
        Merged list of posts
    """
    existing_ids = {p['id'] for p in old_posts if 'id' in p}
    unique_new_posts = [p for p in new_posts if p.get('id') not in existing_ids]
    
    merged = old_posts + unique_new_posts
    
    logger.info(f"Merged posts: {len(old_posts)} old + {len(unique_new_posts)} new = {len(merged)} total")
    
    return merged

def merge_graphs(G_old, G_new):
    """
    Intelligently merge two graphs, preserving and updating attributes.
    
    Args:
        G_old: Existing graph
        G_new: Newly built graph
    
    Returns:
        Merged graph
    """
    G_merged = G_old.copy()
    
    # Merge nodes
    for node in G_new.nodes():
        if node not in G_merged:
            # Add new node with all attributes
            G_merged.add_node(node, **G_new.nodes[node])
        else:
            # Update/merge attributes
            for attr, value in G_new.nodes[node].items():
                if attr in ['posts', 'comments']:
                    # Accumulate counts
                    G_merged.nodes[node][attr] = G_merged.nodes[node].get(attr, 0) + value
                else:
                    # Update with new value (for metrics like pagerank)
                    G_merged.nodes[node][attr] = value
    
    # Merge edges
    for u, v, data in G_new.edges(data=True):
        if G_merged.has_edge(u, v):
            # Increase interaction weight
            G_merged[u][v]['weight'] = G_merged[u][v].get('weight', 1) + data.get('weight', 1)
        else:
            # Add new edge
            G_merged.add_edge(u, v, **data)
    
    logger.info(f"Merged graph: {len(G_merged.nodes())} nodes, {len(G_merged.edges())} edges")
    
    return G_merged

# ===== VALIDATION METRICS =====

def validate_influence_metrics(G, posts):
    """
    Validate PageRank against actual engagement metrics.
    
    Args:
        G: NetworkX graph with pagerank attributes
        posts: List of post dictionaries
    
    Returns:
        Dictionary with validation results and correlations
    """
    validation = []
    pagerank_scores = nx.get_node_attributes(G, 'pagerank')
    
    # Calculate actual influence metrics per user
    user_metrics = defaultdict(lambda: {'scores': [], 'comments': [], 'posts_count': 0})
    
    for post in posts:
        author = post.get('author')
        if author and author != '[deleted]':
            user_metrics[author]['scores'].append(post.get('score', 0))
            user_metrics[author]['comments'].append(post.get('num_comments', 0))
            user_metrics[author]['posts_count'] += 1
    
    # Build validation dataset
    for user, metrics in user_metrics.items():
        if user in pagerank_scores:
            avg_score = np.mean(metrics['scores']) if metrics['scores'] else 0
            avg_comments = np.mean(metrics['comments']) if metrics['comments'] else 0
            
            validation.append({
                'user': user,
                'pagerank': pagerank_scores[user],
                'avg_upvotes': avg_score,
                'avg_comments': avg_comments,
                'total_posts': metrics['posts_count'],
                'actual_influence': avg_score + avg_comments  # Combined metric
            })
    
    if not validation:
        return None
    
    # Calculate correlations
    df = pd.DataFrame(validation)
    correlations = {
        'pagerank_vs_upvotes': df['pagerank'].corr(df['avg_upvotes']),
        'pagerank_vs_comments': df['pagerank'].corr(df['avg_comments']),
        'pagerank_vs_combined': df['pagerank'].corr(df['actual_influence'])
    }
    
    # Determine accuracy level
    combined_corr = correlations['pagerank_vs_combined']
    if combined_corr > 0.7:
        accuracy = 'High'
    elif combined_corr > 0.5:
        accuracy = 'Medium'
    else:
        accuracy = 'Low'
    
    result = {
        'validation_data': validation,
        'correlations': correlations,
        'accuracy': accuracy,
        'sample_size': len(validation)
    }
    
    logger.info(f"Validation: PageRank accuracy = {accuracy} (correlation = {combined_corr:.3f})")
    
    return result

# ===== SENTIMENT-WEIGHTED EDGES =====

def build_sentiment_weighted_graphs(posts, content_analysis):
    """
    Build separate graphs for positive, negative, and neutral interactions.
    
    Args:
        posts: List of posts
        content_analysis: Dictionary of content analysis results
    
    Returns:
        Tuple of (G_positive, G_negative, G_neutral)
    """
    G_positive = nx.DiGraph()
    G_negative = nx.DiGraph()
    G_neutral = nx.DiGraph()
    
    for post in posts:
        author = post.get('author')
        if not author or author == '[deleted]':
            continue
        
        # Get post sentiment
        post_analysis = content_analysis.get(post['id'], {})
        sentiment_type = post_analysis.get('sentiment', 'neutral')
        
        # Ensure nodes exist
        for G in [G_positive, G_negative, G_neutral]:
            if author not in G:
                G.add_node(author, posts=0, comments=0)
            G.nodes[author]['posts'] += 1
        
        # Process comments
        for comment in post.get('comments', []):
            commenter = comment.get('author')
            if not commenter or commenter == '[deleted]':
                continue
            
            # Determine which graph based on sentiment
            if sentiment_type == 'positive':
                G = G_positive
            elif sentiment_type == 'negative':
                G = G_negative
            else:
                G = G_neutral
            
            # Add/update edge
            if commenter not in G:
                G.add_node(commenter, posts=0, comments=0)
            G.nodes[commenter]['comments'] += 1
            
            if G.has_edge(commenter, author):
                G[commenter][author]['weight'] += 1
            else:
                G.add_edge(commenter, author, weight=1)
    
    logger.info(f"Sentiment graphs: Positive={len(G_positive.nodes())}, Negative={len(G_negative.nodes())}, Neutral={len(G_neutral.nodes())}")
    
    return G_positive, G_negative, G_neutral

def compare_sentiment_networks(G_pos, G_neg, G_neu):
    """
    Compare positive, negative, and neutral networks.
    
    Returns:
        Dictionary with comparison metrics
    """
    comparison = {}
    
    for name, G in [('positive', G_pos), ('negative', G_neg), ('neutral', G_neu)]:
        if len(G) > 0:
            comparison[f'{name}_network'] = {
                'nodes': len(G.nodes()),
                'edges': len(G.edges()),
                'density': float(nx.density(G)),
                'avg_clustering': float(nx.average_clustering(G.to_undirected()))
            }
            
            # Top users in this network
            if len(G) > 0:
                pr = nx.pagerank(G)
                top_users = sorted(pr.items(), key=lambda x: x[1], reverse=True)[:5]
                comparison[f'{name}_network']['top_users'] = [
                    {'user': u, 'score': float(s)} for u, s in top_users
                ]
    
    # Find controversial users (high in both positive and negative)
    if len(G_pos) > 0 and len(G_neg) > 0:
        pr_pos = nx.pagerank(G_pos)
        pr_neg = nx.pagerank(G_neg)
        
        common_users = set(pr_pos.keys()) & set(pr_neg.keys())
        controversial = [
            {'user': u, 'positive_score': float(pr_pos[u]), 'negative_score': float(pr_neg[u])}
            for u in common_users
            if pr_pos[u] > 0.001 and pr_neg[u] > 0.001
        ]
        controversial.sort(key=lambda x: x['positive_score'] + x['negative_score'], reverse=True)
        comparison['controversial_users'] = controversial[:10]
    
    return comparison

# ===== TEMPORAL ANALYSIS =====

def build_temporal_graphs(posts, time_windows=['day', 'week']):
    """
    Build graphs across different time windows.
    
    Args:
        posts: List of posts
        time_windows: List of time window types ('day', 'week', 'month')
    
    Returns:
        Dictionary of {period_name: {graph, start, end, posts_count}}
    """
    from ai_sn_analysis_prototype import GraphBuilder
    
    graphs_timeline = {}
    now = datetime.now(timezone.utc)
    
    for window in time_windows:
        if window == 'day':
            delta = timedelta(days=1)
            periods = 7  # Last 7 days
        elif window == 'week':
            delta = timedelta(weeks=1)
            periods = 4  # Last 4 weeks
        else:  # month
            delta = timedelta(days=30)
            periods = 3  # Last 3 months
        
        for i in range(periods):
            start = now - delta * (i + 1)
            end = now - delta * i
            
            # Filter posts by time range
            period_posts = [
                p for p in posts
                if 'created_utc' in p and 
                start <= datetime.fromtimestamp(p['created_utc'], timezone.utc) < end
            ]
            
            if period_posts:
                builder = GraphBuilder()
                G = builder.build_from_posts(period_posts)
                
                graphs_timeline[f"{window}_{i}"] = {
                    'graph': G,
                    'start': start.isoformat(),
                    'end': end.isoformat(),
                    'posts_count': len(period_posts),
                    'nodes_count': len(G.nodes()),
                    'edges_count': len(G.edges())
                }
    
    logger.info(f"Built {len(graphs_timeline)} temporal graphs")
    return graphs_timeline

def detect_community_evolution(graphs_timeline):
    """
    Track how communities evolve over time.
    
    Returns:
        List of evolution records
    """
    from ai_sn_analysis_prototype import detect_communities_louvain
    
    evolution = []
    sorted_periods = sorted(graphs_timeline.items(), key=lambda x: x[1]['start'])
    
    for i in range(len(sorted_periods) - 1):
        period1, data1 = sorted_periods[i]
        period2, data2 = sorted_periods[i + 1]
        
        G1 = data1['graph']
        G2 = data2['graph']
        
        # Detect communities
        detect_communities_louvain(G1)
        detect_communities_louvain(G2)
        
        comm1 = nx.get_node_attributes(G1, 'community_greedy')
        comm2 = nx.get_node_attributes(G2, 'community_greedy')
        
        # Find common users
        common_users = set(G1.nodes()) & set(G2.nodes())
        
        # Calculate stability
        if common_users:
            same_comm = sum(1 for u in common_users if comm1.get(u) == comm2.get(u))
            stability = same_comm / len(common_users)
        else:
            stability = 0.0
        
        evolution.append({
            'period_from': period1,
            'period_to': period2,
            'stability': float(stability),
            'new_users': len(set(G2.nodes()) - set(G1.nodes())),
            'left_users': len(set(G1.nodes()) - set(G2.nodes())),
            'common_users': len(common_users),
            'communities_before': len(set(comm1.values())) if comm1 else 0,
            'communities_after': len(set(comm2.values())) if comm2 else 0
        })
    
    logger.info(f"Computed evolution for {len(evolution)} period transitions")
    return evolution

# ===== MULTI-LAYER NETWORKS =====

def build_multilayer_network(posts):
    """
    Build multiple relationship layers.
    
    Returns:
        Tuple of (layers_dict, layer_metrics)
    """
    layers = {
        'reply': nx.DiGraph(),
        'mention': nx.DiGraph(),
        'topic': nx.Graph()
    }
    
    # Build reply layer
    for post in posts:
        author = post.get('author')
        if not author or author == '[deleted]':
            continue
        
        for comment in post.get('comments', []):
            commenter = comment.get('author')
            if commenter and commenter != '[deleted]':
                layers['reply'].add_edge(commenter, author, interaction='reply')
    
    # Build mention layer
    mention_pattern = r'u/(\w+)'
    
    for post in posts:
        text = f"{post.get('title', '')} {post.get('selftext', '')}"
        mentions = re.findall(mention_pattern, text)
        author = post.get('author')
        
        if author and author != '[deleted]':
            for mentioned in mentions:
                if mentioned != author:
                    layers['mention'].add_edge(author, mentioned, interaction='mention')
        
        # Check comments for mentions
        for comment in post.get('comments', []):
            comm_text = comment.get('body', '')
            comm_mentions = re.findall(mention_pattern, comm_text)
            commenter = comment.get('author')
            
            if commenter and commenter != '[deleted]':
                for mentioned in comm_mentions:
                    if mentioned != commenter:
                        layers['mention'].add_edge(commenter, mentioned, interaction='mention')
    
    # Compute metrics per layer
    layer_metrics = {}
    for layer_name, G in layers.items():
        if len(G) > 0:
            layer_metrics[layer_name] = {
                'nodes': len(G.nodes()),
                'edges': len(G.edges()),
                'density': float(nx.density(G))
            }
            
            # Top users per layer
            if len(G) > 0:
                pr = nx.pagerank(G)
                top_users = sorted(pr.items(), key=lambda x: x[1], reverse=True)[:5]
                layer_metrics[layer_name]['top_users'] = [
                    {'user': u, 'score': float(s)} for u, s in top_users
                ]
    
    logger.info(f"Built {len(layers)} network layers")
    return layers, layer_metrics

# ===== PREDICTIVE ML MODEL =====

def train_viral_predictor(posts, content_analysis, G):
    """
    Train a RandomForest model to predict viral content.
    
    Returns:
        Dictionary with model and performance metrics
    """
    features = []
    labels = []
    
    pagerank_scores = nx.get_node_attributes(G, 'pagerank')
    degree_scores = dict(G.degree())
    
    # Calculate score threshold for "viral" (top 20%)
    all_scores = [p.get('score', 0) for p in posts]
    viral_threshold = np.percentile(all_scores, 80) if all_scores else 10
    
    for post in posts:
        author = post.get('author')
        post_id = post.get('id')
        content = content_analysis.get(post_id, {})
        
        # Feature engineering
        feature_vec = [
            content.get('score', 0.5),                      # Sentiment score
            pagerank_scores.get(author, 0),                 # Author influence
            degree_scores.get(author, 0),                   # Author connections
            len(post.get('title', '')),                     # Title length
            len(post.get('selftext', '')),                  # Body length
            len(content.get('topics', [])),                 # Number of topics
            post.get('created_utc', 0) % 86400 / 3600,     # Hour of day
            1 if post.get('selftext') else 0,               # Has body text
            len(post.get('comments', []))                   # Number of comments
        ]
        
        # Label: viral if score exceeds threshold
        is_viral = 1 if post.get('score', 0) >= viral_threshold else 0
        
        features.append(feature_vec)
        labels.append(is_viral)
    
    if len(features) < 10:
        logger.warning("Insufficient data for training viral predictor")
        return None
    
    # Train model
    X = np.array(features)
    y = np.array(labels)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
    model.fit(X_train, y_train)
    
    # Evaluate
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions, output_dict=True, zero_division=0)
    
    # Feature importance
    feature_names = [
        'Sentiment', 'Author_PageRank', 'Author_Degree', 'Title_Length', 
        'Body_Length', 'Topic_Count', 'Hour_of_Day', 'Has_Body', 'Comment_Count'
    ]
    importance = {name: float(score) for name, score in zip(feature_names, model.feature_importances_)}
    
    result = {
        'model': model,
        'accuracy': float(accuracy),
        'precision': float(report.get('1', {}).get('precision', 0)),
        'recall': float(report.get('1', {}).get('recall', 0)),
        'f1_score': float(report.get('1', {}).get('f1-score', 0)),
        'feature_importance': importance,
        'viral_threshold': float(viral_threshold),
        'training_samples': len(X_train),
        'test_samples': len(X_test)
    }
    
    logger.info(f"Trained viral predictor: accuracy={accuracy:.3f}, precision={result['precision']:.3f}")
    
    return result

# Import GraphBuilder for temporal analysis
try:
    from ai_sn_analysis_prototype import GraphBuilder
except ImportError:
    logger.warning("Could not import GraphBuilder - some features may not work")

"""
Utility functions for error handling, logging, and validation.
"""
import logging
import os
from logging.handlers import RotatingFileHandler
from datetime import datetime, timezone

# Custom Exception Classes
class AnalysisError(Exception):
    """Base exception for analysis errors."""
    pass

class DataFetchError(AnalysisError):
    """Error fetching data from Reddit."""
    pass

class GraphBuildError(AnalysisError):
    """Error building network graph."""
    pass

class ValidationError(AnalysisError):
    """Error during validation."""
    pass

# Logging Setup
def setup_logging(log_file='logs/analysis.log', log_level=logging.INFO):
    """
    Setup comprehensive logging with file rotation.
    
    Args:
        log_file: Path to log file
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    
    Returns:
        Logger instance
    """
    # Create logs directory if it doesn't exist
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    logger = logging.getLogger('SocialNetworkAnalyzer')
    logger.setLevel(log_level)
    
    # Remove existing handlers
    logger.handlers = []
    
    # File handler with rotation (10MB max, keep 5 backups)
    file_handler = RotatingFileHandler(
        log_file, 
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5,
        encoding='utf-8'
    )
    file_handler.setLevel(logging.DEBUG)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Formatter with timestamp, level, file, line, and message
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

# Safe execution wrapper
def safe_execute(func, *args, logger=None, error_msg="Operation failed", **kwargs):
    """
    Safely execute a function with error handling and logging.
    
    Args:
        func: Function to execute
        *args: Positional arguments for func
        logger: Logger instance (optional)
        error_msg: Error message prefix
        **kwargs: Keyword arguments for func
    
    Returns:
        Function result or None on error
    """
    if logger is None:
        logger = logging.getLogger('SocialNetworkAnalyzer')
    
    try:
        result = func(*args, **kwargs)
        return result
    except DataFetchError as e:
        logger.error(f"{error_msg} - Data fetch error: {e}")
        raise
    except GraphBuildError as e:
        logger.error(f"{error_msg} - Graph build error: {e}")
        raise
    except ValidationError as e:
        logger.error(f"{error_msg} - Validation error: {e}")
        raise
    except Exception as e:
        logger.exception(f"{error_msg} - Unexpected error: {e}")
        raise AnalysisError(f"{error_msg}: {e}") from e

# Validation utilities
def validate_subreddit_name(subreddit):
    """Validate subreddit name format."""
    if not subreddit or not isinstance(subreddit, str):
        raise ValidationError("Subreddit name must be a non-empty string")
    
    # Remove r/ prefix if present
    subreddit = subreddit.replace('r/', '').strip()
    
    if not subreddit.isalnum() and '_' not in subreddit:
        raise ValidationError(f"Invalid subreddit name: {subreddit}")
    
    return subreddit

def validate_post_limit(limit):
    """Validate post limit."""
    if not isinstance(limit, int) or limit < 1 or limit > 1000:
        raise ValidationError(f"Post limit must be between 1 and 1000, got: {limit}")
    
    return limit

def validate_time_filter(time_filter):
    """Validate time filter."""
    valid_filters = ['all', 'year', 'month', 'week', 'day', 'hour']
    if time_filter not in valid_filters:
        raise ValidationError(f"Time filter must be one of {valid_filters}, got: {time_filter}")
    
    return time_filter

# Performance monitoring
class PerformanceTimer:
    """Context manager for timing code execution."""
    
    def __init__(self, name, logger=None):
        self.name = name
        self.logger = logger or logging.getLogger('SocialNetworkAnalyzer')
        self.start_time = None
    
    def __enter__(self):
        self.start_time = datetime.now()
        self.logger.info(f"Starting: {self.name}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = (datetime.now() - self.start_time).total_seconds()
        if exc_type is None:
            self.logger.info(f"Completed: {self.name} (took {elapsed:.2f}s)")
        else:
            self.logger.error(f"Failed: {self.name} (after {elapsed:.2f}s)")
        return False  # Don't suppress exceptions

# Data validation
def validate_graph_data(G, min_nodes=1, min_edges=0):
    """
    Validate graph structure.
    
    Args:
        G: NetworkX graph
        min_nodes: Minimum number of nodes required
        min_edges: Minimum number of edges required
    
    Raises:
        ValidationError: If graph doesn't meet requirements
    """
    if len(G.nodes()) < min_nodes:
        raise ValidationError(f"Graph has only {len(G.nodes())} nodes, minimum {min_nodes} required")
    
    if len(G.edges()) < min_edges:
        raise ValidationError(f"Graph has only {len(G.edges())} edges, minimum {min_edges} required")
    
    # Check for isolated nodes
    isolated = list(nx.isolates(G))
    if isolated:
        logger = logging.getLogger('SocialNetworkAnalyzer')
        logger.warning(f"Graph has {len(isolated)} isolated nodes")
    
    return True

def format_metrics_dict(metrics, precision=4):
    """Format metrics dictionary for display."""
    formatted = {}
    for key, value in metrics.items():
        if isinstance(value, float):
            formatted[key] = round(value, precision)
        elif isinstance(value, dict):
            formatted[key] = format_metrics_dict(value, precision)
        else:
            formatted[key] = value
    return formatted

# Import NetworkX (needed for validate_graph_data)
try:
    import networkx as nx
except ImportError:
    pass

"""
Performance optimization utilities.
Includes caching, parallel processing, and profiling tools.
"""
import os
import time
import pickle
import hashlib
import logging
from functools import wraps, lru_cache
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from typing import Callable, Any, Optional
import cProfile
import pstats
from io import StringIO

logger = logging.getLogger('SocialNetworkAnalyzer')

# ===== CACHING =====

class DiskCache:
    """Simple disk-based cache for expensive computations."""
    
    def __init__(self, cache_dir='cache'):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
    
    def _get_cache_path(self, key):
        """Get cache file path for a key."""
        key_hash = hashlib.md5(str(key).encode()).hexdigest()
        return os.path.join(self.cache_dir, f"{key_hash}.pkl")
    
    def get(self, key):
        """Retrieve item from cache."""
        cache_path = self._get_cache_path(key)
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                logger.warning(f"Cache read error: {e}")
        return None
    
    def set(self, key, value):
        """Store item in cache."""
        cache_path = self._get_cache_path(key)
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(value, f)
        except Exception as e:
            logger.warning(f"Cache write error: {e}")
    
    def clear(self):
        """Clear all cache files."""
        import shutil
        if os.path.exists(self.cache_dir):
            shutil.rmtree(self.cache_dir)
        os.makedirs(self.cache_dir, exist_ok=True)
        logger.info("Cache cleared")

# Global cache instance
disk_cache = DiskCache()

def cached(cache_ttl_seconds=3600):
    """
    Decorator for caching function results to disk.
    
    Args:
        cache_ttl_seconds: Time-to-live for cache entries in seconds
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Create cache key from function name and arguments
            cache_key = (func.__name__, args, tuple(sorted(kwargs.items())))
            
            # Try to get from cache
            cached_result = disk_cache.get(cache_key)
            if cached_result is not None:
                cached_value, timestamp = cached_result
                age = time.time() - timestamp
                if age < cache_ttl_seconds:
                    logger.debug(f"Cache hit for {func.__name__} (age: {age:.1f}s)")
                    return cached_value
            
            # Compute result
            result = func(*args, **kwargs)
            
            # Store in cache with timestamp
            disk_cache.set(cache_key, (result, time.time()))
            logger.debug(f"Cached result for {func.__name__}")
            
            return result
        
        return wrapper
    return decorator

# ===== PARALLEL PROCESSING =====

def parallel_map(func: Callable, items: list, max_workers: int = 4, use_processes: bool = False):
    """
    Apply function to items in parallel.
    
    Args:
        func: Function to apply to each item
        items: List of items to process
        max_workers: Maximum number of parallel workers
        use_processes: Use ProcessPoolExecutor instead of ThreadPoolExecutor
    
    Returns:
        List of results in same order as items
    """
    if not items:
        return []
    
    ExecutorClass = ProcessPoolExecutor if use_processes else ThreadPoolExecutor
    
    with ExecutorClass(max_workers=max_workers) as executor:
        results = list(executor.map(func, items))
    
    return results

def parallel_build_graph_from_posts(posts, max_workers=4):
    """
    Build graph from posts using parallel processing.
    
    Args:
        posts: List of post dictionaries
        max_workers: Number of parallel workers
    
    Returns:
        NetworkX graph
    """
    import networkx as nx
    
    def process_post(post):
        """Extract edges from a single post."""
        edges = []
        author = post.get('author')
        if not author or author == '[deleted]':
            return edges
        
        for comment in post.get('comments', []):
            commenter = comment.get('author')
            if commenter and commenter != '[deleted]':
                edges.append((commenter, author))
        
        return edges
    
    # Process posts in parallel
    all_edges_lists = parallel_map(process_post, posts, max_workers=max_workers)
    
    # Flatten edge lists
    all_edges = [edge for edges in all_edges_lists for edge in edges]
    
    # Build graph
    G = nx.DiGraph()
    for u, v in all_edges:
        if G.has_edge(u, v):
            G[u][v]['weight'] += 1
        else:
            G.add_edge(u, v, weight=1)
    
    logger.info(f"Built graph with {len(G.nodes())} nodes and {len(G.edges())} edges using {max_workers} workers")
    
    return G

# ===== PROFILING =====

def profile_function(func):
    """
    Decorator to profile a function's performance.
    
    Usage:
        @profile_function
        def my_function():
            # ...
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        profiler = cProfile.Profile()
        profiler.enable()
        
        try:
            result = func(*args, **kwargs)
        finally:
            profiler.disable()
            
            # Print stats
            s = StringIO()
            ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
            ps.print_stats(20)  # Top 20 slowest functions
            
            logger.info(f"\n{'='*60}\nProfile for {func.__name__}:\n{s.getvalue()}\n{'='*60}")
        
        return result
    
    return wrapper

class ProgressTracker:
    """Track progress of long-running operations."""
    
    def __init__(self, total_steps, description="Processing"):
        self.total_steps = total_steps
        self.current_step = 0
        self.description = description
        self.start_time = time.time()
    
    def update(self, step=None, message=None):
        """Update progress."""
        if step is not None:
            self.current_step = step
        else:
            self.current_step += 1
        
        elapsed = time.time() - self.start_time
        progress_pct = (self.current_step / self.total_steps) * 100
        
        if self.current_step > 0:
            eta = (elapsed / self.current_step) * (self.total_steps - self.current_step)
        else:
            eta = 0
        
        msg = f"{self.description}: {self.current_step}/{self.total_steps} ({progress_pct:.1f}%) - ETA: {eta:.1f}s"
        if message:
            msg += f" - {message}"
        
        logger.info(msg)
        
        return {
            'step': self.current_step,
            'total': self.total_steps,
            'progress': progress_pct,
            'elapsed': elapsed,
            'eta': eta
        }

# ===== BATCH PROCESSING =====

def batch_process(items, batch_size=100, process_func=None):
    """
    Process items in batches.
    
    Args:
        items: List of items to process
        batch_size: Number of items per batch
        process_func: Function to apply to each batch
    
    Yields:
        Results from each batch
    """
    for i in range(0, len(items), batch_size):
        batch = items[i:i+batch_size]
        if process_func:
            yield process_func(batch)
        else:
            yield batch

# ===== MEMORY OPTIMIZATION =====

class MemoryMonitor:
    """Monitor memory usage of operations."""
    
    def __init__(self, name="Operation"):
        self.name = name
        self.start_memory = 0
        
        try:
            import psutil
            self.process = psutil.Process(os.getpid())
            self.available = True
        except ImportError:
            self.available = False
            logger.warning("psutil not installed - memory monitoring unavailable")
    
    def __enter__(self):
        if self.available:
            self.start_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.available:
            end_memory = self.process.memory_info().rss / 1024 / 1024  # MB
            memory_used = end_memory - self.start_memory
            logger.info(f"{self.name}: Memory used = {memory_used:.2f} MB (Total: {end_memory:.2f} MB)")
        return False

# ===== LAZY LOADING =====

class LazyLoader:
    """Lazy load large data structures."""
    
    def __init__(self, load_func, *args, **kwargs):
        self.load_func = load_func
        self.args = args
        self.kwargs = kwargs
        self._data = None
        self._loaded = False
    
    def load(self):
        """Force loading of data."""
        if not self._loaded:
            self._data = self.load_func(*self.args, **self.kwargs)
            self._loaded = True
        return self._data
    
    def __getattr__(self, name):
        """Auto-load on attribute access."""
        if name.startswith('_'):
            return object.__getattribute__(self, name)
        return getattr(self.load(), name)
    
    def __getitem__(self, key):
        """Auto-load on item access."""
        return self.load()[key]

# Performance optimization recommendations
def analyze_performance_bottlenecks(stats_dict):
    """
    Analyze performance statistics and provide recommendations.
    
    Args:
        stats_dict: Dictionary with timing information
    
    Returns:
        List of recommendations
    """
    recommendations = []
    
    # Check for slow operations
    for operation, duration in stats_dict.items():
        if duration > 60:  # More than 1 minute
            recommendations.append({
                'operation': operation,
                'issue': 'Slow operation',
                'duration': duration,
                'suggestion': 'Consider caching or parallel processing'
            })
        elif duration > 30:
            recommendations.append({
                'operation': operation,
                'issue': 'Moderately slow',
                'duration': duration,
                'suggestion': 'May benefit from optimization'
            })
    
    return recommendations

"""
Simple tests to verify the application components work correctly
Run this after setup to verify everything is installed properly
"""

import sys
import os

def test_imports():
    """Test that all required packages are importable"""
    print("Testing imports...")
    try:
        import praw
        import networkx as nx
        import pandas as pd
        import numpy as np
        import plotly.graph_objects as go
        import matplotlib.pyplot as plt
        from tqdm import tqdm
        print("✅ All imports successful")
        return True
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        print("Run: pip install -r requirements.txt")
        return False

def test_environment():
    """Test that environment variables are set"""
    print("\nTesting environment variables...")
    required = ['REDDIT_CLIENT_ID', 'REDDIT_CLIENT_SECRET']
    missing = []
    
    for var in required:
        if not os.environ.get(var):
            missing.append(var)
    
    if missing:
        print(f"⚠️  Missing environment variables: {', '.join(missing)}")
        print("   Edit your .env file and add these credentials")
        print("   Get them from: https://www.reddit.com/prefs/apps")
        return False
    else:
        print("✅ Environment variables set")
        return True

def test_basic_functionality():
    """Test basic app components"""
    print("\nTesting basic functionality...")
    try:
        from ai_sn_analysis_prototype import (
            GeminiClient, UnionFind, GraphBuilder,
            simple_local_text_analysis
        )
        
        # Test text analysis
        result = simple_local_text_analysis("This is a great test!")
        assert 'sentiment' in result
        assert 'topics' in result
        print("  ✅ Text analysis works")
        
        # Test Union-Find
        uf = UnionFind()
        uf.union('a', 'b')
        uf.union('b', 'c')
        assert uf.find('a') == uf.find('c')
        print("  ✅ Union-Find works")
        
        # Test GraphBuilder
        import networkx as nx
        gb = GraphBuilder(directed=True)
        assert isinstance(gb.G, nx.DiGraph)
        print("  ✅ GraphBuilder works")
        
        print("✅ Basic functionality tests passed")
        return True
        
    except Exception as e:
        print(f"❌ Functionality test failed: {e}")
        return False

def test_output_directory():
    """Test that output directory exists"""
    print("\nTesting output directory...")
    if os.path.exists('output'):
        print("✅ Output directory exists")
        return True
    else:
        print("⚠️  Output directory missing (will be created on first run)")
        return True

def main():
    print("=" * 60)
    print("  Running Application Tests")
    print("=" * 60)
    
    # Load .env if available
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    
    results = []
    results.append(test_imports())
    results.append(test_environment())
    results.append(test_basic_functionality())
    results.append(test_output_directory())
    
    print("\n" + "=" * 60)
    if all(results):
        print("  🎉 All Tests Passed!")
        print("=" * 60)
        print("\nYou're ready to run the analysis:")
        print("  python ai_sn_analysis_prototype.py --subreddit python --posts 50")
    else:
        print("  ⚠️  Some Tests Failed")
        print("=" * 60)
        print("\nPlease fix the issues above before running the analysis.")
    
    return all(results)

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)

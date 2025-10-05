"""
Example usage of the AI Social Network Analysis tool
This script demonstrates how to use the analysis pipeline programmatically
"""

import os
from ai_sn_analysis_prototype import run_pipeline

def main():
    # Example 1: Basic analysis of r/python
    print("Example 1: Analyzing r/python with 50 posts")
    print("-" * 60)
    
    # Make sure you have set REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET in .env
    if not os.environ.get('REDDIT_CLIENT_ID'):
        print("ERROR: Please set REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET in .env file")
        print("See QUICKSTART.md for instructions")
        return
    
    try:
        results = run_pipeline(
            subreddit='python',
            posts_limit=50,
            outdir='output/example1',
            time_filter='week'
        )
        
        print("\n✅ Analysis complete!")
        print(f"   - Graph has {results['graph'].number_of_nodes()} nodes")
        print(f"   - Graph has {results['graph'].number_of_edges()} edges")
        print(f"   - Detected {len(results['trends'])} trending topics")
        print(f"   - Results saved to: output/example1/")
        print(f"   - Open output/example1/python_graph.html to view the network!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Make sure your Reddit API credentials are set correctly in .env")
        return
    
    # Example 2: Accessing graph data
    print("\n" + "=" * 60)
    print("Example 2: Working with the graph data")
    print("-" * 60)
    
    G = results['graph']
    
    # Find top 5 most influential users by PageRank
    pageranks = [(node, data.get('pagerank', 0)) for node, data in G.nodes(data=True)]
    pageranks.sort(key=lambda x: x[1], reverse=True)
    
    print("\nTop 5 most influential users (by PageRank):")
    for i, (user, pr) in enumerate(pageranks[:5], 1):
        print(f"  {i}. {user}: {pr:.6f}")
    
    # Find trending topics
    print("\nTop trending topics:")
    for i, (topic, data) in enumerate(results['trends'][:5], 1):
        print(f"  {i}. {topic}: velocity={data['velocity']:.2f}, recent={data['recent']}")
    
    print("\n" + "=" * 60)
    print("Examples complete! Check the output/ folder for all results.")
    print("=" * 60)

if __name__ == '__main__':
    main()

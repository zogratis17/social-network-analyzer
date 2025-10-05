"""
Quick Demo Script - AI Social Network Analyzer
Demonstrates the complete workflow from analysis to visualization
"""

import os
import sys
import time
import subprocess

def print_header(text):
    """Print a formatted header"""
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70 + "\n")

def print_step(number, text):
    """Print a formatted step"""
    print(f"\n{'='*70}")
    print(f"  STEP {number}: {text}")
    print(f"{'='*70}\n")

def run_command(cmd, description):
    """Run a command and display output"""
    print(f"🔄 {description}...")
    print(f"💻 Command: {cmd}\n")
    
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    if result.returncode == 0:
        print("✅ Success!\n")
        if result.stdout:
            print(result.stdout[:500])  # Print first 500 chars
        return True
    else:
        print("❌ Error!\n")
        if result.stderr:
            print(result.stderr[:500])
        return False

def main():
    print_header("🕸️  AI SOCIAL NETWORK ANALYZER - QUICK DEMO")
    
    print("""
    This demo will:
    1. ✅ Check prerequisites
    2. 📊 Analyze r/python (50 posts for speed)
    3. 🔥 Display trending topics
    4. 🚀 Launch the interactive dashboard
    
    Total time: ~30 seconds
    """)
    
    input("\n📝 Press ENTER to start the demo...")
    
    # Step 1: Check prerequisites
    print_step(1, "Checking Prerequisites")
    
    print("Checking Python packages...")
    required = ['praw', 'networkx', 'pandas', 'plotly', 'streamlit']
    
    missing = []
    for package in required:
        try:
            __import__(package)
            print(f"  ✅ {package}")
        except ImportError:
            print(f"  ❌ {package} - MISSING")
            missing.append(package)
    
    if missing:
        print(f"\n⚠️  Missing packages: {', '.join(missing)}")
        response = input("Install now? (y/n): ")
        if response.lower() == 'y':
            run_command(
                f"{sys.executable} -m pip install {' '.join(missing)}",
                "Installing packages"
            )
        else:
            print("❌ Cannot continue without required packages.")
            return
    
    # Check .env
    if not os.path.exists('.env'):
        print("\n⚠️  .env file not found!")
        print("Please create .env with your Reddit API credentials.")
        print("\nExample:")
        print("  REDDIT_CLIENT_ID=your_id")
        print("  REDDIT_CLIENT_SECRET=your_secret")
        print("  REDDIT_USER_AGENT=demo/1.0")
        return
    else:
        print("\n✅ .env file found")
    
    input("\n📝 Press ENTER to continue to analysis...")
    
    # Step 2: Run analysis
    print_step(2, "Running Social Network Analysis")
    
    print("Analyzing r/python with 50 posts...")
    print("This will take about 10-15 seconds...\n")
    
    cmd = f"{sys.executable} ai_sn_analysis_prototype.py --subreddit python --posts 50"
    
    start_time = time.time()
    success = run_command(cmd, "Running analysis")
    elapsed = time.time() - start_time
    
    if not success:
        print("❌ Analysis failed. Check your .env credentials.")
        return
    
    print(f"\n⏱️  Analysis completed in {elapsed:.1f} seconds!")
    
    # Check output files
    output_files = [
        'python_graph.json',
        'python_nodes.csv',
        'python_edges.csv',
        'python_trends.json',
        'python_graph.html'
    ]
    
    print("\n📁 Generated files:")
    for fname in output_files:
        path = os.path.join('output', fname)
        if os.path.exists(path):
            size = os.path.getsize(path) / 1024
            print(f"  ✅ {fname} ({size:.1f} KB)")
        else:
            print(f"  ❌ {fname} - NOT FOUND")
    
    input("\n📝 Press ENTER to view trending topics...")
    
    # Step 3: View trends
    print_step(3, "Displaying Trending Topics")
    
    run_command(
        f"{sys.executable} view_trends.py",
        "Loading trending topics"
    )
    
    input("\n📝 Press ENTER to launch the dashboard...")
    
    # Step 4: Launch dashboard
    print_step(4, "Launching Interactive Dashboard")
    
    print("""
    🚀 Starting Streamlit dashboard...
    
    The dashboard will open in your browser at:
    📊 http://localhost:8501
    
    Features to explore:
    ✅ Overview - Network statistics and metrics
    ✅ Communities - Detected groups and structures
    ✅ Influencers - Top users by PageRank
    ✅ Trending Topics - What's hot in the community
    ✅ Network Graph - Interactive visualization
    ✅ AI Insights - Sentiment and content analysis
    ✅ Analytics - Export and recommendations
    
    🛑 Press Ctrl+C in the terminal to stop the dashboard
    """)
    
    input("\n📝 Press ENTER to launch (this will keep running)...")
    
    try:
        subprocess.run([
            sys.executable,
            "-m",
            "streamlit",
            "run",
            "dashboard.py",
            "--server.headless=true"
        ])
    except KeyboardInterrupt:
        print("\n\n👋 Dashboard stopped!")
    
    print_header("🎉 DEMO COMPLETE!")
    
    print("""
    You've just seen the complete workflow:
    
    ✅ Data Collection from Reddit
    ✅ Network Graph Construction
    ✅ Community Detection (Union-Find + Greedy)
    ✅ Influence Analysis (PageRank)
    ✅ Trend Detection (AI + TF-IDF)
    ✅ Interactive Visualization
    
    Next steps:
    
    1. Try analyzing different subreddits:
       python ai_sn_analysis_prototype.py --subreddit machinelearning --posts 100
    
    2. Explore the dashboard tabs:
       python launch_dashboard.py
    
    3. Read the documentation:
       - README.md - Main guide
       - DASHBOARD_GUIDE.md - Dashboard features
       - PROJECT_COMPLETE.md - Full project overview
    
    Happy analyzing! 🕸️
    """)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Demo cancelled by user.")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

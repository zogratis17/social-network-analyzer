#!/usr/bin/env python
"""
Launch script for AI Social Network Analyzer Dashboard
Checks prerequisites and starts Streamlit
"""

import subprocess
import sys
import os

def check_requirements():
    """Check if all required packages are installed"""
    print("🔍 Checking prerequisites...")
    
    required_packages = {
        'streamlit': 'Streamlit',
        'pandas': 'Pandas',
        'plotly': 'Plotly',
        'networkx': 'NetworkX'
    }
    
    missing = []
    for package, name in required_packages.items():
        try:
            __import__(package)
            print(f"  ✅ {name}")
        except ImportError:
            print(f"  ❌ {name} not found")
            missing.append(package)
    
    if missing:
        print(f"\n⚠️  Missing packages: {', '.join(missing)}")
        print("📦 Installing missing packages...")
        subprocess.run([sys.executable, "-m", "pip", "install"] + missing)
        print("\n✅ Installation complete!")
    
    return True

def check_data():
    """Check if analysis data exists"""
    output_dir = "output"
    
    if not os.path.exists(output_dir):
        print(f"\n⚠️  Output directory '{output_dir}' not found!")
        print("\n💡 To generate data, run:")
        print("   python ai_sn_analysis_prototype.py --subreddit python --posts 100")
        return False
    
    files = os.listdir(output_dir)
    if not files:
        print(f"\n⚠️  No analysis files found in '{output_dir}'!")
        print("\n💡 To generate data, run:")
        print("   python ai_sn_analysis_prototype.py --subreddit python --posts 100")
        return False
    
    # Count unique analyses
    subreddits = set([f.split('_')[0] for f in files if '_' in f])
    
    if subreddits:
        print(f"\n✅ Found {len(subreddits)} analysis: {', '.join(subreddits)}")
        return True
    else:
        print(f"\n⚠️  No valid analysis files found!")
        return False

def main():
    print("=" * 60)
    print("🕸️  AI SOCIAL NETWORK ANALYZER DASHBOARD")
    print("=" * 60)
    print()
    
    # Check requirements
    if not check_requirements():
        print("\n❌ Prerequisites check failed!")
        sys.exit(1)
    
    # Check data
    has_data = check_data()
    
    if not has_data:
        response = input("\n❓ Launch dashboard anyway? (y/n): ")
        if response.lower() != 'y':
            print("\n👋 Run an analysis first, then launch the dashboard!")
            sys.exit(0)
    
    print("\n" + "=" * 60)
    print("🚀 LAUNCHING DASHBOARD...")
    print("=" * 60)
    print("\n📊 Dashboard will open in your browser at: http://localhost:8501")
    print("🛑 Press Ctrl+C to stop the dashboard")
    print()
    
    # Launch Streamlit
    try:
        subprocess.run([
            sys.executable, 
            "-m", 
            "streamlit", 
            "run", 
            "dashboard.py",
            "--server.headless=true",
            "--browser.gatherUsageStats=false"
        ])
    except KeyboardInterrupt:
        print("\n\n👋 Dashboard stopped!")
        sys.exit(0)

if __name__ == "__main__":
    main()

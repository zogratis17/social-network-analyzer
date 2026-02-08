"""
Quick setup script for AI Social Network Analysis
This script will help you set up the environment and configuration
"""

import os
import sys
import subprocess

def check_python_version():
    """Check if Python version is 3.8+"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required.")
        print(f"   Current version: {sys.version}")
        return False
    print(f"✅ Python version: {sys.version.split()[0]}")
    return True

def install_dependencies():
    """Install required packages"""
    print("\n📦 Installing dependencies...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError:
        print("❌ Failed to install dependencies")
        return False

def create_env_file():
    """Create .env file from template if it doesn't exist"""
    if os.path.exists('.env'):
        print("\n✅ .env file already exists")
        return True
    
    if not os.path.exists('.env.example'):
        print("\n❌ .env.example template not found")
        return False
    
    print("\n📝 Creating .env file from template...")
    with open('.env.example', 'r') as src:
        content = src.read()
    
    with open('.env', 'w') as dst:
        dst.write(content)
    
    print("✅ .env file created")
    print("\n⚠️  IMPORTANT: Edit .env and add your Reddit API credentials!")
    print("   Get credentials from: https://www.reddit.com/prefs/apps")
    return True

def create_output_dir():
    """Create output directory"""
    if not os.path.exists('output'):
        os.makedirs('output')
        print("\n✅ Created output directory")
    return True

def main():
    print("=" * 60)
    print("  AI Social Network Analysis - Setup")
    print("=" * 60)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Install dependencies
    if not install_dependencies():
        sys.exit(1)
    
    # Create .env file
    if not create_env_file():
        sys.exit(1)
    
    # Create output directory
    create_output_dir()
    
    print("\n" + "=" * 60)
    print("  Setup Complete! 🎉")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Edit .env file with your Reddit API credentials")
    print("2. Run the analysis:")
    print("   python ai_sn_analysis_prototype.py --subreddit python --posts 100")
    print("\nFor more information, see README.md")

if __name__ == "__main__":
    main()

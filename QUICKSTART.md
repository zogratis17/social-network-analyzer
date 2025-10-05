# Quick Start Guide

## 🚀 Getting Started in 3 Steps

### Step 1: Install Dependencies

Run the setup script to automatically install all required packages:

```bash
python setup.py
```

Or install manually:

```bash
pip install -r requirements.txt
```

### Step 2: Configure Reddit API

1. Go to https://www.reddit.com/prefs/apps
2. Click "Create App" or "Create Another App"
3. Fill in the form:
   - **name**: My Social Network Analyzer (or any name)
   - **App type**: Select "script"
   - **description**: (optional)
   - **about url**: (leave blank)
   - **redirect uri**: http://localhost:8080 (required but not used)
4. Click "Create app"
5. Copy your credentials:
   - **client_id**: The string under "personal use script"
   - **client_secret**: The "secret" field

6. Edit your `.env` file and add:
   ```
   REDDIT_CLIENT_ID=your_client_id_here
   REDDIT_CLIENT_SECRET=your_secret_here
   REDDIT_USER_AGENT=ai-sn-analysis/0.1
   ```

### Step 3: Run Your First Analysis

```bash
python ai_sn_analysis_prototype.py --subreddit python --posts 100
```

This will:
- Fetch 100 posts from r/python
- Build a network graph of user interactions
- Detect communities
- Calculate influence metrics
- Generate visualizations in the `output/` folder

## 📊 View Results

Open `output/python_graph.html` in your web browser to see the interactive network visualization!

## 🔧 Common Issues

**Issue**: "REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET must be set"
- **Solution**: Make sure your `.env` file exists and contains valid credentials

**Issue**: Import errors when running the script
- **Solution**: Run `pip install -r requirements.txt` to install dependencies

**Issue**: "429 Too Many Requests" from Reddit
- **Solution**: The app already includes rate limiting. Wait a few minutes and try again.

## 💡 Tips

- Start with smaller datasets (50-200 posts) for faster testing
- Use `--time-filter week` to analyze recent activity
- The interactive HTML graph is zoomable and draggable
- Check the CSV files for detailed node and edge data

## 📖 More Information

See `README.md` for complete documentation and advanced usage.

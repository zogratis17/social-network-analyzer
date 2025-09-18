# src/data_collection.py
import praw
import json
from config import REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, REDDIT_USER_AGENT, DATA_PATH_RAW, MAX_POSTS_FETCH
import os

def fetch_reddit_posts(subreddit="technology", limit=MAX_POSTS_FETCH):
    reddit = praw.Reddit(
        client_id=REDDIT_CLIENT_ID,
        client_secret=REDDIT_CLIENT_SECRET,
        user_agent=REDDIT_USER_AGENT
    )
    posts = []
    for submission in reddit.subreddit(subreddit).hot(limit=limit):
        posts.append({
            "id": submission.id,
            "title": submission.title,
            "author": str(submission.author),
            "score": submission.score,
            "num_comments": submission.num_comments,
            "created_utc": submission.created_utc
        })
    return posts

if __name__ == "__main__":
    os.makedirs(DATA_PATH_RAW, exist_ok=True)
    data = fetch_reddit_posts()
    file_path = os.path.join(DATA_PATH_RAW, "technology_raw.json")
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"Saved {len(data)} posts from r/technology to {file_path}")

# src/ai_analysis.py

def analyze_content_with_gemini(posts):
    # Mock analysis
    results = [{"post": post, "trend_score": 0.8} for post in posts]
    return results

def detect_trending_content(posts, threshold=0.7):
    analyzed = analyze_content_with_gemini(posts)
    trending = [item for item in analyzed if item["trend_score"] >= threshold]
    return trending

if __name__ == "__main__":
    sample_posts = [
        "AI is transforming healthcare!",
        "Random meme post",
        "New AI tools released today"
    ]
    print("Analyzed Content:", analyze_content_with_gemini(sample_posts))
    print("Trending Posts:", detect_trending_content(sample_posts))

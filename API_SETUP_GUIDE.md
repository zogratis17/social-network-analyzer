# API Setup Guide

This guide will help you get the required API keys for the Social Network Analyzer.

---

## 🔑 Required: Reddit API Credentials

### Step-by-Step Instructions:

1. **Go to Reddit Apps Page**
   - Visit: https://www.reddit.com/prefs/apps
   - Log in to your Reddit account if not already logged in

2. **Create a New App**
   - Scroll to the bottom and click **"create another app..."** or **"are you a developer? create an app..."**

3. **Fill in the Application Form**
   ```
   Name:           My Social Network Analyzer (or any name you want)
   App type:       ⦿ script  (Select this radio button)
   Description:    (optional - leave blank or add description)
   About url:      (optional - leave blank)
   Redirect uri:   http://localhost:8080  (required, but not actually used)
   ```

4. **Create the App**
   - Click **"create app"** button

5. **Copy Your Credentials**
   After creation, you'll see your app details:
   
   ```
   [App Icon]  My Social Network Analyzer
   personal use script
   [Random string here] ← This is your CLIENT_ID
   
   secret: [Another random string] ← This is your CLIENT_SECRET
   ```

6. **Update Your `.env` File**
   Open `.env` and replace the placeholders:
   ```bash
   REDDIT_CLIENT_ID=the_random_string_under_personal_use_script
   REDDIT_CLIENT_SECRET=the_string_after_secret
   REDDIT_USER_AGENT=ai-sn-analysis/0.1
   ```

### Example:
If you see:
```
personal use script
abc123xyz789

secret: def456uvw012
```

Your `.env` should have:
```bash
REDDIT_CLIENT_ID=abc123xyz789
REDDIT_CLIENT_SECRET=def456uvw012
REDDIT_USER_AGENT=ai-sn-analysis/0.1
```

---

## ✨ Optional: Google Gemini API (for Advanced AI Analysis)

**Note**: The app works perfectly fine without Gemini! It will use a local fallback analyzer. Gemini just provides more sophisticated sentiment analysis and topic extraction.

### Step-by-Step Instructions:

1. **Get a Gemini API Key**
   - Visit: https://makersuite.google.com/app/apikey
   - Or: https://aistudio.google.com/apikey
   - Sign in with your Google account

2. **Create an API Key**
   - Click **"Create API Key"**
   - Select a Google Cloud project or create a new one
   - Copy the generated API key

3. **Update Your `.env` File**
   ```bash
   GEMINI_API_KEY=your_actual_gemini_api_key_here
   ```

### Important Notes:
- Gemini has a **free tier** with generous limits for testing
- The free tier includes:
  - 15 requests per minute
  - 1 million tokens per day
  - 1500 requests per day
- Perfect for analyzing hundreds of Reddit posts per day

### Pricing (if you exceed free tier):
- Very affordable for occasional use
- Check current pricing: https://ai.google.dev/pricing

---

## 🧪 Testing Your Configuration

After setting up your API keys, run the test script:

```bash
python test_setup.py
```

This will verify:
- ✅ All packages are installed
- ✅ Environment variables are set
- ✅ Basic functionality works
- ✅ Gemini API is accessible (if configured)

---

## 🚀 Running Your First Analysis

Once your `.env` is configured:

### Without Gemini (Local Analysis):
```bash
python ai_sn_analysis_prototype.py --subreddit python --posts 50
```

### With Gemini (Advanced AI Analysis):
```bash
python ai_sn_analysis_prototype.py --subreddit python --posts 50 --gemini-key $GEMINI_API_KEY
```

Or if you set `GEMINI_API_KEY` in `.env`, just run:
```bash
python ai_sn_analysis_prototype.py --subreddit python --posts 50
```

---

## 🔒 Security Best Practices

1. **Never commit your `.env` file to Git**
   - It's already in `.gitignore` ✅
   - Only commit `.env.example` as a template

2. **Keep your API keys private**
   - Don't share them in screenshots
   - Don't paste them in public forums

3. **Regenerate if compromised**
   - Reddit: Delete the app and create a new one
   - Gemini: Revoke the old key and create a new one

---

## ❓ Troubleshooting

### "REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET must be set"
- Make sure `.env` file exists in the project root
- Check that you've replaced the placeholder values
- Make sure there are no extra spaces or quotes around the values

### "Failed to initialize Gemini API"
- Verify your API key is correct
- Check your internet connection
- Ensure you haven't exceeded free tier limits
- Try again in a few minutes (rate limiting)

### "Import google.generativeai could not be resolved"
- Run: `pip install google-generativeai`
- Or run: `pip install -r requirements.txt`

---

## 📊 What Gets Analyzed?

### With Local Analyzer (No Gemini):
- ✅ Sentiment (positive/negative/neutral) - rule-based
- ✅ Simple topic extraction - word frequency
- ✅ Basic viral score - heuristic

### With Gemini API:
- ✅ Advanced sentiment analysis - AI-powered
- ✅ Contextual topic extraction - understands meaning
- ✅ Better viral prediction - ML-based
- ✅ More accurate results - trained on massive datasets

---

## 🎉 You're Ready!

Once you have your Reddit credentials (and optionally Gemini API key), you can analyze any subreddit and explore the social network dynamics!

See `QUICKSTART.md` for usage examples.

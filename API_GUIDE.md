# 🚀 API Quick Start Guide

## Running the FastAPI Server

### Start the API server:
```bash
python api.py
```

Or using uvicorn directly:
```bash
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at: `http://localhost:8000`

Interactive API documentation: `http://localhost:8000/docs`

---

## API Endpoints

### 1. **Start Analysis**
```http
POST /analyze
```

**Request Body:**
```json
{
  "subreddit": "python",
  "num_posts": 100,
  "time_filter": "week",
  "use_gemini": false,
  "append_mode": "auto",
  "enable_advanced": true
}
```

**Response:**
```json
{
  "status": "started",
  "task_id": "123e4567-e89b-12d3-a456-426614174000",
  "message": "Analysis of r/python started",
  "subreddit": "python"
}
```

**cURL Example:**
```bash
curl -X POST "http://localhost:8000/analyze" \
  -H "Content-Type: application/json" \
  -d '{
    "subreddit": "python",
    "num_posts": 100,
    "time_filter": "week"
  }'
```

---

### 2. **Check Task Status**
```http
GET /status/{task_id}
```

**Response:**
```json
{
  "task_id": "123e4567-e89b-12d3-a456-426614174000",
  "status": "completed",
  "subreddit": "python",
  "progress": 100,
  "message": null,
  "created_at": "2025-10-15T10:30:00",
  "completed_at": "2025-10-15T10:32:30",
  "error": null
}
```

---

### 3. **Get Results**
```http
GET /results/{subreddit}
```

**Response:**
```json
{
  "subreddit": "python",
  "graph": {
    "nodes": 4821,
    "edges": 6336
  },
  "trends": [...],
  "validation": {...},
  "sentiment_networks": {...},
  "evolution": [...],
  "predictor": {...}
}
```

---

### 4. **List Analyzed Subreddits**
```http
GET /list
```

**Response:**
```json
{
  "subreddits": [
    {
      "subreddit": "python",
      "nodes": 4821,
      "edges": 6336,
      "last_analyzed": "2025-10-15T10:32:30"
    }
  ]
}
```

---

### 5. **Download Files**
```http
GET /download/{subreddit}/{file_type}
```

File types: `nodes`, `edges`, `graph`, `graphml`, `html`, `trends`

**Example:**
```bash
curl "http://localhost:8000/download/python/nodes" -o python_nodes.csv
```

---

### 6. **Delete Analysis**
```http
DELETE /delete/{subreddit}
```

**Response:**
```json
{
  "message": "Deleted 10 files for r/python",
  "files": ["python_graph.json", "python_nodes.csv", ...]
}
```

---

### 7. **Health Check**
```http
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2025-10-15T10:30:00",
  "tasks_active": 2
}
```

---

## Python Client Example

```python
import requests
import time

# API base URL
BASE_URL = "http://localhost:8000"

# 1. Start analysis
response = requests.post(f"{BASE_URL}/analyze", json={
    "subreddit": "python",
    "num_posts": 100,
    "time_filter": "week",
    "enable_advanced": True
})

task_id = response.json()['task_id']
print(f"Task started: {task_id}")

# 2. Poll for completion
while True:
    status = requests.get(f"{BASE_URL}/status/{task_id}").json()
    print(f"Status: {status['status']} - Progress: {status['progress']}%")
    
    if status['status'] == 'completed':
        break
    elif status['status'] == 'failed':
        print(f"Error: {status['error']}")
        exit(1)
    
    time.sleep(5)  # Wait 5 seconds

# 3. Get results
results = requests.get(f"{BASE_URL}/results/python").json()
print(f"Graph: {results['graph']['nodes']} nodes, {results['graph']['edges']} edges")

# 4. Download CSV
response = requests.get(f"{BASE_URL}/download/python/nodes")
with open("python_nodes.csv", "wb") as f:
    f.write(response.content)

print("Done!")
```

---

## JavaScript Client Example

```javascript
const BASE_URL = "http://localhost:8000";

async function analyzeSubreddit(subreddit) {
    // Start analysis
    const startResponse = await fetch(`${BASE_URL}/analyze`, {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({
            subreddit: subreddit,
            num_posts: 100,
            time_filter: 'week'
        })
    });
    
    const {task_id} = await startResponse.json();
    console.log(`Task started: ${task_id}`);
    
    // Poll for completion
    while (true) {
        const statusResponse = await fetch(`${BASE_URL}/status/${task_id}`);
        const status = await statusResponse.json();
        
        console.log(`Status: ${status.status} - ${status.progress}%`);
        
        if (status.status === 'completed') {
            break;
        } else if (status.status === 'failed') {
            throw new Error(status.error);
        }
        
        await new Promise(resolve => setTimeout(resolve, 5000));
    }
    
    // Get results
    const resultsResponse = await fetch(`${BASE_URL}/results/${subreddit}`);
    const results = await resultsResponse.json();
    
    console.log(`Graph: ${results.graph.nodes} nodes, ${results.graph.edges} edges`);
    return results;
}

// Usage
analyzeSubreddit('python').then(results => {
    console.log('Analysis complete!', results);
});
```

---

## Advanced Features

### Enable/Disable Advanced Analytics

```json
{
  "enable_advanced": false  // Disables temporal, validation, sentiment networks
}
```

### Data Append Modes

- `"auto"`: Append if last analysis > 7 days old
- `"always"`: Always merge with existing data
- `"never"`: Always replace (fresh analysis)

```json
{
  "append_mode": "auto"
}
```

---

## Production Deployment

### Using systemd (Linux):

Create `/etc/systemd/system/sn-api.service`:
```ini
[Unit]
Description=Social Network Analysis API
After=network.target

[Service]
Type=simple
User=www-data
WorkingDirectory=/path/to/social-network-analyzer
Environment="REDDIT_CLIENT_ID=your_id"
Environment="REDDIT_CLIENT_SECRET=your_secret"
ExecStart=/usr/bin/python3 api.py
Restart=always

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl enable sn-api
sudo systemctl start sn-api
```

---

## Rate Limiting & Best Practices

1. **Reddit API**: Limited to ~60 requests/minute
2. **Gemini AI**: Limited by your API quota
3. **Concurrent Requests**: API handles background tasks
4. **Caching**: Results are cached in `output/` directory

**Recommendations:**
- Start with `num_posts=50` for testing
- Use `use_gemini=false` for faster analysis
- Poll status endpoint every 5-10 seconds (don't spam!)
- Download files only after task completion

---

## Monitoring & Logs

View API logs:
```bash
tail -f logs/analysis.log
```

Check active tasks:
```bash
curl http://localhost:8000/health
```

---

## Error Handling

**Common Errors:**

- `404`: Subreddit or task not found
- `400`: Invalid parameters
- `500`: Internal server error

**Example Error Response:**
```json
{
  "detail": "Invalid subreddit name"
}
```

**Retry Logic:**
```python
import time

def analyze_with_retry(subreddit, max_retries=3):
    for attempt in range(max_retries):
        try:
            response = requests.post(f"{BASE_URL}/analyze", json={
                "subreddit": subreddit,
                "num_posts": 100
            })
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            if attempt == max_retries - 1:
                raise
            time.sleep(2 ** attempt)  # Exponential backoff
```

---

## Security Considerations

🔐 **For Production:**

1. **Add Authentication:**
```python
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

security = HTTPBearer()

@app.post("/analyze")
async def analyze(request: AnalysisRequest, credentials: HTTPAuthorizationCredentials = Depends(security)):
    # Verify token
    ...
```

2. **Rate Limiting:**
```python
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

@app.post("/analyze")
@limiter.limit("5/minute")
async def analyze(...):
    ...
```

3. **HTTPS Only:** Use reverse proxy (nginx) with SSL
4. **Environment Variables:** Never hardcode API keys
5. **Input Validation:** Already implemented via Pydantic

---

## API Documentation

Interactive Swagger UI: `http://localhost:8000/docs`
OpenAPI Schema: `http://localhost:8000/openapi.json`

Enjoy the API! 🚀

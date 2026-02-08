"""
FastAPI REST API for Social Network Analysis.
Provides programmatic access to analysis features.
"""
import os
import json
import logging
from typing import Optional, List
from datetime import datetime
import uuid

from fastapi import FastAPI, BackgroundTasks, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("API")

# Create FastAPI app
app = FastAPI(
    title="Social Network Analysis API",
    description="REST API for Reddit social network analysis with advanced analytics",
    version="2.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory task storage (use Redis in production)
tasks = {}

# ===== REQUEST/RESPONSE MODELS =====

class AnalysisRequest(BaseModel):
    subreddit: str = Field(..., description="Subreddit name (without r/)")
    num_posts: int = Field(100, ge=10, le=500, description="Number of posts to analyze")
    time_filter: str = Field("week", description="Time filter: all, year, month, week, day")
    use_gemini: bool = Field(False, description="Enable Google Gemini AI analysis")
    append_mode: str = Field("auto", description="Data handling: auto, always, never")
    enable_advanced: bool = Field(True, description="Enable advanced analytics")

class AnalysisResponse(BaseModel):
    status: str
    task_id: str
    message: str
    subreddit: str

class TaskStatus(BaseModel):
    task_id: str
    status: str  # pending, running, completed, failed
    subreddit: str
    progress: Optional[int] = None
    message: Optional[str] = None
    created_at: str
    completed_at: Optional[str] = None
    error: Optional[str] = None

class SubredditInfo(BaseModel):
    subreddit: str
    nodes: int
    edges: int
    communities: int
    top_influencers: List[dict]
    last_analyzed: str

# ===== BACKGROUND TASK FUNCTIONS =====

def run_analysis_task(task_id: str, request: AnalysisRequest):
    """Run analysis in background."""
    try:
        tasks[task_id]['status'] = 'running'
        tasks[task_id]['progress'] = 10
        
        from ai_sn_analysis_prototype import run_pipeline
        
        # Determine Gemini API key
        gemini_key = os.environ.get('GEMINI_API_KEY') if request.use_gemini else None
        
        logger.info(f"Starting analysis for r/{request.subreddit}")
        
        # Run pipeline
        result = run_pipeline(
            subreddit=request.subreddit,
            posts_limit=request.num_posts,
            outdir='output',
            gemini_api_key=gemini_key,
            time_filter=request.time_filter,
            append_mode=request.append_mode,
            enable_advanced=request.enable_advanced
        )
        
        tasks[task_id]['status'] = 'completed'
        tasks[task_id]['progress'] = 100
        tasks[task_id]['completed_at'] = datetime.now().isoformat()
        tasks[task_id]['result'] = {
            'nodes': len(result['graph'].nodes()),
            'edges': len(result['graph'].edges()),
            'posts_analyzed': len(result['posts'])
        }
        
        logger.info(f"Analysis complete for r/{request.subreddit}")
        
    except Exception as e:
        tasks[task_id]['status'] = 'failed'
        tasks[task_id]['error'] = str(e)
        tasks[task_id]['completed_at'] = datetime.now().isoformat()
        logger.error(f"Analysis failed for r/{request.subreddit}: {e}")

# ===== API ENDPOINTS =====

@app.get("/")
async def root():
    """API root endpoint."""
    return {
        "name": "Social Network Analysis API",
        "version": "2.0.0",
        "endpoints": {
            "analyze": "/analyze - Start new analysis",
            "status": "/status/{task_id} - Check task status",
            "results": "/results/{subreddit} - Get analysis results",
            "list": "/list - List analyzed subreddits",
            "health": "/health - Health check"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "tasks_active": sum(1 for t in tasks.values() if t['status'] in ['pending', 'running'])
    }

@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_subreddit(request: AnalysisRequest, background_tasks: BackgroundTasks):
    """
    Start a new subreddit analysis.
    
    The analysis runs in the background. Use the returned task_id to check progress.
    """
    task_id = str(uuid.uuid4())
    
    # Validate subreddit name
    subreddit = request.subreddit.replace('r/', '').strip()
    if not subreddit:
        raise HTTPException(status_code=400, detail="Invalid subreddit name")
    
    # Create task record
    tasks[task_id] = {
        'task_id': task_id,
        'status': 'pending',
        'subreddit': subreddit,
        'progress': 0,
        'created_at': datetime.now().isoformat(),
        'completed_at': None,
        'error': None
    }
    
    # Start background task
    background_tasks.add_task(run_analysis_task, task_id, request)
    
    return AnalysisResponse(
        status="started",
        task_id=task_id,
        message=f"Analysis of r/{subreddit} started",
        subreddit=subreddit
    )

@app.get("/status/{task_id}", response_model=TaskStatus)
async def get_task_status(task_id: str):
    """Get the status of an analysis task."""
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    return TaskStatus(**tasks[task_id])

@app.get("/results/{subreddit}")
async def get_results(subreddit: str):
    """
    Get analysis results for a subreddit.
    
    Returns comprehensive data including graph metrics, communities, influencers, etc.
    """
    output_dir = "output"
    
    # Load graph data
    graph_json = os.path.join(output_dir, f"{subreddit}_graph.json")
    if not os.path.exists(graph_json):
        raise HTTPException(status_code=404, detail=f"No analysis found for r/{subreddit}")
    
    try:
        # Load all available data
        results = {'subreddit': subreddit}
        
        # Graph data
        with open(graph_json, 'r', encoding='utf-8') as f:
            graph_data = json.load(f)
            results['graph'] = {
                'nodes': len(graph_data['nodes']),
                'edges': len(graph_data['links'])
            }
        
        # Load additional data files
        data_files = {
            'trends': f"{subreddit}_trends.json",
            'validation': f"{subreddit}_validation.json",
            'sentiment_networks': f"{subreddit}_sentiment_networks.json",
            'evolution': f"{subreddit}_evolution.json",
            'layers': f"{subreddit}_layers.json",
            'predictor': f"{subreddit}_predictor.json"
        }
        
        for key, filename in data_files.items():
            filepath = os.path.join(output_dir, filename)
            if os.path.exists(filepath):
                with open(filepath, 'r', encoding='utf-8') as f:
                    results[key] = json.load(f)
        
        return results
        
    except Exception as e:
        logger.error(f"Error loading results for {subreddit}: {e}")
        raise HTTPException(status_code=500, detail=f"Error loading results: {str(e)}")

@app.get("/list")
async def list_analyses():
    """List all analyzed subreddits."""
    output_dir = "output"
    
    if not os.path.exists(output_dir):
        return {"subreddits": []}
    
    # Find all graph JSON files
    subreddits = []
    for filename in os.listdir(output_dir):
        if filename.endswith('_graph.json'):
            subreddit = filename.replace('_graph.json', '')
            
            # Get file modification time
            filepath = os.path.join(output_dir, filename)
            mtime = datetime.fromtimestamp(os.path.getmtime(filepath))
            
            # Load basic stats
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    graph_data = json.load(f)
                    subreddits.append({
                        'subreddit': subreddit,
                        'nodes': len(graph_data['nodes']),
                        'edges': len(graph_data['links']),
                        'last_analyzed': mtime.isoformat()
                    })
            except:
                pass
    
    return {"subreddits": sorted(subreddits, key=lambda x: x['last_analyzed'], reverse=True)}

@app.get("/download/{subreddit}/{file_type}")
async def download_file(subreddit: str, file_type: str):
    """
    Download analysis files.
    
    file_type options: nodes, edges, graph, graphml, html
    """
    output_dir = "output"
    
    file_mapping = {
        'nodes': f"{subreddit}_nodes.csv",
        'edges': f"{subreddit}_edges.csv",
        'graph': f"{subreddit}_graph.json",
        'graphml': f"{subreddit}_graph.graphml",
        'html': f"{subreddit}_graph.html",
        'trends': f"{subreddit}_trends.json"
    }
    
    if file_type not in file_mapping:
        raise HTTPException(status_code=400, detail=f"Invalid file type. Choose from: {list(file_mapping.keys())}")
    
    filepath = os.path.join(output_dir, file_mapping[file_type])
    
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail=f"File not found: {file_mapping[file_type]}")
    
    return FileResponse(filepath, filename=file_mapping[file_type])

@app.delete("/delete/{subreddit}")
async def delete_analysis(subreddit: str):
    """Delete all analysis files for a subreddit."""
    output_dir = "output"
    
    deleted_files = []
    patterns = [
        f"{subreddit}_*.json",
        f"{subreddit}_*.csv",
        f"{subreddit}_*.graphml",
        f"{subreddit}_*.html"
    ]
    
    import glob
    for pattern in patterns:
        for filepath in glob.glob(os.path.join(output_dir, pattern)):
            try:
                os.remove(filepath)
                deleted_files.append(os.path.basename(filepath))
            except Exception as e:
                logger.error(f"Error deleting {filepath}: {e}")
    
    if not deleted_files:
        raise HTTPException(status_code=404, detail=f"No files found for r/{subreddit}")
    
    return {
        "message": f"Deleted {len(deleted_files)} files for r/{subreddit}",
        "files": deleted_files
    }

# ===== RUN SERVER =====

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )

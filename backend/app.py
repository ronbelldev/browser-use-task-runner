"""
FastAPI backend for browser-use task runner.
Provides REST API and Server-Sent Events for real-time updates.
"""

import os
import json
import asyncio
from typing import List, Dict, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from browser_runner import BrowserRunner, run_task_with_streaming
from utils import (
    validate_task_prompt,
    export_steps_to_notebook,
    create_step_data
)

# Load environment variables
load_dotenv()

# Store for active tasks and their results
task_store: Dict[str, Dict[str, Any]] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    print("Starting Browser Use Task Runner API...")
    yield
    # Shutdown
    print("Shutting down Browser Use Task Runner API...")


# Create FastAPI app
app = FastAPI(
    title="Browser Use Task Runner",
    description="API for running browser automation tasks with real-time updates",
    version="1.0.0",
    lifespan=lifespan
)

# Configure CORS for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000", "http://localhost:3001", "http://127.0.0.1:3001"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request/Response Models
class TaskRequest(BaseModel):
    """Request model for running a task."""
    prompt: str = Field(..., min_length=3, max_length=10000, description="The task prompt to execute")
    openai_api_key: str = Field(None, description="OpenAI API key (optional, uses env var if not provided)")


class TaskResponse(BaseModel):
    """Response model for task submission."""
    task_id: str
    message: str
    status: str


class StepData(BaseModel):
    """Model for step data."""
    step_number: int
    step_name: str
    status: str
    code: str = ""
    screenshot: str = ""
    logs: List[str] = []
    error: str = None
    timestamp: str = ""


class ExportRequest(BaseModel):
    """Request model for exporting to notebook."""
    task_id: str
    steps: List[Dict[str, Any]]
    task_prompt: str


class ExportResponse(BaseModel):
    """Response model for notebook export."""
    notebook_json: str
    filename: str


# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "browser-use-task-runner"}


@app.get("/")
async def root():
    """Root endpoint with API info."""
    return {
        "name": "Browser Use Task Runner API",
        "version": "1.0.0",
        "endpoints": {
            "/run-task": "POST - Run a browser automation task (SSE stream)",
            "/export-notebook": "POST - Export steps to Jupyter notebook",
            "/health": "GET - Health check"
        }
    }


@app.post("/run-task")
async def run_task(request: TaskRequest):
    """
    Run a browser automation task with real-time SSE updates.
    
    This endpoint returns a Server-Sent Events stream that provides
    real-time updates as the browser automation progresses.
    """
    # Validate prompt
    is_valid, error_msg = validate_task_prompt(request.prompt)
    if not is_valid:
        raise HTTPException(status_code=400, detail=error_msg)
    
    # Get OpenAI API key
    api_key = request.openai_api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=400,
            detail="OpenAI API key is required. Provide it in the request or set OPENAI_API_KEY environment variable."
        )
    
    async def event_generator():
        """Generate SSE events for the task execution."""
        try:
            async for step_data in run_task_with_streaming(request.prompt, api_key):
                # Format as SSE
                event_data = json.dumps(step_data)
                yield f"data: {event_data}\n\n"
                
                # Small delay to prevent overwhelming the client
                await asyncio.sleep(0.1)
            
            # Send completion event
            yield f"event: complete\ndata: {json.dumps({'status': 'completed'})}\n\n"
            
        except Exception as e:
            error_data = json.dumps({
                "type": "error",
                "error": str(e),
                "status": "failed"
            })
            yield f"event: error\ndata: {error_data}\n\n"
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


@app.post("/export-notebook", response_model=ExportResponse)
async def export_notebook(request: ExportRequest):
    """
    Export automation steps to a Jupyter notebook.
    
    Args:
        request: Contains task_id, steps data, and original prompt
        
    Returns:
        Notebook JSON and suggested filename
    """
    try:
        notebook_json = export_steps_to_notebook(
            steps=request.steps,
            task_prompt=request.task_prompt
        )
        
        # Generate filename
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"browser_automation_{timestamp}.ipynb"
        
        return ExportResponse(
            notebook_json=notebook_json,
            filename=filename
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to export notebook: {str(e)}"
        )


@app.post("/validate-prompt")
async def validate_prompt(request: Dict[str, str]):
    """
    Validate a task prompt before submission.
    
    Args:
        request: Dictionary with 'prompt' key
        
    Returns:
        Validation result
    """
    prompt = request.get("prompt", "")
    is_valid, error_msg = validate_task_prompt(prompt)
    
    return {
        "valid": is_valid,
        "error": error_msg if not is_valid else None
    }


# Demo endpoint for testing without actual browser
@app.post("/run-task-demo")
async def run_task_demo(request: TaskRequest):
    """
    Demo endpoint that simulates browser automation for testing.
    Useful for frontend development without actual browser automation.
    """
    async def demo_event_generator():
        """Generate demo SSE events."""
        demo_steps = [
            {
                "step_number": 1,
                "step_name": "Initializing browser",
                "status": "running",
                "code": "browser = Browser(headless=True)",
                "screenshot": "",
                "logs": ["Starting browser automation..."],
                "timestamp": "2024-01-01T12:00:00"
            },
            {
                "step_number": 1,
                "step_name": "Initializing browser",
                "status": "success",
                "code": "browser = Browser(headless=True)",
                "screenshot": "",
                "logs": ["Starting browser automation...", "Browser initialized successfully"],
                "timestamp": "2024-01-01T12:00:01"
            },
            {
                "step_number": 2,
                "step_name": "Navigating to page",
                "status": "running",
                "code": 'await page.goto("https://example.com")',
                "screenshot": "",
                "logs": ["Navigating to target URL..."],
                "timestamp": "2024-01-01T12:00:02"
            },
            {
                "step_number": 2,
                "step_name": "Navigating to page",
                "status": "success",
                "code": 'await page.goto("https://example.com")',
                "screenshot": "",
                "logs": ["Navigating to target URL...", "Page loaded successfully"],
                "timestamp": "2024-01-01T12:00:03"
            },
            {
                "step_number": 3,
                "step_name": "Extracting content",
                "status": "running",
                "code": 'content = await page.content()',
                "screenshot": "",
                "logs": ["Extracting page content..."],
                "timestamp": "2024-01-01T12:00:04"
            },
            {
                "step_number": 3,
                "step_name": "Extracting content",
                "status": "success",
                "code": 'content = await page.content()',
                "screenshot": "",
                "logs": ["Extracting page content...", "Content extracted: Example Domain"],
                "timestamp": "2024-01-01T12:00:05"
            },
            {
                "step_number": 4,
                "step_name": "Task completed",
                "status": "success",
                "code": "",
                "screenshot": "",
                "logs": ["Browser automation completed successfully"],
                "timestamp": "2024-01-01T12:00:06"
            },
            {
                "type": "completion",
                "success": True,
                "total_steps": 4,
                "total_time": 6.0,
                "error": None
            }
        ]
        
        for step in demo_steps:
            event_data = json.dumps(step)
            yield f"data: {event_data}\n\n"
            await asyncio.sleep(1)  # Simulate real-time delays
        
        yield f"event: complete\ndata: {json.dumps({'status': 'completed'})}\n\n"
    
    return StreamingResponse(
        demo_event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


if __name__ == "__main__":
    import uvicorn
    
    # Get port from environment or use default
    port = int(os.getenv("PORT", 8000))
    
    print(f"Starting server on http://localhost:{port}")
    print("API Documentation available at http://localhost:{port}/docs")
    
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=port,
        reload=True,
        log_level="info"
    )

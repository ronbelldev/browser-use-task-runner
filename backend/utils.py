"""
Utility functions for the browser-use task runner.
Handles screenshot encoding, notebook generation, and helper functions.
"""

import base64
import json
import os
from datetime import datetime
from typing import List, Dict, Any, Optional
import nbformat
from nbformat.v4 import new_notebook, new_code_cell, new_markdown_cell


def encode_screenshot_to_base64(screenshot_bytes: bytes) -> str:
    """
    Encode screenshot bytes to base64 string for transmission to frontend.
    
    Args:
        screenshot_bytes: Raw screenshot bytes from Playwright
        
    Returns:
        Base64 encoded string prefixed with data URI scheme
    """
    if not screenshot_bytes:
        return ""
    
    base64_str = base64.b64encode(screenshot_bytes).decode('utf-8')
    return f"data:image/png;base64,{base64_str}"


def decode_base64_screenshot(base64_str: str) -> bytes:
    """
    Decode base64 screenshot string back to bytes.
    
    Args:
        base64_str: Base64 encoded string (with or without data URI prefix)
        
    Returns:
        Raw screenshot bytes
    """
    # Remove data URI prefix if present
    if base64_str.startswith('data:'):
        base64_str = base64_str.split(',')[1]
    
    return base64.b64decode(base64_str)


def generate_timestamp() -> str:
    """Generate ISO format timestamp for step tracking."""
    return datetime.now().isoformat()


def sanitize_code_for_display(code: str) -> str:
    """
    Sanitize code string for safe display in frontend.
    
    Args:
        code: Raw code string
        
    Returns:
        Sanitized code string
    """
    if not code:
        return ""
    
    # Remove any potentially harmful characters
    # Keep newlines and standard code characters
    return code.strip()


def create_step_data(
    step_number: int,
    step_name: str,
    status: str = "running",
    code: str = "",
    screenshot: str = "",
    logs: List[str] = None,
    error: str = None
) -> Dict[str, Any]:
    """
    Create a standardized step data dictionary.
    
    Args:
        step_number: Sequential step number
        step_name: Human-readable step name
        status: Current status (running, success, failed)
        code: Executed code for this step
        screenshot: Base64 encoded screenshot
        logs: List of log messages
        error: Error message if failed
        
    Returns:
        Standardized step data dictionary
    """
    return {
        "step_number": step_number,
        "step_name": step_name,
        "status": status,
        "code": sanitize_code_for_display(code),
        "screenshot": screenshot,
        "logs": logs or [],
        "error": error,
        "timestamp": generate_timestamp()
    }


def export_steps_to_notebook(steps: List[Dict[str, Any]], task_prompt: str) -> str:
    """
    Export automation steps to a Jupyter notebook format.
    
    Args:
        steps: List of step data dictionaries
        task_prompt: Original task prompt from user
        
    Returns:
        JSON string of the notebook
    """
    nb = new_notebook()
    
    # Add title and introduction
    title_cell = new_markdown_cell(f"""# Browser Automation Task

## Original Prompt
{task_prompt}

## Generated Steps
This notebook contains the automated browser steps executed for the above task.
Generated on: {generate_timestamp()}
""")
    nb.cells.append(title_cell)
    
    # Add setup cell
    setup_code = """# Setup - Install required packages
# !pip install browser-use playwright
# !playwright install chromium

from browser_use import Agent
from langchain_openai import ChatOpenAI
import asyncio
"""
    setup_cell = new_code_cell(setup_code)
    nb.cells.append(setup_cell)
    
    # Add each step
    for step in steps:
        # Markdown cell with step info
        step_md = f"""### Step {step.get('step_number', '?')}: {step.get('step_name', 'Unknown')}

**Status:** {step.get('status', 'unknown')}
**Timestamp:** {step.get('timestamp', 'N/A')}
"""
        if step.get('error'):
            step_md += f"\n**Error:** {step.get('error')}"
        
        if step.get('logs'):
            step_md += f"\n\n**Logs:**\n```\n" + "\n".join(step.get('logs', [])) + "\n```"
        
        md_cell = new_markdown_cell(step_md)
        nb.cells.append(md_cell)
        
        # Code cell if there's code
        if step.get('code'):
            code_cell = new_code_cell(step.get('code'))
            nb.cells.append(code_cell)
    
    # Add summary cell
    summary_md = f"""## Summary

- **Total Steps:** {len(steps)}
- **Successful:** {sum(1 for s in steps if s.get('status') == 'success')}
- **Failed:** {sum(1 for s in steps if s.get('status') == 'failed')}
"""
    summary_cell = new_markdown_cell(summary_md)
    nb.cells.append(summary_cell)
    
    return nbformat.writes(nb)


def save_notebook_to_file(notebook_json: str, filename: str = None) -> str:
    """
    Save notebook JSON to a file.
    
    Args:
        notebook_json: JSON string of the notebook
        filename: Optional filename, generates timestamped name if not provided
        
    Returns:
        Path to saved file
    """
    if not filename:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"browser_automation_{timestamp}.ipynb"
    
    # Ensure output directory exists
    output_dir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(output_dir, exist_ok=True)
    
    filepath = os.path.join(output_dir, filename)
    
    with open(filepath, 'w') as f:
        f.write(notebook_json)
    
    return filepath


def format_error_message(error: Exception) -> str:
    """
    Format an exception into a user-friendly error message.
    
    Args:
        error: The exception to format
        
    Returns:
        Formatted error message string
    """
    error_type = type(error).__name__
    error_msg = str(error)
    
    return f"{error_type}: {error_msg}"


def validate_task_prompt(prompt: str) -> tuple[bool, str]:
    """
    Validate the task prompt from user.
    
    Args:
        prompt: User's task prompt
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not prompt:
        return False, "Task prompt cannot be empty"
    
    if len(prompt.strip()) < 3:
        return False, "Task prompt is too short"
    
    if len(prompt) > 10000:
        return False, "Task prompt is too long (max 10000 characters)"
    
    return True, ""

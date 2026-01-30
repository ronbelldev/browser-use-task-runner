"""
Browser automation runner using browser-use library.
Handles the execution of browser tasks and streams updates.
"""

import asyncio
import traceback
import os
from typing import AsyncGenerator, Dict, Any, List, Callable, Optional
from dataclasses import dataclass, field
import json
import base64

from browser_use import Agent, Browser
from langchain_openai import ChatOpenAI

from utils import (
    encode_screenshot_to_base64,
    create_step_data,
    format_error_message,
    generate_timestamp
)


@dataclass
class TaskResult:
    """Result of a browser automation task."""
    success: bool
    steps: List[Dict[str, Any]] = field(default_factory=list)
    error: Optional[str] = None
    total_time: float = 0.0


class BrowserRunner:
    """
    Handles browser automation tasks using browser-use library.
    Provides streaming updates for real-time dashboard display.
    """
    
    def __init__(self, openai_api_key: str = None):
        """
        Initialize the browser runner.
        
        Args:
            openai_api_key: OpenAI API key for the LLM agent
        """
        self.openai_api_key = openai_api_key
        self.current_step = 0
        self.steps: List[Dict[str, Any]] = []
        self.browser: Optional[Browser] = None
        self._stop_requested = False
        self._update_callback: Optional[Callable] = None
        
    def _create_step_callback(self):
        """Create callback for browser-use agent steps."""
        async def on_step(browser_state, agent_output, step_number):
            """Called after each step by browser-use agent."""
            self.current_step = step_number + 1  # step_number is 0-indexed
            
            # Extract action information
            action_info = ""
            step_name = f"Step {self.current_step}"
            
            if agent_output and hasattr(agent_output, 'action'):
                action = agent_output.action
                if action:
                    try:
                        if hasattr(action, 'model_dump'):
                            action_dict = action.model_dump()
                            action_info = json.dumps(action_dict, indent=2, default=str)
                            # Get a more descriptive step name
                            for key in action_dict:
                                if action_dict[key] is not None and key not in ['index', 'xpath', 'element_description']:
                                    step_name = f"{key.replace('_', ' ').title()}"
                                    break
                        else:
                            action_info = str(action)
                    except Exception:
                        action_info = str(action)
            
            # Get screenshot from browser state
            screenshot = ""
            if browser_state and hasattr(browser_state, 'screenshot') and browser_state.screenshot:
                try:
                    if isinstance(browser_state.screenshot, bytes):
                        screenshot = encode_screenshot_to_base64(browser_state.screenshot)
                    elif isinstance(browser_state.screenshot, str):
                        if not browser_state.screenshot.startswith('data:'):
                            screenshot = f"data:image/png;base64,{browser_state.screenshot}"
                        else:
                            screenshot = browser_state.screenshot
                except Exception as e:
                    print(f"Screenshot error: {e}")
            
            # Extract any result text
            logs = []
            if browser_state and hasattr(browser_state, 'url'):
                logs.append(f"URL: {browser_state.url}")
            
            step_data = create_step_data(
                step_number=self.current_step,
                step_name=step_name,
                status="success",
                code=action_info,
                screenshot=screenshot,
                logs=logs
            )
            
            self.steps.append(step_data)
            
            if self._update_callback:
                await self._update_callback(step_data)
        
        return on_step
        
    async def run_task(
        self,
        task_prompt: str,
        update_callback: Callable[[Dict[str, Any]], Any]
    ) -> TaskResult:
        """
        Run a browser automation task.
        
        Args:
            task_prompt: The task description from the user
            update_callback: Async function to call with step updates
            
        Returns:
            TaskResult with all steps and final status
        """
        import time
        start_time = time.time()
        
        self.current_step = 0
        self.steps = []
        self._stop_requested = False
        self._update_callback = update_callback
        
        try:
            # Send initial status
            init_step = create_step_data(
                step_number=0,
                step_name="Initializing browser",
                status="running",
                code="",
                screenshot="",
                logs=["Starting browser automation..."]
            )
            await update_callback(init_step)
            
            # Set the API key in environment if provided
            if self.openai_api_key:
                os.environ["OPENAI_API_KEY"] = self.openai_api_key
            
            # Create browser session with headless config
            self.browser = Browser(
                headless=True,
                disable_security=True
            )
            
            # Update initialization step
            init_step["status"] = "success"
            init_step["logs"].append("Browser initialized successfully")
            await update_callback(init_step)
            self.steps.append(init_step)
            
            # Create LLM
            llm = ChatOpenAI(
                model="gpt-4o",
                api_key=self.openai_api_key
            )
            
            # Send task start update
            self.current_step = 1
            task_step = create_step_data(
                step_number=self.current_step,
                step_name="Starting task execution",
                status="running",
                code=f'task = "{task_prompt}"',
                screenshot="",
                logs=[f"Executing task: {task_prompt}"]
            )
            await update_callback(task_step)
            
            # Create agent with step callback
            agent = Agent(
                task=task_prompt,
                llm=llm,
                browser=self.browser,
                register_new_step_callback=self._create_step_callback()
            )
            
            # Run the agent
            history = await agent.run(max_steps=15)
            
            # Update task step as complete
            task_step["status"] = "success"
            task_step["logs"].append("Task execution completed")
            
            # Final step
            final_step = create_step_data(
                step_number=self.current_step + 1,
                step_name="Task completed",
                status="success",
                code="",
                screenshot="",
                logs=["Browser automation completed successfully"]
            )
            self.steps.append(final_step)
            await update_callback(final_step)
            
            total_time = time.time() - start_time
            
            return TaskResult(
                success=True,
                steps=self.steps,
                total_time=total_time
            )
            
        except Exception as e:
            error_msg = format_error_message(e)
            traceback_str = traceback.format_exc()
            
            # Send error step
            error_step = create_step_data(
                step_number=self.current_step + 1,
                step_name="Error occurred",
                status="failed",
                code="",
                screenshot="",
                logs=[traceback_str],
                error=error_msg
            )
            self.steps.append(error_step)
            await update_callback(error_step)
            
            total_time = time.time() - start_time
            
            return TaskResult(
                success=False,
                steps=self.steps,
                error=error_msg,
                total_time=total_time
            )
            
        finally:
            # Cleanup
            await self._cleanup()
    
    async def _cleanup(self):
        """Clean up browser resources."""
        try:
            if self.browser:
                await self.browser.close()
        except Exception as e:
            print(f"Cleanup error: {e}")
        finally:
            self.browser = None
    
    def request_stop(self):
        """Request the current task to stop."""
        self._stop_requested = True


async def run_task_with_streaming(
    task_prompt: str,
    openai_api_key: str
) -> AsyncGenerator[Dict[str, Any], None]:
    """
    Run a browser task and yield step updates as they occur.
    
    Args:
        task_prompt: The task to execute
        openai_api_key: OpenAI API key
        
    Yields:
        Step data dictionaries as they are produced
    """
    update_queue = asyncio.Queue()
    
    async def queue_update(step_data: Dict[str, Any]):
        await update_queue.put(step_data)
    
    runner = BrowserRunner(openai_api_key=openai_api_key)
    
    # Start the task in background
    task = asyncio.create_task(runner.run_task(task_prompt, queue_update))
    
    # Yield updates as they come
    while not task.done():
        try:
            step_data = await asyncio.wait_for(update_queue.get(), timeout=0.5)
            yield step_data
        except asyncio.TimeoutError:
            continue
    
    # Get any remaining updates
    while not update_queue.empty():
        step_data = await update_queue.get()
        yield step_data
    
    # Get the final result
    result = await task
    
    # Yield final completion message
    yield {
        "type": "completion",
        "success": result.success,
        "total_steps": len(result.steps),
        "total_time": result.total_time,
        "error": result.error
    }

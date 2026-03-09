import json
import logging
import os
import subprocess
import sys
import time
from typing import List, Optional
from uuid import uuid4
from celery.result import AsyncResult
from fastapi import WebSocket
from pathlib import Path
import redis

def import_stefan_fl():
    import os
    import sys
    sys.path.append(str(Path(__file__).parents[1]))
    sys.path.append(str(Path(os.environ.get("STEFAN_FL_PATH", "/mnt/d/ST/experimental/llm_fl_code/openai")).parent))
    import stefan_fl

import_stefan_fl()

# Celery configuration
CELERY_BROKER_URL = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0")
CELERY_RESULT_BACKEND = os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/0")

# Initialize Redis client for task state manager (shared with Celery broker)
_redis_client = redis.from_url(CELERY_RESULT_BACKEND, decode_responses=True)

# Initialize task state manager on module load
from task_state_manager import initialize_task_state_manager, get_task_state_manager
initialize_task_state_manager(redis_client=_redis_client)

from fl_types import Log, Task, SingleTask, TaskAgent, TaskCreationForm, TaskData, TaskNodeStatus


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        logger.info(
            f"Websocket client {websocket.client}:{websocket.url} disconnected"
        )
        self.active_connections.remove(websocket)

    async def send_json(self, websocket: WebSocket, data: any):
        await websocket.send_json(data)


APP_DIR = Path(__file__).parent
MODELS_DIR = APP_DIR / "models"

def build_task_object(form: TaskCreationForm, demo: bool = False) -> SingleTask:
    """Build distributed SingleTask objects (navigator + agents), save to Redis, and start Celery task execution"""    
    from celery_tasks import execute_task, get_server_queue

    task_uuid = str(uuid4())
    state_manager = get_task_state_manager()
    navigator_id = state_manager.id if hasattr(state_manager, 'id') and state_manager.id else "navigator"
    
    # Create agent SingleTask objects
    distributed_tasks = {}
    for agent_id in form.agents:
        agent_task_uuid = str(uuid4())
        distributed_tasks[agent_id] = agent_task_uuid
        
        # Create agent SingleTask
        agent_task = SingleTask(
            uuid=agent_task_uuid,
            device_id=agent_id,
            status=TaskNodeStatus.PREPARED,
            data=TaskData(task_info={
                "parent_task_uuid": task_uuid,
                "agent_id": agent_id
            }),
            logs=[],
            metrics=[],
            distributed_tasks={}
        )
        state_manager.save_single_task(agent_task)
        logger.info(f"Created agent SingleTask {agent_task_uuid} for agent {agent_id}")
    
    # Create navigator SingleTask
    navigator_task = SingleTask(
        uuid=task_uuid,
        device_id=navigator_id,
        status=TaskNodeStatus.PREPARED,
        data=TaskData(task_info=form.task_info),
        logs=[],
        metrics=[],
        distributed_tasks=distributed_tasks
    )
    state_manager.save_single_task(navigator_task)
    logger.info(f"Created navigator SingleTask {task_uuid} with {len(distributed_tasks)} agents")
    
    # Start Celery task execution
    execute_task.apply_async(args=[task_uuid], queue=get_server_queue(), task_id=task_uuid)
    logger.info(f"Started Celery task {task_uuid}")
    
    return navigator_task

def retrive_model(model_name: str) -> Optional[dict]:
    model = MODELS_DIR / model_name
    model_zip_file = model / "model.zip"
    metadata_file = model / "metadata.json"
    
    if not model_zip_file.exists() or not metadata_file.exists():
        return
    if not model_zip_file.is_file() or not metadata_file.is_file():
        return
    
    with open(metadata_file, "r") as f:
        metadata = json.load(f)
        if "model_fn" not in metadata or "dataloader_fn" not in metadata:
            return
    
    return {
        "model_zip_file": str(model_zip_file.absolute()),
        "metadata_file": str(metadata_file.absolute()),
        "metadata": metadata,
    }


def create_custom_model_from_task_info(task_info: dict, temp_dir: Path) -> Optional[dict]:
    """
    Create a temporary model from custom_model_zip_base64 and custom_model_metadata_json in task_info.
    Returns model_info dict similar to retrive_model, or None if custom model not provided.
    The temp_dir will be cleaned up automatically when the context exits.
    """
    import base64
    import zipfile
    
    custom_model_zip_base64 = task_info.get("custom_model_zip_base64")
    custom_model_metadata_json = task_info.get("custom_model_metadata_json")
    
    if not custom_model_zip_base64 or not custom_model_metadata_json:
        logger.warning("Custom model data missing: zip_base64 or metadata_json not found in task_info")
        return None
    
    logger.info(f"Processing custom model: zip_base64 length={len(custom_model_zip_base64)}, metadata_json length={len(custom_model_metadata_json)}")
    
    # Parse metadata
    try:
        metadata = json.loads(custom_model_metadata_json)
        if "model_fn" not in metadata or "dataloader_fn" not in metadata:
            logger.error(f"Custom model metadata missing required fields. Got keys: {list(metadata.keys())}")
            return None
        logger.info(f"Parsed metadata: model_fn={metadata.get('model_fn')}, dataloader_fn={metadata.get('dataloader_fn')}")
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse custom model metadata as JSON: {e}")
        logger.error(f"Metadata content (first 200 chars): {custom_model_metadata_json[:200]}")
        return None
    
    # Decode base64 zip file
    try:
        zip_content = base64.b64decode(custom_model_zip_base64)
        logger.info(f"Decoded zip file: {len(zip_content)} bytes")
    except Exception as e:
        logger.error(f"Failed to decode custom model zip file from base64: {e}")
        return None
    
    # Create temp model directory
    model_temp_dir = temp_dir / "custom_model"
    model_temp_dir.mkdir(parents=True, exist_ok=True)
    
    # Write zip file
    model_zip_file = model_temp_dir / "model.zip"
    try:
        with open(model_zip_file, "wb") as f:
            f.write(zip_content)
        logger.info(f"Written zip file to {model_zip_file} ({len(zip_content)} bytes)")
    except Exception as e:
        logger.error(f"Failed to write zip file to {model_zip_file}: {e}")
        return None
    
    # Write metadata.json
    metadata_file = model_temp_dir / "metadata.json"
    try:
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Written metadata.json to {metadata_file}")
    except Exception as e:
        logger.error(f"Failed to write metadata.json to {metadata_file}: {e}")
        return None
    
    logger.info(f"Successfully created temporary custom model at {model_temp_dir}")
    
    return {
        "model_zip_file": str(model_zip_file.absolute()),
        "metadata_file": str(metadata_file.absolute()),
        "metadata": metadata,
        "is_temp": True,  # Flag to indicate this is a temporary model
    }

def get_available_models() -> List[dict]:
    models = []
    for model in MODELS_DIR.glob("*"):
        if model.is_dir():
            metadata = retrive_model(model.name)
            if metadata is None or "metadata" not in metadata:
                continue
            
            models.append({
                "id": model.name,
                "name": model.name,
                "dataset": metadata["metadata"],
            })
    return models

def find_task():
    """Find a task by ID from Redis, composed from distributed SingleTask objects"""
    async def _cb(task_id: str) -> Optional[Task]:
        from celery_app import celery_app
        state_manager = get_task_state_manager()
        
        # Get composed task from distributed SingleTasks
        task = state_manager.get_task(task_id)
        if not task:
            return None
        
        # Sync Celery status to Redis
        result = AsyncResult(task_id, app=celery_app)
        celery_status = result.status
        
        # Map Celery statuses to TaskNodeStatus
        status_map = {
            "PENDING": TaskNodeStatus.PREPARED,
            "STARTED": TaskNodeStatus.STARTED,
            "SUCCESS": TaskNodeStatus.ENDED,
            "FAILURE": TaskNodeStatus.ERROR,
            "REVOKED": TaskNodeStatus.ERROR,
            "RETRY": TaskNodeStatus.STARTED,
        }
        
        # Get the mapped status, default to UNKNOWN if not in map
        mapped_status = status_map.get(celery_status, TaskNodeStatus.UNKNOWN)
        
        # Update navigator SingleTask status if it has changed
        navigator_task = state_manager.get_single_task(task_id)
        if navigator_task and navigator_task.status != mapped_status:
            navigator_task.status = mapped_status
            
            # Add error log for failure cases
            if celery_status == "FAILURE":
                error_info = result.info if hasattr(result, 'info') and result.info else 'Unknown error'
                error_log = Log(
                    log=f"Task {task_id} failed: {error_info}",
                    prefix="navigator",
                    level="error",
                    ts=time.time()
                )
                navigator_task.logs.append(error_log)
            
            state_manager.save_single_task(navigator_task)
            # Re-compose task to reflect changes
            task = state_manager.get_task(task_id)
        
        return task
    return _cb

def find_task_agent():
    async def _cb(task_id: str, agent_id: str) -> Optional[TaskAgent]:
        task = await find_task()(task_id)
        if task:
            return task.agents[agent_id]
    return _cb

def install_requirements(requirements_file="requirements.txt"):
    """
    Installs packages listed in a requirements.txt file using pip.
    """
    try:
        # Construct the pip command
        pip_command = [sys.executable, "-m", "pip", "install", "-r", requirements_file]
        
        # Execute the command
        process = subprocess.run(pip_command, capture_output=True, text=True, check=True)
        
        # Print output if successful
        logger.info("Requirements installed successfully:")
        logger.info(process.stdout)
        return True
    except subprocess.CalledProcessError as e:
        # Handle errors during the installation
        logger.error(f"Error installing requirements: {e}")
        logger.error(f"Stderr: {e.stderr}")
        logger.error(f"Stdout: {e.stdout}")
    except FileNotFoundError:
        logger.error(f"Error: The file '{requirements_file}' was not found.")
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
    return False
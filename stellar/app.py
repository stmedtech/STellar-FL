from contextlib import asynccontextmanager
import json
import logging
import asyncio
import traceback
from pathlib import Path
import shutil
from typing import Any, List, Optional
import io
import zipfile

from celery.result import AsyncResult
from fastapi import Request, WebSocket, WebSocketDisconnect
from fastapi import FastAPI, Query
from fastapi.responses import FileResponse, JSONResponse, Response
from starlette.middleware.base import BaseHTTPMiddleware
import uvicorn

from utils import (
    ConnectionManager,
    get_available_models,
    find_task,
    build_task_object,
)
from fl_types import (
    FLArchitecture,
    TaskEntry,
    TaskCreationForm,
)
from task_state_manager import get_task_state_manager
from celery_app import celery_app
from celery_tasks import generate_model_codebase, get_server_queue

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize task state manager and default devices on application startup"""
    # This ensures the manager is initialized even if celery_app hasn't been imported
    state_manager = get_task_state_manager()
    logger.info("Task state manager initialized")
    yield

app = FastAPI(lifespan=lifespan)

class GlobalExceptionMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        try:
            response = await call_next(request)
            return response
        except ValueError as ve:
            logger.error(f"ValueError caught: {ve}")
            traceback.print_exc()
            return JSONResponse(
                status_code=400,
                content={"error": "Invalid input", "detail": str(ve)}
            )
        except Exception as e:
            logger.exception("Unhandled error occurred")
            traceback.print_exc()
            return JSONResponse(
                status_code=500,
                content={"error": "Internal server error", "detail": str(e)}
            )

app.add_middleware(GlobalExceptionMiddleware)

@app.get("/system/healthcheck")
async def healthcheck():
    return "Healthy"

@app.get("/system/ports")
async def get_ports(port: Optional[int] = Query(default=None)):
    if port is None:
        return 8001
    return port

@app.get("/system/agents")
async def get_agents():
    """Get all devices from Redis"""
    state_manager = get_task_state_manager()
    return state_manager.get_devices()

@app.post("/system/clear")
async def system_clear():
    state_manager = get_task_state_manager()
    for task_id in state_manager.list_task_ids():
        try:
            task_result = AsyncResult(task_id, app=celery_app)
            if task_result.status == "SUCCESS":
                workspace_zip_file = Path(task_result.get()["config"]["job_output_dir"])
                if workspace_zip_file.exists():
                    if workspace_zip_file.is_file():
                        workspace_zip_file.unlink()
                    else:
                        shutil.rmtree(workspace_zip_file)
            task_result.forget()
            state_manager.delete_task(task_id)
        except Exception as e:
            logger.error(f"Error clearing task {task_id}: {e}")
            continue
    return len(state_manager.list_task_ids()) == 0

@app.post("/tasks/models/generate")
async def generate_model(model_name: str, prompt: str):
    result = generate_model_codebase.apply_async(args=[model_name, prompt], queue=get_server_queue())
    return result.id

@app.get("/tasks/models/{task_id}")
async def get_model(task_id: str):
    result = AsyncResult(task_id, app=celery_app)
    return result.status

app.get("/tasks/models")(get_available_models)
app.get("/tasks/openfl/models")(get_available_models)

@app.get("/tasks/nvflare/models")
async def get_nvflare_models():
    return [model["id"] for model in get_available_models()]

@app.get("/tasks")
def get_tasks():
    state_manager = get_task_state_manager()
    
    tasks_list: List[TaskEntry] = []
    for task_id in state_manager.list_task_ids():
        task = state_manager.get_task(task_id)
        if task and task.navigator_data.task_info.get("reference_token"):
            architecture = FLArchitecture(task.navigator_data.task_info.get("architecture", FLArchitecture.nvflare.value))
            tasks_list.append(TaskEntry(
                id=task_id,
                type=architecture,
                status=task.navigator_status,
            ))
    
    return tasks_list

app.get("/tasks/{task_id}")(find_task())

@app.post("/tasks/{task_id}/download")
async def download_task(task_id: str):
    state_manager = get_task_state_manager()
    task = state_manager.get_task(task_id)
    if task:
        task_result = AsyncResult(task_id, app=celery_app)
        file_name = f"{task_id}.zip"
        if task_result.status == "PENDING" or task_result.status == "STARTED":
            return Response(
                content="Task is still running",
                media_type="text/plain",
                headers={"Content-Disposition": f"attachment; filename={file_name}"}
            )
        elif task_result.status == "SUCCESS":
            workspace_zip_file = task_result.get()["config"]["job_output_dir"]
            return FileResponse(workspace_zip_file, media_type="application/zip", filename=file_name)
        else:
            # Return an empty zip file if task is ended but not successful
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                pass  # Create empty zip
            zip_buffer.seek(0)
            return Response(
                content=zip_buffer.getvalue(),
                media_type="application/zip",
                headers={"Content-Disposition": f"attachment; filename={file_name}"}
            )
        raise ValueError(f"Task {task_id} failed: {task_result.status}")
    else:
        raise ValueError(f"Task {task_id} not found")


@app.post("/tasks/{architecture}")
async def create_task(architecture: str, form: TaskCreationForm):
    for arch in FLArchitecture:
        if arch.value.lower() == architecture.lower():
            architecture = arch
            break
    else:
        raise ValueError(f"Invalid architecture: {architecture}")

    state_manager = get_task_state_manager()

    # Log custom model data if present
    has_custom_model = bool(form.task_info.get("custom_model_zip_base64") and form.task_info.get("custom_model_metadata_json"))
    if has_custom_model:
        zip_size = len(form.task_info.get("custom_model_zip_base64", ""))
        metadata_size = len(form.task_info.get("custom_model_metadata_json", ""))
        logger.info(f"Received custom model in task_info: zip_base64_size={zip_size}, metadata_json_size={metadata_size}")
    else:
        logger.info(f"No custom model in task_info. Model name: {form.task_info.get('model_name')}")

    # Build task object and save to Redis (creates distributed SingleTasks)
    form.task_info["architecture"] = architecture.value
    navigator_task = build_task_object(form, demo=True)
    logger.info(f"Created and saved navigator task {navigator_task.uuid} to Redis with {len(navigator_task.distributed_tasks)} agents")
    
    return navigator_task.uuid

manager = ConnectionManager()

async def stream_attribute(target: Any, attribute_key: str, update_s=1):
    """Stream attribute changes from a target object"""
    last_attribute = ""
    while True:
        current_attribute = str(getattr(target, attribute_key))
        if current_attribute != last_attribute:
            yield current_attribute
            last_attribute = current_attribute
        await asyncio.sleep(update_s)


async def stream_list(target: list, update_s=1):
    """Stream list updates, sending new items as they're added"""
    yield target
    last_i = len(target)
    while True:
        current_i = len(target)
        if current_i > last_i:
            yield target[last_i:current_i]
            last_i = current_i
        await asyncio.sleep(update_s)




@app.websocket("/ws/system/agents")
async def ws_agents(websocket: WebSocket):
    """Stream device updates from Redis"""
    state_manager = get_task_state_manager()
    await manager.connect(websocket)
    try:
        last_val = ""
        while True:
            devices = state_manager.get_devices()
            current_val = json.dumps(devices)
            if current_val != last_val:
                await manager.send_json(websocket, devices)
                last_val = current_val
            await asyncio.sleep(5)
    except WebSocketDisconnect:
        manager.disconnect(websocket)


@app.websocket("/ws/tasks/{task_id}/metrics")
async def ws_metrics(websocket: WebSocket, task_id: str):
    state_manager = get_task_state_manager()
    await manager.connect(websocket)
    try:
        # First send all existing metrics
        task = state_manager.get_task(task_id)
        if task and task.metrics:
            await manager.send_json(websocket, [metric.model_dump(mode='json') for metric in task.metrics])
        
        # Then stream new metrics from Redis
        last_metrics_count = len(task.metrics) if task else 0
        while True:
            current_task = state_manager.get_task(task_id)
            if current_task and len(current_task.metrics) > last_metrics_count:
                new_metrics = current_task.metrics[last_metrics_count:]
                await manager.send_json(websocket, [metric.model_dump(mode='json') for metric in new_metrics])
                last_metrics_count = len(current_task.metrics)
            elif not current_task:
                # Task not found
                break
            await asyncio.sleep(1)
    except WebSocketDisconnect:
        manager.disconnect(websocket)


@app.websocket("/ws/tasks/{task_id}/navigator_status")
async def ws_navigator_status(websocket: WebSocket, task_id: str):
    state_manager = get_task_state_manager()
    await manager.connect(websocket)
    try:
        last_status = None
        while True:
            task = state_manager.get_task(task_id)
            if task and task.navigator_status != last_status:
                await manager.send_json(websocket, task.navigator_status.value)
                last_status = task.navigator_status
            elif not task:
                # Task not found
                break
            await asyncio.sleep(1)
    except WebSocketDisconnect:
        manager.disconnect(websocket)


@app.websocket("/ws/tasks/{task_id}/navigator_logs")
async def ws_navigator_logs(websocket: WebSocket, task_id: str):
    state_manager = get_task_state_manager()
    await manager.connect(websocket)
    try:
        # First send all existing logs
        task = state_manager.get_task(task_id)
        if task and task.navigator_logs:
            await manager.send_json(websocket, [log.model_dump(mode='json') for log in task.navigator_logs])
        
        # Then stream new logs from Redis
        last_logs_count = len(task.navigator_logs) if task else 0
        while True:
            current_task = state_manager.get_task(task_id)
            if current_task and len(current_task.navigator_logs) > last_logs_count:
                new_logs = current_task.navigator_logs[last_logs_count:]
                await manager.send_json(websocket, [log.model_dump(mode='json') for log in new_logs])
                last_logs_count = len(current_task.navigator_logs)
            elif not current_task:
                # Task not found
                break
            await asyncio.sleep(1)
    except WebSocketDisconnect:
        manager.disconnect(websocket)


@app.websocket("/ws/tasks/{task_id}/agent/{agent_id}/status")
async def ws_agent_status(websocket: WebSocket, task_id: str, agent_id: str):
    state_manager = get_task_state_manager()
    await manager.connect(websocket)
    try:
        last_status = None
        while True:
            task = state_manager.get_task(task_id)
            if task and agent_id in task.agents:
                agent = task.agents[agent_id]
                if agent.status != last_status:
                    await manager.send_json(websocket, agent.status.value)
                    last_status = agent.status
            elif not task:
                # Task not found
                break
            await asyncio.sleep(1)
    except WebSocketDisconnect:
        manager.disconnect(websocket)


@app.websocket("/ws/tasks/{task_id}/agent/{agent_id}/logs")
async def ws_agent_logs(websocket: WebSocket, task_id: str, agent_id: str):
    state_manager = get_task_state_manager()
    await manager.connect(websocket)
    try:
        # First send all existing logs
        task = state_manager.get_task(task_id)
        if task and agent_id in task.agents:
            agent = task.agents[agent_id]
            if agent.logs:
                await manager.send_json(websocket, [log.model_dump(mode='json') for log in agent.logs])
            last_logs_count = len(agent.logs)
        else:
            last_logs_count = 0
        
        # Then stream new logs from Redis
        while True:
            current_task = state_manager.get_task(task_id)
            if current_task and agent_id in current_task.agents:
                agent = current_task.agents[agent_id]
                if len(agent.logs) > last_logs_count:
                    new_logs = agent.logs[last_logs_count:]
                    await manager.send_json(websocket, [log.model_dump(mode='json') for log in new_logs])
                    last_logs_count = len(agent.logs)
            elif not current_task:
                # Task not found
                break
            await asyncio.sleep(1)
    except WebSocketDisconnect:
        manager.disconnect(websocket)


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=1524, reload=True)
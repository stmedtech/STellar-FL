from dataclasses import dataclass
import json
import logging
import random
import threading
import time
import asyncio
from pathlib import Path
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from uuid import uuid4
from pydantic import BaseModel

from fastapi import WebSocket, WebSocketDisconnect
from fastapi import FastAPI, Query
import uvicorn

from fl_types import (
    FLArchitecture,
    TaskNodeStatus,
    DeviceStatus,
    TaskCreationForm,
    Task,
    TaskAgent,
    TaskData,
    Log,
    LogType,
    Metric,
)
from utils import ConnectionManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

APP_DIR = Path(__file__).parent
MODELS_DIR = APP_DIR / "models"

USER_REFERENCE_TOKEN = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VyX2lkIjoxLCJ1c2VyX2VtYWlsIjoiaHN1Lmpvc2VwaC5vdXlhbmdAc3RtZWRpY2FsLnR3IiwidG9rZW5faWQiOiJEZW1vIn0.afdiG8fkfJRiUSCbUr9cpf6OQvSv_roC9vfypeipjk8"

device_names = [f"agent_{i}" for i in range(3)]
devices = {
    device: {
        "id": device,
        "reference_token": USER_REFERENCE_TOKEN,
        "status": DeviceStatus.HEALTHY,
        "last_ts": time.time(),
        "sys_info": {
            "system": "linux",
            "release": "22.04",
            "version": "5.15.0-100-generic",
            "machine": "x86_64",
            "processor": "Intel(R) Core(TM) i7-10750H CPU @ 2.60GHz",
            "in_docker": False,
            "security_information": {
                "public_key": "public_key",
            },
            "CPU_information": {
                "physical_cores": 10,
                "total_cores": 10,
            },
            "RAM_information": {
                "total": "64GB",
            },
            "GPU_information": [
                {
                    "ID": 0,
                    "UUID": "uuid_0",
                    "name": "NVIDIA GeForce RTX 3090",
                    "memory": "12GB",
                }
            ]
        }
    }
    for device in device_names
}

def build_task_object(architecture: FLArchitecture, form: TaskCreationForm, demo: bool = False) -> Task:
    def _update_task(task: Task):
        last_i = 0
        
        time.sleep(random.random() * 10)
        
        def _update_status(status: TaskNodeStatus):
            task.navigator_status = status
            for agent_name in task.agents.keys():
                task.agents[agent_name].status = status
        
        def _update_logs(message: str, level: str):
            task.navigator_logs.append(Log(log=message, prefix="navigator", level=level, ts=time.time()))
            for agent_name in task.agents.keys():
                task.agents[agent_name].logs.append(Log(log=message, prefix=agent_name, level=level, ts=time.time()))
        
        _update_status(TaskNodeStatus.STARTED)
        
        while True:
            last_value = task.metrics[-1].value if len(task.metrics) > 0 else 0
            for agent_name in task.agents.keys():
                new_value = last_value + random.random()
                if new_value > 100:
                    _update_status(TaskNodeStatus.ENDED)
                    return
                task.metrics.append(Metric(site=agent_name, tag=f"demo_metric_accuracy", step=last_i, value=new_value))
            last_i += 1
            
            _update_logs("Hello, world!", "info")
            
            time.sleep(1)
    
    task = Task(
        uuid=uuid4().hex,
        navigator_id=device_names[0],
        navigator_status=TaskNodeStatus.PREPARED,
        navigator_data=TaskData(task_info=form.task_info),
        navigator_logs=[],
        agents={device: TaskAgent(status=TaskNodeStatus.PREPARED, logs=[]) for device in device_names},
        metrics=[],
    )
    
    if demo:
        threading.Thread(target=_update_task, args=(task,), daemon=True).start()
    
    return task

def get_available_models() -> List[dict]:
    models = []
    for model in MODELS_DIR.glob("*"):
        if model.is_dir():
            model_zip_file = model / "model.zip"
            metadata_file = model / "metadata.json"
            
            if not model_zip_file.exists() or not metadata_file.exists():
                continue
            if not model_zip_file.is_file() or not metadata_file.is_file():
                continue
            
            with open(metadata_file, "r") as f:
                metadata = json.load(f)
                if "model_fn" not in metadata or "dataloader_fn" not in metadata:
                    continue
            
            models.append({
                "id": model.name,
                "name": model.name,
                "dataset": metadata,
            })
    return models

tasks: Dict[FLArchitecture, Dict[str, Task]] = {}

def find_task():
    async def _cb(task_id: str) -> Optional[Task]:
        task = next((task for _, tasks_dict in tasks.items() for task in tasks_dict.values() if task.uuid == task_id), None)
        return task
    return _cb

def find_task_agent():
    async def _cb(task_id: str, agent_id: str) -> Optional[TaskAgent]:
        task = await find_task()(task_id)
        if task:
            return task.agents[agent_id]
    return _cb

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
    return devices

@app.post("/system/clear")
async def system_clear():
    return True

app.get("/tasks/models")(get_available_models)
app.get("/tasks/openfl/models")(get_available_models)

@app.get("/tasks/nvflare/models")
async def get_nvflare_models():
    return [model["id"] for model in get_available_models()]

@app.get("/tasks")
def get_tasks():
    return [
        {
            "id": task_id,
            "type": architecture.value,
        }
        for architecture, tasks_dict in tasks.items()
        for task_id in tasks_dict.keys()
    ]

app.get("/tasks/{task_id}")(find_task())


@app.post("/tasks/{architecture}")
async def create_task(architecture: str, form: TaskCreationForm):
    for arch in FLArchitecture:
        if arch.value.lower() == architecture.lower():
            architecture = arch
            break
    else:
        raise ValueError(f"Invalid architecture: {architecture}")

    task = build_task_object(architecture, form)
    if architecture not in tasks:
        tasks[architecture] = {}
    tasks[architecture][task.uuid] = task
    return task.uuid

manager = ConnectionManager()

async def stream_attribute(target: Any, attribute_key: str, update_s=5):
    last_attribute = ""
    while True:
        current_attribute = str(getattr(target, attribute_key))
        if current_attribute != last_attribute:
            yield current_attribute
            last_attribute = current_attribute
        await asyncio.sleep(update_s)


async def stream_list(target: list, update_s=5):
    yield target
    last_i = len(target)
    while True:
        current_i = len(target)
        if current_i > last_i:
            yield target[last_i:current_i]
            last_i = current_i
        await asyncio.sleep(update_s)


async def stream_dict(target: dict, seriize_fn=None, update_s=5):
    last_val = ""
    while True:
        current_val = (
            str(target) if isinstance(seriize_fn, type(None)) else seriize_fn(target)
        )
        if current_val != last_val:
            yield current_val
            last_val = current_val
        await asyncio.sleep(update_s)


@app.websocket("/ws/system/agents")
async def ws_agents(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        async for agents in stream_dict(agents):
            await manager.send_json(websocket, agents)
    except WebSocketDisconnect:
        manager.disconnect(websocket)


@app.websocket("/ws/tasks/{task_id}/metrics")
async def ws_metrics(websocket: WebSocket, task_id: str):
    task = await find_task()(task_id)
    if task:
        await manager.connect(websocket)
        try:
            async for items in stream_list(task.metrics, update_s=1):
                metrics: List[Metric] = items
                await manager.send_json(
                    websocket, [metric.model_dump() for metric in metrics]
                )
        except WebSocketDisconnect:
            manager.disconnect(websocket)


@app.websocket("/ws/tasks/{task_id}/navigator_status")
async def ws_navigator_status(websocket: WebSocket, task_id: str):
    task = await find_task()(task_id)
    if task:
        await manager.connect(websocket)
        try:
            async for attr in stream_attribute(task, "navigator_status"):
                await manager.send_json(websocket, attr)
        except WebSocketDisconnect:
            manager.disconnect(websocket)


@app.websocket("/ws/tasks/{task_id}/navigator_logs")
async def ws_navigator_logs(websocket: WebSocket, task_id: str):
    task = await find_task()(task_id)
    if task:
        await manager.connect(websocket)
        try:
            async for items in stream_list(task.navigator_logs):
                logs: List[LogType] = items
                await manager.send_json(websocket, [log.model_dump() for log in logs])
        except WebSocketDisconnect:
            manager.disconnect(websocket)


@app.websocket("/ws/tasks/{task_id}/agent/{agent_id}/status")
async def ws_agent_status(websocket: WebSocket, task_id: str, agent_id: str):
    agent = await find_task_agent()(task_id, agent_id)
    if agent:
        await manager.connect(websocket)
        try:
            async for status in stream_attribute(agent, "status"):
                await manager.send_json(websocket, status)
        except WebSocketDisconnect:
            manager.disconnect(websocket)


@app.websocket("/ws/tasks/{task_id}/agent/{agent_id}/logs")
async def ws_agent_logs(websocket: WebSocket, task_id: str, agent_id: str):
    agent = await find_task_agent()(task_id, agent_id)
    if agent:
        await manager.connect(websocket)
        try:
            async for items in stream_list(agent.logs):
                logs: List[LogType] = items
                await manager.send_json(websocket, [log.model_dump() for log in logs])
        except WebSocketDisconnect:
            manager.disconnect(websocket)


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=1524, reload=True)
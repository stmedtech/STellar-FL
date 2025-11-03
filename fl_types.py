from enum import Enum
from typing import Dict, List, Union
from pydantic import BaseModel

class DeviceStatus(str, Enum):
    TO_BE_CONFIRMED = "TO_BE_CONFIRMED"
    CONFIRMED = "CONFIRMED"
    HEALTHY = "HEALTHY"
    TIMEOUT = "TIMEOUT"
    OFFLINE = "OFFLINE"

class FLArchitecture(str, Enum):
    openfl = 'OpenFL'
    nvflare = 'NVFlare'
    flower = 'Flower'

class TaskNodeStatus(str, Enum):
    PREPARED = "PREPARED"
    STARTED = "STARTED"
    AUDIT = "AUDIT"
    ERROR = "ERROR"
    ENDED = "ENDED"
    CLEANED = "CLEANED"
    UNKNOWN = "UNKNOWN"

class TaskData(BaseModel):
    task_info: dict

class Log(BaseModel):
    log: str
    prefix: str
    level: str
    ts: float

class ErrorLog(BaseModel):
    error: str
    traceback: str

class Metric(BaseModel):
    site: str
    tag: str
    step: int
    value: float

LogType = Union[Log, ErrorLog, List[Metric], dict]

class TaskAgent(BaseModel):
    status: TaskNodeStatus
    logs: List[LogType]

class TaskEntry(BaseModel):
    id: str
    type: FLArchitecture
    status: TaskNodeStatus

class TaskStage(str, Enum):
    ERROR = "ERROR"
    PREPARED = "PREPARED"
    STARTED = "STARTED"
    ENDED = "ENDED"

class TaskTopForm(BaseModel):
    agents: List[str]

class Task(BaseModel):
    uuid: str
    navigator_id: str
    navigator_status: TaskNodeStatus
    navigator_data: TaskData
    navigator_logs: List[LogType]
    agents: Dict[str, TaskAgent]
    metrics: List[Metric]

class TaskCreationForm(BaseModel):
    agents: List[str]
    task_info: dict
    
    @property
    def reference_token(self) -> str:
        return self.task_info.get("reference_token")
    
    @property
    def experiment_name(self) -> str:
        return self.task_info.get("experiment_name")
    
    @property
    def model_name(self) -> str:
        return self.task_info.get("model_name")
    
    @property
    def epoch(self) -> int:
        return self.task_info.get("epoch")
    
    @property
    def server_host(self) -> str:
        server_host = self.task_info.get("server_host")
        director_host = self.task_info.get("director_host")
        return server_host or director_host or "localhost"
    
    @property
    def server_ports(self) -> List[int]:
        server_port = self.task_info.get("server_port")
        admin_port = self.task_info.get("admin_port")
        admin_experiment_port = self.task_info.get("admin_experiment_port")
        secure_port = self.task_info.get("secure_port")
        insecure_port = self.task_info.get("insecure_port")
        server_ports = [port for port in [server_port, admin_port, admin_experiment_port, secure_port, insecure_port] if port is not None]
        server_ports = [int(port) for port in server_ports]
        return server_ports

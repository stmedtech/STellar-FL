from enum import Enum
from typing import Dict, List, Union
from pydantic import BaseModel, Field

from stefan_fl.core.interfaces.fl_config import BaseFLConfig

class DeviceStatus(str, Enum):
    TO_BE_CONFIRMED = "TO_BE_CONFIRMED"
    CONFIRMED = "CONFIRMED"
    HEALTHY = "HEALTHY"
    TIMEOUT = "TIMEOUT"
    OFFLINE = "OFFLINE"

class FLArchitecture(str, Enum):
    dummy = 'Dummy'
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
    status: TaskNodeStatus = Field(default=TaskNodeStatus.PREPARED)
    logs: List[LogType] = Field(default_factory=list)

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

class SingleTask(BaseModel):
    uuid: str
    device_id: str
    status: TaskNodeStatus
    data: TaskData
    logs: List[LogType]
    metrics: List[Metric]
    distributed_tasks: Dict[str, str] = Field(default_factory=dict)  # Maps agent_id -> agent_task_uuid

class Task(BaseModel):
    uuid: str
    navigator_id: str
    navigator_status: TaskNodeStatus
    navigator_data: TaskData
    navigator_logs: List[LogType]
    agents: Dict[str, TaskAgent]
    metrics: List[Metric]
    
    @property
    def form(self) -> 'TaskCreationForm':
        return TaskCreationForm(
            agents=list(self.agents.keys()),
            task_info=self.navigator_data.task_info,
        )

class TaskCreationForm(BaseModel):
    agents: List[str]
    task_info: dict
    
    @property
    def architecture(self) -> FLArchitecture:
        return FLArchitecture(self.task_info.get("architecture", FLArchitecture.nvflare.value))
    
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
    def rounds(self) -> int:
        return self.task_info.get("rounds") or self.task_info.get("epoch") or 10
    
    @property
    def local_epochs(self) -> int:
        return self.task_info.get("local_epochs", 1)
    
    @property
    def batch_size(self) -> int:
        return self.task_info.get("batch_size", 8)
    
    @property
    def learning_rate(self) -> float:
        return self.task_info.get("learning_rate", 0.01)
    
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
        for i in range(len(server_ports), 3):
            server_ports.append(8080 + i)
        return server_ports
    
    @property
    def fl_config(self) -> BaseFLConfig:
        return BaseFLConfig(
            framework=self.architecture.value.lower(),
            clients=self.agents,
            rounds=self.rounds,
            local_epochs=self.local_epochs,
            learning_rate=self.learning_rate,
            batch_size=self.batch_size,
            server_host=self.server_host,
            server_ports=self.server_ports,
        )

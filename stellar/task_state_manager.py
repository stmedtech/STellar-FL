"""
Task State Manager using Redis with Pydantic Task objects.

Provides thread-safe Redis-based storage for Task objects with serialization/deserialization.
"""
import json
import logging
import time
from typing import Optional, Dict, List
import redis
from pydantic import ValidationError

from fl_types import Task, SingleTask, TaskNodeStatus, TaskAgent, TaskData, Log, Metric, DeviceStatus

logger = logging.getLogger(__name__)

# Redis key prefix for task states
REDIS_TASK_KEY_PREFIX = "stellar:task:"
REDIS_TASK_INDEX_KEY = "stellar:tasks:index"


class TaskStateManager:
    """Manages task state storage in Redis using Pydantic Task objects."""
    
    def __init__(self, redis_url: str = "redis://localhost:6379/0", redis_client=None):
        """
        Initialize the task state manager.
        
        Args:
            redis_url: Redis connection URL
            redis_client: Optional existing Redis client
        """
        if redis_client is not None:
            self.redis_client = redis_client
        else:
            self.redis_client = redis.from_url(redis_url, decode_responses=True)
        
        # Test connection
        try:
            self.redis_client.ping()
            logger.info(f"Connected to Redis at {redis_url}")
        except redis.ConnectionError as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise
    
    def _get_task_key(self, task_id: str) -> str:
        """Get Redis key for a task."""
        return f"{REDIS_TASK_KEY_PREFIX}{task_id}"
    
    def get_single_task(self, task_id: str) -> Optional[SingleTask]:
        """
        Get a SingleTask object from Redis.
        
        Args:
            task_id: Task UUID
            
        Returns:
            SingleTask object if found, None otherwise
        """
        key = self._get_task_key(task_id)
        try:
            task_json = self.redis_client.get(key)
            if task_json is None:
                return None
            
            # Deserialize from JSON using Pydantic
            task_dict = json.loads(task_json)
            return SingleTask.model_validate(task_dict)
        except (json.JSONDecodeError, ValidationError) as e:
            logger.error(f"Failed to deserialize task {task_id}: {e}")
            return None
        except Exception as e:
            logger.error(f"Error retrieving task {task_id}: {e}")
            return None
    
    def save_single_task(self, task: SingleTask) -> bool:
        """
        Save a SingleTask object to Redis.
        
        Args:
            task: SingleTask object to save
            
        Returns:
            True if successful, False otherwise
        """
        key = self._get_task_key(task.uuid)
        try:
            # Serialize to JSON using Pydantic model_dump
            task_dict = task.model_dump(mode='json')
            task_json = json.dumps(task_dict)
            
            # Save to Redis
            self.redis_client.set(key, task_json)
            
            # Add to index
            self.redis_client.sadd(REDIS_TASK_INDEX_KEY, task.uuid)
            
            logger.debug(f"SingleTask {task.uuid} state synchronized")
            return True
        except Exception as e:
            logger.error(f"Error saving SingleTask {task.uuid}: {e}")
            return False
    
    def get_task(self, task_id: str) -> Optional[Task]:
        """
        Get a Task object composed from distributed SingleTask objects.
        
        Composes the original Task structure from the navigator SingleTask and its distributed agent SingleTasks.
        This is used for API compatibility - the underlying storage uses distributed SingleTask objects.
        
        Args:
            task_id: Navigator task UUID (main task)
            
        Returns:
            Task object if found, None otherwise
        """
        navigator_task = self.get_single_task(task_id)
        if not navigator_task:
            return None
        
        # Compose Task from SingleTask and distributed tasks
        agents = {}
        all_metrics = list(navigator_task.metrics)
        
        # Load all agent tasks
        for agent_id, agent_task_uuid in navigator_task.distributed_tasks.items():
            agent_task = self.get_single_task(agent_task_uuid)
            if agent_task:
                agents[agent_id] = TaskAgent(
                    status=agent_task.status,
                    logs=agent_task.logs
                )
                # Aggregate metrics from agent tasks
                all_metrics.extend(agent_task.metrics)
        
        # Compose into Task structure
        return Task(
            uuid=navigator_task.uuid,
            navigator_id=navigator_task.device_id,
            navigator_status=navigator_task.status,
            navigator_data=navigator_task.data,
            navigator_logs=navigator_task.logs,
            agents=agents,
            metrics=all_metrics
        )
    
    def update_task_status(self, task_id: str, status: TaskNodeStatus) -> Optional[SingleTask]:
        """Update SingleTask status."""
        task = self.get_single_task(task_id)
        if task is None:
            return None
        task.status = status
        if self.save_single_task(task):
            return task
        return None
    
    def add_task_log(self, task_id: str, log: Log, agent_id: Optional[str] = None) -> bool:
        """
        Add a log entry to a SingleTask.
        
        Args:
            task_id: Task UUID (navigator or agent task UUID)
            log: Log object to add
            agent_id: Optional agent ID (if None, adds to navigator task, otherwise finds agent task)
            
        Returns:
            True if successful
        """
        if agent_id is None:
            # Add to navigator task
            task = self.get_single_task(task_id)
            if task is None:
                return False
            task.logs.append(log)
            return self.save_single_task(task)
        else:
            # Find agent task via navigator's distributed_tasks
            navigator_task = self.get_single_task(task_id)
            if navigator_task is None or agent_id not in navigator_task.distributed_tasks:
                return False
            agent_task_uuid = navigator_task.distributed_tasks[agent_id]
            agent_task = self.get_single_task(agent_task_uuid)
            if agent_task is None:
                return False
            agent_task.logs.append(log)
            return self.save_single_task(agent_task)
    
    def add_task_metric(self, task_id: str, metric: Metric) -> bool:
        """Add a metric to a SingleTask."""
        task = self.get_single_task(task_id)
        if task is None:
            return False
        
        task.metrics.append(metric)
        return self.save_single_task(task)
    
    def update_agent_status(self, task_id: str, agent_id: str, status: TaskNodeStatus) -> bool:
        """Update an agent's status in their SingleTask."""
        # Find agent task via navigator's distributed_tasks
        navigator_task = self.get_single_task(task_id)
        if navigator_task is None or agent_id not in navigator_task.distributed_tasks:
            return False
        agent_task_uuid = navigator_task.distributed_tasks[agent_id]
        agent_task = self.get_single_task(agent_task_uuid)
        if agent_task is None:
            return False
        agent_task.status = status
        return self.save_single_task(agent_task)
    
    def delete_task(self, task_id: str) -> bool:
        """Delete a task from Redis."""
        key = self._get_task_key(task_id)
        try:
            self.redis_client.delete(key)
            self.redis_client.srem(REDIS_TASK_INDEX_KEY, task_id)
            return True
        except Exception as e:
            logger.error(f"Error deleting task {task_id}: {e}")
            return False
    
    def list_task_ids(self) -> List[str]:
        """List all task IDs."""
        try:
            return list(self.redis_client.smembers(REDIS_TASK_INDEX_KEY))
        except Exception as e:
            logger.error(f"Error listing tasks: {e}")
            return []
    
    def exists(self, task_id: str) -> bool:
        """Check if a task exists."""
        key = self._get_task_key(task_id)
        return self.redis_client.exists(key) > 0
    
    @property
    def id(self) -> str:
        devices = self.get_devices()
        if len(devices):
            return list(devices.keys())[0]
    
    # ========== Device Management Methods ==========
    
    def get_devices(self) -> Dict[str, dict]:
        """
        Get all devices from Redis.
        
        Returns:
            Dictionary of device_id -> device_info
        """
        from celery_tasks import get_clients
        try:
            devices_dict = {}
            
            for client_name, reference_token in get_clients().items():
                try:
                    devices_dict[client_name] = {
                        "id": client_name,
                        "reference_token": reference_token,
                        "status": DeviceStatus.HEALTHY.value,
                        "last_ts": time.time(),
                        "sys_info": {
                            "system": "system",
                            "release": "release",
                            "version": "version",
                            "machine": "machine",
                            "processor": "processor",
                            "in_docker": False,
                            "security_information": {"public_key": "public_key",},
                            "CPU_information": {"physical_cores": -1, "total_cores": -1,},
                            "RAM_information": {"total": ""},
                            "GPU_information": []
                        }
                        # "sys_info": {
                        #     "system": "linux",
                        #     "release": "22.04",
                        #     "version": "5.15.0-100-generic",
                        #     "machine": "x86_64",
                        #     "processor": "Intel(R) Core(TM) i7-10750H CPU @ 2.60GHz",
                        #     "in_docker": False,
                        #     "security_information": {
                        #         "public_key": "public_key",
                        #     },
                        #     "CPU_information": {
                        #         "physical_cores": 10,
                        #         "total_cores": 10,
                        #     },
                        #     "RAM_information": {
                        #         "total": "64GB",
                        #     },
                        #     "GPU_information": [
                        #         {
                        #             "ID": 0,
                        #             "UUID": "uuid_0",
                        #             "name": "NVIDIA GeForce RTX 3090",
                        #             "memory": "12GB",
                        #         }
                        #     ]
                        # }
                    }
                except Exception as e:
                    logger.error(f"Failed to get device {client_name}: {e}")
                    continue
            
            return devices_dict
        except (json.JSONDecodeError, Exception) as e:
            logger.error(f"Failed to get devices from Redis: {e}")
            return {}
    
    def get_device(self, device_id: str) -> Optional[dict]:
        """
        Get a single device by ID.
        
        Args:
            device_id: Device identifier
            
        Returns:
            Device info dict if found, None otherwise
        """
        devices = self.get_devices()
        return devices.get(device_id)


# Global instance (will be initialized in app startup)
_task_state_manager: Optional[TaskStateManager] = None


def get_task_state_manager(redis_url: Optional[str] = None) -> TaskStateManager:
    """Get or create the global task state manager instance."""
    global _task_state_manager
    if _task_state_manager is None:
        _task_state_manager = TaskStateManager(redis_url=redis_url)
    return _task_state_manager


def initialize_task_state_manager(redis_url: Optional[str] = None, redis_client=None):
    """Initialize the global task state manager."""
    global _task_state_manager
    _task_state_manager = TaskStateManager(redis_url=redis_url, redis_client=redis_client)
    return _task_state_manager


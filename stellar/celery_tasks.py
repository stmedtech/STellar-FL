"""
Celery tasks for executing federated learning tasks.

Tasks update Task objects in Redis via TaskStateManager for real-time streaming.
"""
import base64
from contextlib import contextmanager
import concurrent.futures
import inspect
import os
from pathlib import Path, PureWindowsPath
import shutil
import socket
import sys
import tempfile
import threading
import time
import logging
import traceback
import random
import time
import re
from typing import Dict, List
import uuid
import zipfile
from celery import Task as CeleryTask

import stellar_client
from stellar_client.resources.compute import ComputeRun
from stellar_client.exceptions import ComputeError

from celery_app import celery_app
from fl_types import FLArchitecture, SingleTask, TaskNodeStatus, Log, Metric, TaskData
from utils import MODELS_DIR, install_requirements, retrive_model
from task_state_manager import TaskStateManager, get_task_state_manager
from llm import call_cline, locate_model_codebase, save_model_codebase

from stefan_fl.cli import import_object
from stefan_fl.core.interfaces.framework_adapter import FLTrainingMetrics, FLTrainingMode
from stefan_fl.core.interfaces.framework_factory import FLFrameworkFactory

logger = logging.getLogger(__name__)

# Initialize stellar client (lazy initialization)
_stellar_client = None

def get_stellar_client():
    """Get or create stellar client instance."""
    global _stellar_client
    if _stellar_client is None:
        STELLAR_NODE_URL = os.getenv('STELLAR_NODE_URL')
        _stellar_client = stellar_client.from_env()
    return _stellar_client

LLM_MODEL_DIR = Path(os.getenv('LLM_MODEL_DIR', '/app/models'))

def get_server_queue() -> str:
    return 'server'

def get_client_queue(client_id: str) -> str:
    return f'client_{client_id}'

def get_clients() -> Dict[str, str]:
    """Get available Stellar devices (replaces Celery worker discovery).

    Returns:
        Dictionary mapping device_id to reference_token.
    """
    client = get_stellar_client()
    devices = client.devices.list(include_self=False)

    # Return device_id -> reference_token mapping using device.attrs['ReferenceToken']
    clients = {}
    for device in devices:
        device_id = device.id
        reference_token = device.attrs.get('ReferenceToken')
        clients[device_id] = reference_token

    return clients

# Custom exception classes for Stellar operations
class PrepareFLClientError(ComputeError):
    """Error during FL client preparation."""
    pass

class ClientExecutionError(ComputeError):
    """Error during FL client execution."""
    pass

class ClientCleanupError(ComputeError):
    """Error during FL client cleanup."""
    pass

# Helper functions for Stellar compute operations
def is_execution_success(run: ComputeRun) -> bool:
    """Check if compute run was successful."""
    return run.is_terminal and not run.is_running and run.exit_code == 0

@contextmanager
def follow_execution(device_id: str, command: str, args: List[str], state_manager: TaskStateManager = None, agent_task_uuid: str = None):
    """Context manager for executing commands and streaming logs.
    
    Args:
        device_id: Device ID to execute on
        command: Command to execute
        args: Command arguments
        state_manager: Optional task state manager for log updates
        agent_task_uuid: Optional agent task UUID for log updates
    """
    client = get_stellar_client()
    with client.compute.run(device_id, command, args) as run:
        # Stream logs and update state if state_manager provided
        if state_manager and agent_task_uuid:
            for line_data in run.stream_output():
                log_type = line_data.get("type", "stdout")
                log_data = line_data.get("data", "")
                log_level = "error" if log_type == "stderr" else "info"
                
                logger.info(f'[{device_id}][{run.id}][{log_type}] {log_data}')
                
                # Update agent task logs
                agent_task = state_manager.get_single_task(agent_task_uuid)
                if agent_task:
                    log_entry = Log(
                        log=log_data,
                        prefix=device_id,
                        level=log_level,
                        ts=time.time()
                    )
                    agent_task.logs.append(log_entry)
                    state_manager.save_single_task(agent_task)
        else:
            # Just stream without updating state
            for line_data in run.stream_output():
                logger.info(f'[{device_id}][{run.id}][{line_data.get("type", "stdout")}] {line_data.get("data", "")}')
        
        yield run

def stream_execution(device_id: str, command: str, args: List[str]) -> bool:
    """Execute command and return success status."""
    with follow_execution(device_id, command, args) as run:
        return is_execution_success(run)

def get_data_dir(device_id: str) -> Path:
    """Get data directory path from device config."""
    client = get_stellar_client()
    run = client.compute.run(device_id, 'stellar', ['config'])
    run.wait()
    result = ''.join([line.get("data", "") for line in run.stream_output()])
    run.reload()
    run.remove()
    if run.exit_code != 0:
        message = 'Failed to get config'
        logger.error(f'[{device_id}] {message}')
        raise PrepareFLClientError(message)
    match = re.search(r'DataDir:([^\s\}]+)', result)
    if not match:
        message = f"Could not extract DataDir from config: {result}"
        logger.error(f'[{device_id}] {message}')
        raise PrepareFLClientError(message)
    return Path(match.group(1))

def ping_device(device_id: str):
    """Ping a device to check connectivity."""
    client = get_stellar_client()
    device = client.devices.get(device_id)
    logger.info(f'[{device_id}] Ping: {device.ping()}')

def client_pool_execution(fn, list_of_args: List[tuple]):
    """Execute function in parallel across multiple clients using ThreadPoolExecutor.
    
    Args:
        fn: Function to execute
        list_of_args: List of tuples containing arguments for each execution
        
    Returns:
        Dictionary mapping first argument (device_id) to result
    """
    def _fn(*args):
        try:
            return fn(*args)
        except Exception as e:
            logger.error(f"[{args[0]}] Error: {e}")
            logger.error(traceback.format_exc())
            return e
    
    client_results = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        # Submit tasks and get future objects
        futures = [executor.submit(_fn, *args) for args in list_of_args]
        client_results = {args[0]: future.result() for args, future in zip(list_of_args, futures)}
    return client_results

def pool_execution_error_handler(client_results: dict):
    """Handle errors from pool execution results.
    
    Args:
        client_results: Dictionary mapping device_id to result
        
    Raises:
        Exception: The first exception found in results
    """
    e = None
    for device_id, result in client_results.items():
        if isinstance(result, Exception):
            logger.error(f"[{device_id}] Error: {result}")
            logger.error(traceback.format_exc())
            e = result
    
    if e is not None:
        raise e

def prepare_fl_device(device_id: str, env_name: str, option: str):
    """Prepare FL client environment on remote device.
    
    Args:
        device_id: Device ID to prepare
        env_name: Conda environment name
        option: Framework option for stefan_fl installation
        
    Raises:
        PrepareFLClientError: If preparation fails
    """
    STEFAN_WHL_DIR = Path(os.getenv('STEFAN_WHL_DIR'))
    PYTHON_VERSION = '3.11'
    
    assert STEFAN_WHL_DIR.exists(), 'STEFAN_WHL_DIR does not exist'
    assert STEFAN_WHL_DIR.is_dir(), 'STEFAN_WHL_DIR is not a directory'
    stefan_fl_whl_path = list(STEFAN_WHL_DIR.glob('*.whl'))[0]
    assert stefan_fl_whl_path.exists(), 'stefan_fl_whl_path does not exist'
    assert stefan_fl_whl_path.is_file(), 'stefan_fl_whl_path is not a file'
    
    prefix = f'[{device_id}] '
    logger.info(f'{prefix}Preparing FL device with option {option}...')
    
    logger.info(prefix + 'Checking conda installation...')
    if stream_execution(device_id, 'stellar', ['conda', 'path']):
        logger.info(prefix + 'Conda installed')
    else:
        logger.warning(prefix + 'Conda not found, installing...')
        if stream_execution(device_id, 'stellar', ['conda', 'install-conda']):
            logger.info(prefix + 'Conda installed')
        else:
            message = 'Failed to install conda'
            logger.error(prefix + message)
            raise PrepareFLClientError(message)
    
    logger.info(prefix + f'Checking environment {env_name}...')
    if stream_execution(device_id, 'stellar', ['conda', 'get', env_name]):
        logger.info(prefix + 'Environment found')
    else:
        logger.warning(prefix + 'Environment not found, creating...')
        if stream_execution(device_id, 'stellar', ['conda', 'create', env_name, '--python', PYTHON_VERSION]):
            logger.info(prefix + 'Environment created')
        else:
            message = 'Failed to create environment'
            logger.error(prefix + message)
            raise PrepareFLClientError(message)
    
    logger.info(prefix + 'Checking stellar-client installation...')
    if stream_execution(device_id, 'stellar', ['conda', 'run', 'run', '-n', env_name, 'pip', 'show', 'stellar-client']):
        logger.info(prefix + 'stellar-client installed')
    else:
        logger.warning(prefix + 'stellar-client not found, installing...')
        if stream_execution(device_id, 'stellar', ['conda', 'install-client', env_name]):
            logger.info(prefix + 'stellar-client installed')
        else:
            message = 'Failed to install stellar-client'
            logger.error(prefix + message)
            raise PrepareFLClientError(message)
    
    logger.info(prefix + 'Updating stefan_fl installation...')
    
    with RemoteTempFile(device_id, env_name, str(stefan_fl_whl_path), random_suffix=False) as temp_file:
        if stream_execution(device_id, 'stellar', ['conda', 'run', 'run', '-n', env_name, '--no-capture-output', 'pip', 'install', f'{temp_file}[{option}]']):
            logger.info(prefix + 'stefan_fl installed')
        else:
            if stream_execution(device_id, 'stellar', ['conda', 'run', 'run', '-n', env_name, '--no-capture-output', 'pip', 'show', 'stefan_fl']):
                logger.info(prefix + 'stefan_fl already installed')
            else:
                message = 'Failed to check if stefan_fl is installed'
                logger.error(prefix + message)
                raise PrepareFLClientError(message)
    
    logger.info(prefix + 'FL client prepared')
    
    return True

def check_ports_availability(device_id: str, env_name: str, ports: List[int]):
    """Check if ports are available on remote device.
    
    Args:
        device_id: Device ID to check ports on
        env_name: Conda environment name
        ports: List of port numbers to check
        
    Raises:
        PrepareFLClientError: If any port is not available
    """
    prefix = f'[{device_id}]'
    logger.info(prefix + 'Checking ports availability...')
    for port in ports:
        if stream_execution(device_id, 'stellar',
            [
                'conda', 'run', 'run', '-n', env_name, 'python', '-c',
                f'import socket, sys; s = socket.socket(socket.AF_INET, socket.SOCK_STREAM); s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1); s.bind(("0.0.0.0", {port}))',
            ]
        ):
            logger.info(prefix + f'Port {port} is available')
        else:
            message = f'Port {port} is not available'
            logger.error(prefix + message)
            raise PrepareFLClientError(message)
    
    logger.info(prefix + 'Ports available')
    
    return True

class RemoteTempFile:
    """Context manager for uploading and managing temporary files on remote devices."""
    
    def __init__(self, device_id: str, env_name: str, file_name: str, random_suffix: bool = True):
        self.device_id = device_id
        self.env_name = env_name
        self.file_name = file_name
        
        file_path = Path(file_name)
        if random_suffix:
            self.temp_file_name = f'{file_path.stem}_{uuid.uuid4().hex}{file_path.suffix}'
        else:
            self.temp_file_name = file_path.name
        self.remote_path = None
    
    def __enter__(self):
        client = get_stellar_client()
        data_dir = get_data_dir(self.device_id)
        if PureWindowsPath(data_dir).is_absolute():
            self.remote_path = str(PureWindowsPath(data_dir, self.temp_file_name))
        else:
            self.remote_path = str(data_dir / self.temp_file_name)
        
        device = client.devices.get(self.device_id)
        uploaded = False
        retries = 0
        while not uploaded and retries < 10:
            try:
                device.files().upload(self.file_name, self.temp_file_name)
                if stream_execution(self.device_id, 'stellar', ['conda', 'run', 'run', '-n', self.env_name, 'python', '-c', f'import os, sys, zipfile; sys.exit(0 if os.path.exists("{self.remote_path}") and zipfile.is_zipfile("{self.remote_path}") else 1)']):
                    uploaded = True
                    break
                else:
                    raise PrepareFLClientError('Failed to check if file is uploaded')
            except Exception as e:
                logger.warning(f'[{self.device_id}] Failed to upload file {self.file_name}: {e}, retrying...')
                time.sleep(1)
        
        if not uploaded:
            message = 'Failed to upload file after 10 retries'
            logger.error(f'[{self.device_id}] {message}')
            raise PrepareFLClientError(message)
        
        logger.info(f'[{self.device_id}] File {self.file_name} uploaded to {self.remote_path}')
        return self.remote_path
    
    def __exit__(self, exc_type, exc_value, traceback):
        if self.remote_path:
            if stream_execution(self.device_id, 'stellar',
                [
                    'conda',
                    'run',
                    'run',
                    '-n',
                    self.env_name,
                    'python',
                    '-c',
                    f'import os; os.remove(r"{self.remote_path}")',
                ]
            ):
                logger.info(f'[{self.device_id}] File {self.remote_path} removed')
            else:
                message = f'Failed to remove file {self.remote_path}'
                logger.error(f'[{self.device_id}] {message}')
                # Don't raise exception on cleanup failure, just log it
                logger.warning(f'[{self.device_id}] Cleanup warning: {message}')

class RemotePorts:
    """Context manager for managing port forwarding on remote devices."""
    
    def __init__(self, device_id: str, env_name: str, server_host: str, ports: List[int], my_id: str):
        self.device_id = device_id
        self.env_name = env_name
        self.ports = ports
        self.server_host = server_host
        self.my_id = my_id
    
    def __enter__(self):
        client = get_stellar_client()
        for port in self.ports:
            # Create proxy from remote device to server
            try:
                # Execute on remote device to create proxy connection
                if stream_execution(self.device_id, 'stellar',
                    [
                        'conda',
                        'run',
                        'run',
                        '-n',
                        self.env_name,
                        'python',
                        '-c',
                        f'import stellar_client;client = stellar_client.from_env();client.proxy.create("{self.my_id}", {port}, "{self.server_host}", {port})',
                    ]
                ):
                    logger.info(f'[{self.device_id}] Port {port} proxy created')
                else:
                    message = f'Port {port} proxy not created'
                    logger.error(f'[{self.device_id}] {message}')
                    raise ClientExecutionError(message)
            except Exception as e:
                logger.error(f'[{self.device_id}] Failed to create proxy for port {port}: {e}')
                raise ClientExecutionError(f'Failed to create proxy for port {port}: {e}')
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        for port in self.ports:
            try:
                # Close proxy on remote device
                if stream_execution(self.device_id, 'stellar',
                    [
                        'conda',
                        'run',
                        'run',
                        '-n',
                        self.env_name,
                        'python',
                        '-c',
                        f'import stellar_client;client = stellar_client.from_env();client.proxy.get({port}).close()',
                    ]
                ):
                    logger.info(f'[{self.device_id}] Port {port} proxy closed')
                else:
                    message = f'Port {port} proxy not closed'
                    logger.warning(f'[{self.device_id}] {message}')
                    # Don't raise exception on cleanup failure, just log it
            except Exception as e:
                logger.warning(f'[{self.device_id}] Failed to close proxy for port {port}: {e}')

class TempProxyWhitelist:
    """Context manager for temporarily adding device to policy whitelist."""
    
    def __init__(self, device_id: str):
        """Initialize TempProxyWhitelist.
        
        Args:
            device_id: Device ID to add/remove from whitelist
        """
        self.device_id = device_id
    
    def __enter__(self):
        """Add device to whitelist if policy is enabled."""
        client = get_stellar_client()
        try:
            policy = client.policy.get()
            if policy.enable:
                client.policy.add_to_whitelist(self.device_id)
                logger.info(f'[{self.device_id}] Added to whitelist')
            else:
                logger.warning('Policy is not enabled')
        except Exception as e:
            logger.warning(f'Failed to add {self.device_id} to whitelist: {e}')
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        """Remove device from whitelist if policy is enabled."""
        client = get_stellar_client()
        try:
            policy = client.policy.get()
            if policy.enable:
                client.policy.remove_from_whitelist(self.device_id)
                logger.info(f'[{self.device_id}] Removed from whitelist')
            else:
                logger.warning('Policy is not enabled')
        except Exception as e:
            logger.warning(f'Failed to remove {self.device_id} from whitelist: {e}')

class CallbackTask(CeleryTask):
    """Custom task class that provides state updates during execution"""
    
    def on_success(self, retval, task_id, args, kwargs):
        """Called on successful task completion"""
        task_uuid = args[0] if args else kwargs.get("task_id")
        if task_uuid:
            state_manager = get_task_state_manager()
            navigator_task = state_manager.get_single_task(task_uuid)
            if navigator_task:
                navigator_task.status = TaskNodeStatus.ENDED
                state_manager.save_single_task(navigator_task)
            logger.info(f"Task {task_uuid} completed successfully")
    
    def on_failure(self, exc, task_id, args, kwargs, einfo):
        """Called on task failure"""
        task_uuid = args[0] if args else kwargs.get("task_id")
        if task_uuid:
            state_manager = get_task_state_manager()
            navigator_task = state_manager.get_single_task(task_uuid)
            if navigator_task:
                navigator_task.status = TaskNodeStatus.ERROR
                # Add error log
                error_log = Log(
                    log=str(exc),
                    prefix="navigator",
                    level="error",
                    ts=time.time()
                )
                navigator_task.logs.append(error_log)
                state_manager.save_single_task(navigator_task)
            logger.error(f"Task {task_uuid} failed: {exc}")

@celery_app.task(name="celery_tasks.generate_model_codebase")
def generate_model_codebase(model_name: str, prompt: str):
    assert not (Path(MODELS_DIR) / model_name).exists(), f"❌ Model {model_name} already exists"
    
    logger.info(f"Generating model codebase for model: {model_name}")
    
    mount_path = Path(LLM_MODEL_DIR) / model_name / str(uuid.uuid4())
    mount_path.mkdir(parents=True, exist_ok=True)
    local_mount_path = Path(MODELS_DIR) / model_name / mount_path.name
    local_mount_path.mkdir(parents=True, exist_ok=True)
    
    call_cline(prompt, str(mount_path.absolute()))
    zip_file = locate_model_codebase(str(local_mount_path.absolute()))
    assert zip_file is not None, f"❌ Zip file not found"
    assert zip_file.exists(), f"❌ Zip file does not exist"
    assert zip_file.is_file(), f"❌ Zip file is not a file"
    assert zip_file.suffix == '.zip', f"❌ Zip file is not a zip file"
    logger.info(f"LLM generated zip file: {zip_file}")
    
    save_dir = save_model_codebase(model_name, zip_file, MODELS_DIR)
    logger.info(f"Saved model codebase to: {save_dir}")
    
    shutil.rmtree(local_mount_path)
    
    return save_dir

@contextmanager
def temp_codebase_dir():
    with tempfile.TemporaryDirectory(prefix="stefan_fl_codebase_") as tmp_dir:
        codebase_dir = Path(tmp_dir)
        sys.path.append(tmp_dir)
        yield codebase_dir
        sys.path.remove(tmp_dir)

def find_free_ports(num_ports, start: int = None, end: int = None):
    free_ports = []
    retries = 0
    if start is not None and end is not None:
        while len(free_ports) < num_ports:
            port = random.randint(start, end)
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('localhost', port))
                free_ports.append(port)
            retries += 1
            if retries > 100:
                raise Exception(f"❌ Failed to find {num_ports} free ports")
    else:
        while len(free_ports) < num_ports:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('localhost', 0))  # Bind to a free port provided by the OS
                port = s.getsockname()[1]  # Get the assigned port number
                free_ports.append(port)
            retries += 1
            if retries > 100:
                raise Exception(f"❌ Failed to find {num_ports} free ports")
    return free_ports

def dummy_training_process(navigator_task: SingleTask):
    logger.info(f"Starting dummy training process for task: {navigator_task.uuid}")
    
    state_manager = get_task_state_manager()
    task_id = navigator_task.uuid
    
    # Get task info from navigator task
    task_info = navigator_task.data.task_info.copy()
    task_info["architecture"] = FLArchitecture.flower.value  # Override for dummy
    
    # Create TaskCreationForm for config
    from fl_types import TaskCreationForm
    form = TaskCreationForm(
        agents=list(navigator_task.distributed_tasks.keys()),
        task_info=task_info
    )
    
    config = form.fl_config
    sites = form.agents
    
    # Simulate the starting of the navigator task
    navigator_task.status = TaskNodeStatus.STARTED
    start_log = Log(log=f"Task {task_id} started (mocked)", prefix="navigator", level="info", ts=time.time())
    navigator_task.logs.append(start_log)
    state_manager.save_single_task(navigator_task)
    
    # Update agent tasks
    for site in sites:
        if site in navigator_task.distributed_tasks:
            agent_task_uuid = navigator_task.distributed_tasks[site]
            agent_task = state_manager.get_single_task(agent_task_uuid)
            if agent_task:
                agent_task.status = TaskNodeStatus.STARTED
                agent_start_log = Log(log=f"Agent {site} started", prefix=site, level="info", ts=time.time())
                agent_task.logs.append(agent_start_log)
                state_manager.save_single_task(agent_task)

    # Prepare dummy metrics structure for collection
    class DummyMetrics:
        def __init__(self):
            self.history = {}
            self.last_metrics_i = {}

    # Simulate FL metrics generation and collect using metrics_update_fn
    def metrics_update_fn(metrics: FLTrainingMetrics):
        if not hasattr(metrics, 'last_metrics_i') or metrics.last_metrics_i is None:
            metrics.last_metrics_i = {}

        updated_count = 0
        for site, metric in metrics.history.items():
            if site not in metrics.last_metrics_i:
                metrics.last_metrics_i[site] = {}

            for tag, values in metric.items():
                if tag not in metrics.last_metrics_i[site]:
                    metrics.last_metrics_i[site][tag] = -1

                for i, value in enumerate(values):
                    if i > metrics.last_metrics_i[site][tag]:
                        metrics.last_metrics_i[site][tag] = i
                        # Add metric to navigator task
                        navigator_task.metrics.append(
                            Metric(
                                site=site,
                                tag=tag,
                                step=i,
                                value=value,
                            )
                        )
                        updated_count += 1
        logger.info(f"Task metrics: {updated_count} metrics updated")
        state_manager.save_single_task(navigator_task)

    dummy_metrics = DummyMetrics()
    for site in sites:
        dummy_metrics.history[site] = {"loss": [], "accuracy": []}

    # Mocked training process - generate metrics and log updates through metrics_update_fn
    for i in range(form.rounds):
        round_idx = i + 1
        for site in sites:
            # Mock metric values
            loss = round(random.uniform(0.5, 1.5) / (round_idx), 4)
            accuracy = round(random.uniform(0.7, 0.99) * (round_idx / 3), 4)
            # Add to dummy metrics history
            dummy_metrics.history[site]["loss"].append(loss)
            dummy_metrics.history[site]["accuracy"].append(accuracy)
            # Log info to agent task
            if site in navigator_task.distributed_tasks:
                agent_task_uuid = navigator_task.distributed_tasks[site]
                agent_task = state_manager.get_single_task(agent_task_uuid)
                if agent_task:
                    metric_log = Log(
                        log=f"Round {round_idx}: loss={loss}, accuracy={accuracy}",
                        prefix=site,
                        level="info",
                        ts=time.time()
                    )
                    agent_task.logs.append(metric_log)
                    state_manager.save_single_task(agent_task)
        # Update metrics via the function
        metrics_update_fn(dummy_metrics)
        time.sleep(0.5)  # simulate computation delay

    # Simulate random info and warnings
    info_log = Log(
        log="Mocked info: data loader initialized.",
        prefix="navigator",
        level="info",
        ts=time.time()
    )
    warn_log = Log(
        log="Mocked warning: slow network detected.",
        prefix="navigator",
        level="warning",
        ts=time.time()
    )
    navigator_task.logs.extend([info_log, warn_log])
    state_manager.save_single_task(navigator_task)

    # Simulate finishing of the task
    navigator_task.status = TaskNodeStatus.ENDED
    end_log = Log(log=f"Task {task_id} completed", prefix="navigator", level="info", ts=time.time())
    navigator_task.logs.append(end_log)
    state_manager.save_single_task(navigator_task)
    
    for site in sites:
        if site in navigator_task.distributed_tasks:
            agent_task_uuid = navigator_task.distributed_tasks[site]
            agent_task = state_manager.get_single_task(agent_task_uuid)
            if agent_task:
                agent_task.status = TaskNodeStatus.ENDED
                agent_end_log = Log(log=f"Agent {site} completed", prefix=site, level="info", ts=time.time())
                agent_task.logs.append(agent_end_log)
                state_manager.save_single_task(agent_task)

    # Gather metrics into output form
    metrics_output = []
    for m in navigator_task.metrics:
        metrics_output.append({
            "site": m.site,
            "tag": m.tag,
            "step": m.step,
            "value": m.value,
        })
    
    with tempfile.TemporaryDirectory(prefix="stellar_task_output_") as temp_dir:
        temp_dir = Path(temp_dir)
        zip_file_path = Path(MODELS_DIR) / f"{temp_dir.name}.zip"
        shutil.make_archive(str(zip_file_path).replace('.zip', ''), 'zip', temp_dir)
        config.job_output_dir = str(zip_file_path.absolute())

    return {
        "status": "success",
        "framework": config.framework,
        "mode": FLTrainingMode.DISTRIBUTED,
        "metrics": metrics_output,
        "config": config.dict(),
    }

def get_log_file_path(task_id: str) -> Path:
    return MODELS_DIR / f"stellar_task_{task_id}.log"

def execute_task_with_log(func):
    @contextmanager
    def log_to_file(task_id: str):
        # Add a file handler to the root logger
        handler = logging.FileHandler(get_log_file_path(task_id))
        handler.setLevel(logging.INFO)
        handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)'))
        logging.root.addHandler(handler)
        yield
        # Remove the file handler from the root logger
        logging.root.removeHandler(handler)
        
        get_log_file_path(task_id).unlink(missing_ok=True)
    
    def _cb(self, task_id: str):
        with log_to_file(task_id):
            result = func(self, task_id)
            zip_file_path = Path(result['config']['job_output_dir'])
            if zip_file_path.is_file():
                with zipfile.ZipFile(zip_file_path, "a", compression=zipfile.ZIP_DEFLATED) as zip_file:
                    zip_file.write(get_log_file_path(task_id), "full_task.log")
            else:
                with zipfile.ZipFile(zip_file_path, "w", compression=zipfile.ZIP_DEFLATED) as zip_file:
                    zip_file.write(get_log_file_path(task_id), "full_task.log")
            return result
    
    return _cb
            

@celery_app.task(base=CallbackTask, bind=True, name="celery_tasks.execute_task")
@execute_task_with_log
def execute_task(self, task_id: str):
    logger.info(f"Starting task execution for task_id: {task_id}")
    state_manager = get_task_state_manager()
    
    # Get navigator SingleTask
    navigator_task: SingleTask = state_manager.get_single_task(task_id)
    assert navigator_task is not None, f"❌ Navigator task {task_id} not found in Redis"
    
    # Get task info to check architecture
    task_info = navigator_task.data.task_info
    architecture = FLArchitecture(task_info.get("architecture", FLArchitecture.nvflare.value))
    
    if architecture == FLArchitecture.dummy:
        return dummy_training_process(navigator_task)
    
    # Create TaskCreationForm for config
    from fl_types import TaskCreationForm
    form = TaskCreationForm(
        agents=list(navigator_task.distributed_tasks.keys()),
        task_info=task_info
    )
    
    config = form.fl_config
    if os.getenv("SERVER_HOST") is not None:
        config.server_host = os.getenv("SERVER_HOST")
    if os.getenv("SERVER_PORT_RANGE") is not None:
        splits = os.getenv("SERVER_PORT_RANGE").split('-')
        assert len(splits) == 2, f"❌ SERVER_PORT_RANGE must be in the format of 'start-end'"
        start = int(splits[0])
        end = int(splits[1])
        assert start < end, f"❌ SERVER_PORTS start must be less than end"
        config.server_ports = find_free_ports(3, start, end)
    else:
        # TODO: Uncomment this when we have a way to find free ports
        config.server_ports = find_free_ports(3)
    
    logger.info(f"Task FL Config: {config}")
    adapter = FLFrameworkFactory.create_adapter(config)
    
    # Check for custom model in task_info first, otherwise use permanent model
    custom_model_temp_dir = None
    model_info = None
    
    # Try to create custom model from task_info
    from utils import create_custom_model_from_task_info
    
    # Check if custom model data is present in task_info
    has_custom_model_data = bool(task_info.get("custom_model_zip_base64") and task_info.get("custom_model_metadata_json"))
    
    if has_custom_model_data:
        logger.info(f"Custom model data detected in task_info for task {task_id}")
        custom_model_temp_dir = Path(tempfile.gettempdir()) / f"stellar_custom_model_{task_id}"
        if custom_model_temp_dir.exists():
            shutil.rmtree(custom_model_temp_dir)
        custom_model_temp_dir.mkdir(parents=True, exist_ok=True)
        model_info = create_custom_model_from_task_info(task_info, custom_model_temp_dir)
        
        if model_info is None:
            logger.error(f"Failed to create custom model from task_info for task {task_id}")
            raise AssertionError(f"❌ Failed to create custom model from uploaded data. Please check the zip file and metadata.json format.")
        
        logger.info(f"Using custom model from task_info, temp dir: {custom_model_temp_dir}")
    else:
        # No custom model, use permanent model
        logger.info(f"No custom model data in task_info, using permanent model: {form.model_name}")
        model_info = retrive_model(form.model_name)
        if model_info is None:
            raise AssertionError(f"❌ Model {form.model_name} not found in permanent models directory")
    
    assert model_info is not None, f"❌ Model {form.model_name} not found"
    assert "model_zip_file" in model_info, f"❌ Model {form.model_name} model zip file not found"
    model_zip_file = model_info["model_zip_file"]
    assert model_zip_file is not None, f"❌ Model {form.model_name} model zip file not found"
    model_zip_file = Path(model_zip_file)
    assert model_zip_file.exists(), f"❌ Model {form.model_name} model zip file does not exist"
    assert model_zip_file.is_file(), f"❌ Model {form.model_name} model zip file is not a file"
    assert model_zip_file.suffix == '.zip', f"❌ Model {form.model_name} model zip file is not a zip file"
    assert "metadata" in model_info, f"❌ Model {form.model_name} metadata not found"
    model_metadata = model_info["metadata"]
    assert model_metadata is not None, f"❌ Model {form.model_name} metadata not found"
    model_fn_str = model_metadata["model_fn"]
    assert model_fn_str is not None, f"❌ Model function for model {form.model_name} not found"
    dataloader_fn_str = model_metadata["dataloader_fn"]
    assert dataloader_fn_str is not None, f"❌ Dataloader function for model {form.model_name} not found"
    # Update status to STARTED
    navigator_task.status = TaskNodeStatus.STARTED
    start_log = Log(log=f"Task {task_id} started", prefix="navigator", level="info", ts=time.time())
    navigator_task.logs.append(start_log)
    state_manager.save_single_task(navigator_task)
    
    class ClientsStarterFn:
        """Starter function for FL clients using Stellar nodes."""
        
        def __init__(self, navigator_task: SingleTask, state_manager: TaskStateManager, requirements_content: str, config, my_id: str):
            self.navigator_task = navigator_task
            self.state_manager = state_manager
            self.client_threads = {}
            self.requirements_content = requirements_content
            self.config = config
            self.my_id = my_id
            self.env_name = f'stefan-{config.framework}'
        
        def execute_fl_client(self, device_id: str, workspace_zip_file: str):
            """Execute FL client on remote device using stellar_client.
            """
            prefix = f'[{device_id}]'
            logger.info(f'{prefix}Executing FL client...')
            
            # Get agent task UUID
            if device_id not in self.navigator_task.distributed_tasks:
                logger.error(f"Agent {device_id} not found in distributed_tasks")
                return
            agent_task_uuid = self.navigator_task.distributed_tasks[device_id]
            
            try:
                # Upload workspace zip and execute FL client
                with TempProxyWhitelist(device_id):
                    with RemoteTempFile(device_id, self.env_name, workspace_zip_file) as remote_temp_file:
                        with RemotePorts(device_id, self.env_name, self.config.server_host, self.config.server_ports, self.my_id):
                            requirements_content_b64 = base64.b64encode(self.requirements_content.encode()).decode()
                            if not stream_execution(device_id, 'stellar', ['conda', 'run', 'run', '-n', self.env_name, 'python', '-c', f'import subprocess, base64; f = open("requirements.txt", "wb"); f.write(base64.b64decode("{requirements_content_b64}")); f.close(); subprocess.run(["pip", "install", "-r", "requirements.txt"], capture_output=True, text=True, check=True)']):
                                message = 'Failed to install requirements'
                                logger.error(f'{prefix}{message}')
                                raise ClientExecutionError(message)
                            
                            logger.info(f'{prefix}Executing FL join workspace...')
                            # Execute FL client with log streaming
                            with follow_execution(
                                device_id, 
                                'stellar', 
                                ['conda', 'run', 'run', '-n', self.env_name, '--no-capture-output', 'python', '-m', 'stefan_fl', 'join', remote_temp_file],
                                state_manager=self.state_manager,
                                agent_task_uuid=agent_task_uuid
                            ) as run:
                                if is_execution_success(run):
                                    logger.info(f'{prefix}FL client executed successfully')
                                    # Update agent task status
                                    agent_task = self.state_manager.get_single_task(agent_task_uuid)
                                    if agent_task:
                                        agent_task.status = TaskNodeStatus.ENDED
                                        agent_end_log = Log(log=f"Agent {device_id} completed", prefix=device_id, level="info", ts=time.time())
                                        agent_task.logs.append(agent_end_log)
                                        self.state_manager.save_single_task(agent_task)
                                else:
                                    # Execution failed
                                    logs = "".join([f"[{line.get('type', 'stdout')}] {line.get('data', '')}" for line in run.stream_output()])
                                    message = f'Failed to execute FL client [{device_id}]\nExit code: {run.exit_code}\nLogs:\n{logs}'
                                    logger.error(f'{prefix}{message}')
                                    
                                    # Update agent task status
                                    agent_task = self.state_manager.get_single_task(agent_task_uuid)
                                    if agent_task:
                                        agent_task.status = TaskNodeStatus.ERROR
                                        agent_error_log = Log(log=f"Agent {device_id} failed: {message}", prefix=device_id, level="error", ts=time.time())
                                        agent_task.logs.append(agent_error_log)
                                        self.state_manager.save_single_task(agent_task)
                                    
                                    raise ClientExecutionError(message)
            except (ClientExecutionError, ClientCleanupError, ComputeError) as e:
                logger.error(f"Agent {device_id} failed: {e}")
                agent_task = self.state_manager.get_single_task(agent_task_uuid)
                if agent_task:
                    agent_task.status = TaskNodeStatus.ERROR
                    agent_error_log = Log(log=f"Agent {device_id} failed: {str(e)}", prefix=device_id, level="error", ts=time.time())
                    agent_task.logs.append(agent_error_log)
                    agent_error_log = Log(log=traceback.format_exc(), prefix=device_id, level="error", ts=time.time())
                    agent_task.logs.append(agent_error_log)
                    self.state_manager.save_single_task(agent_task)
                raise
            except Exception as e:
                logger.error(f"Agent {device_id} failed with unexpected error: {e}")
                logger.error(traceback.format_exc())
                agent_task = self.state_manager.get_single_task(agent_task_uuid)
                if agent_task:
                    agent_task.status = TaskNodeStatus.ERROR
                    agent_error_log = Log(log=f"Agent {device_id} failed: {str(e)}", prefix=device_id, level="error", ts=time.time())
                    agent_task.logs.append(agent_error_log)
                    agent_error_log = Log(log=traceback.format_exc(), prefix=device_id, level="error", ts=time.time())
                    agent_task.logs.append(agent_error_log)
                    self.state_manager.save_single_task(agent_task)
                raise ClientExecutionError(f"Unexpected error executing FL client on {device_id}: {e}") from e
        
        def start_client(self, client: str, workspace_zip_file: str):
            """Start FL client execution on remote device in a separate thread."""
            clients = get_clients()
            assert client in clients, f"❌ Client {client} not found"
            
            # Get agent task UUID
            if client not in self.navigator_task.distributed_tasks:
                logger.error(f"Agent {client} not found in distributed_tasks")
                return
            agent_task_uuid = self.navigator_task.distributed_tasks[client]
            
            def _client_fn():
                try:
                    # Update agent task status
                    agent_task = self.state_manager.get_single_task(agent_task_uuid)
                    if agent_task:
                        agent_task.status = TaskNodeStatus.STARTED
                        agent_log = Log(log=f"Agent {client} started", prefix=client, level="info", ts=time.time())
                        agent_task.logs.append(agent_log)
                        self.state_manager.save_single_task(agent_task)
                    
                    # Execute FL client
                    self.execute_fl_client(client, workspace_zip_file)
                except Exception as e:
                    logger.error(f"Agent {client} thread failed: {e}")
                    logger.error(traceback.format_exc())
                    agent_task = self.state_manager.get_single_task(agent_task_uuid)
                    if agent_task:
                        agent_task.status = TaskNodeStatus.ERROR
                        agent_error_log = Log(log=f"Agent {client} thread failed: {str(e)}", prefix=client, level="error", ts=time.time())
                        agent_task.logs.append(agent_error_log)
                        agent_error_log = Log(log=traceback.format_exc(), prefix=client, level="error", ts=time.time())
                        agent_task.logs.append(agent_error_log)
                        self.state_manager.save_single_task(agent_task)
            
            thread = threading.Thread(target=_client_fn)
            thread.start()
            self.client_threads[client] = thread
        
        def join_clients(self):
            """Wait for all client threads to complete."""
            for thread in self.client_threads.values():
                thread.join()
            self.client_threads.clear()
        
        def __call__(self, clients_starter: dict):
            """Start FL clients for all devices in clients_starter dict.
            
            Args:
                clients_starter: Dictionary mapping device_id -> workspace_zip_file path
            """
            for client, workspace_zip_file in clients_starter.items():
                # Update agent task
                if client in self.navigator_task.distributed_tasks:
                    agent_task_uuid = self.navigator_task.distributed_tasks[client]
                    agent_task = self.state_manager.get_single_task(agent_task_uuid)
                    if agent_task:
                        agent_task.status = TaskNodeStatus.STARTED
                        agent_log = Log(log=f"Agent {client} started", prefix=client, level="info", ts=time.time())
                        agent_task.logs.append(agent_log)
                        self.state_manager.save_single_task(agent_task)
                        
                        # Also log to navigator
                        navigator_log = Log(log=f"Agent {client} started", prefix=client, level="info", ts=time.time())
                        self.navigator_task.logs.append(navigator_log)
                        self.state_manager.save_single_task(self.navigator_task)
                self.start_client(client, workspace_zip_file)
    
    def metrics_update_fn(metrics: FLTrainingMetrics):
        if not hasattr(metrics, 'last_metrics_i') or metrics.last_metrics_i is None:
            metrics.last_metrics_i = {}
        
        updated_count = 0
        for site, metric in metrics.history.items():
            if site not in metrics.last_metrics_i:
                metrics.last_metrics_i[site] = {}
            
            for tag, values in metric.items():
                if tag not in metrics.last_metrics_i[site]:
                    metrics.last_metrics_i[site][tag] = -1
                
                for i, value in enumerate(values):
                    if i > metrics.last_metrics_i[site][tag]:
                        metrics.last_metrics_i[site][tag] = i
                        task_metric_log = Log(log=f"Metric {tag} from {site} updated with value {value}", prefix="navigator", level="info", ts=time.time())
                        navigator_task.logs.append(task_metric_log)
                        navigator_task.metrics.append(
                            Metric(
                                site=site,
                                tag=tag,
                                step=i,
                                value=value,
                            )
                        )
                        updated_count += 1
        logger.info(f"Task metrics: {updated_count} metrics updated")
        state_manager.save_single_task(navigator_task)
    
    with temp_codebase_dir() as codebase_dir:
        shutil.unpack_archive(model_zip_file, codebase_dir, 'zip')
        requirements_content = ''
        for file in codebase_dir.glob('**/requirements.txt'):
            assert install_requirements(str(file.absolute())), f"❌ Failed to install requirements from {file.absolute()}"
            with open(file, 'r') as f:
                requirements_content += f.read()
        model_fn = import_object(model_fn_str)
        assert inspect.getsource(model_fn), f"❌ Model function {model_fn_str} source file not found"
        dataloader_fn = import_object(dataloader_fn_str)
        assert inspect.getsource(dataloader_fn), f"❌ Dataloader function {dataloader_fn_str} source file not found"
        
        source_code_entries = [str(path) for path in codebase_dir.glob('*')]
        logger.info(f"Source code entries: {source_code_entries}")
        
        # Get local node ID for proxy setup
        client = get_stellar_client()
        my_id = client.info().id
        logger.info(f"Local node ID: {my_id}")
        
        # Get list of client device IDs
        clients = list(navigator_task.distributed_tasks.keys())
        env_name = f'stefan-{config.framework}'
        
        def execute_with_status_reporting(device_id: str, operation_fn, start_log_msg: str, success_log_msg: str, error_log_msg: str, *args, **kwargs):
            """Execute an operation with status reporting to agent tasks.
            
            Args:
                device_id: Device ID
                operation_fn: Function to execute
                start_log_msg: Log message when operation starts
                success_log_msg: Log message when operation succeeds
                error_log_msg: Log message when operation fails (should include {device_id} placeholder)
                *args, **kwargs: Arguments to pass to operation_fn
            """
            agent_task_uuid = navigator_task.distributed_tasks.get(device_id)
            if agent_task_uuid:
                agent_task = state_manager.get_single_task(agent_task_uuid)
                if agent_task:
                    agent_task.status = TaskNodeStatus.STARTED
                    agent_log = Log(log=start_log_msg, prefix=device_id, level="info", ts=time.time())
                    agent_task.logs.append(agent_log)
                    state_manager.save_single_task(agent_task)
            
            try:
                operation_fn(device_id, *args, **kwargs)
                if agent_task_uuid:
                    agent_task = state_manager.get_single_task(agent_task_uuid)
                    if agent_task:
                        agent_log = Log(log=success_log_msg, prefix=device_id, level="info", ts=time.time())
                        agent_task.logs.append(agent_log)
                        state_manager.save_single_task(agent_task)
            except Exception as e:
                if agent_task_uuid:
                    agent_task = state_manager.get_single_task(agent_task_uuid)
                    if agent_task:
                        agent_task.status = TaskNodeStatus.ERROR
                        agent_error_log = Log(log=error_log_msg.format(device_id=device_id, error=str(e)), prefix=device_id, level="error", ts=time.time())
                        agent_task.logs.append(agent_error_log)
                        state_manager.save_single_task(agent_task)
                raise
        
        # Ping all devices to check connectivity
        logger.info("Pinging all devices...")
        navigator_log = Log(log="Pinging all devices to check connectivity", prefix="navigator", level="info", ts=time.time())
        navigator_task.logs.append(navigator_log)
        state_manager.save_single_task(navigator_task)
        
        def ping_device_wrapper(device_id: str):
            execute_with_status_reporting(
                device_id,
                ping_device,
                f"Pinging device {device_id}",
                f"Device {device_id} ping successful",
                "Failed to ping device {device_id}: {error}"
            )
        
        results = client_pool_execution(ping_device_wrapper, [(client_id,) for client_id in clients])
        pool_execution_error_handler(results)
        
        # Prepare FL devices
        logger.info("Preparing FL devices...")
        navigator_log = Log(log="Preparing FL client environments on all devices", prefix="navigator", level="info", ts=time.time())
        navigator_task.logs.append(navigator_log)
        state_manager.save_single_task(navigator_task)
        
        def prepare_fl_device_wrapper(device_id: str, env_name: str, option: str):
            execute_with_status_reporting(
                device_id,
                prepare_fl_device,
                f"Preparing FL environment {env_name}",
                f"FL environment {env_name} prepared successfully",
                f"Failed to prepare FL environment {env_name} on device {{device_id}}: {{error}}",
                env_name,
                option
            )
        
        results = client_pool_execution(prepare_fl_device_wrapper, [(client_id, env_name, config.framework) for client_id in clients])
        pool_execution_error_handler(results)
        
        # Check ports availability
        logger.info("Checking ports availability...")
        navigator_log = Log(log=f"Checking ports availability on all devices: {config.server_ports}", prefix="navigator", level="info", ts=time.time())
        navigator_task.logs.append(navigator_log)
        state_manager.save_single_task(navigator_task)
        
        def check_ports_availability_wrapper(device_id: str, env_name: str, ports: List[int]):
            execute_with_status_reporting(
                device_id,
                check_ports_availability,
                f"Checking ports availability: {ports}",
                f"All ports available: {ports}",
                f"Failed to check ports availability on device {{device_id}}: {{error}}",
                env_name,
                ports
            )
        
        results = client_pool_execution(check_ports_availability_wrapper, [(client_id, env_name, config.server_ports) for client_id in clients])
        pool_execution_error_handler(results)
        
        logger.info("All devices prepared and ready for FL training")
        navigator_log = Log(log="All devices prepared and ready for FL training", prefix="navigator", level="info", ts=time.time())
        navigator_task.logs.append(navigator_log)
        state_manager.save_single_task(navigator_task)
        
        clients_start_fn = ClientsStarterFn(navigator_task, state_manager, requirements_content, config, my_id)
        result = adapter.run_training(
            model_fn=model_fn,
            dataloader_fn=dataloader_fn,
            config=config,
            mode=FLTrainingMode.DISTRIBUTED,
            clients_start_fn=clients_start_fn,
            metrics_update_fn=metrics_update_fn,
            source_code_entries=source_code_entries
        )
        clients_start_fn.join_clients()
        logger.info(f"Task execution result: {result}")
    
    try:
        assert result is not None, f"❌ Task execution result is None"
        assert 'status' in result, f"❌ Task execution result is missing status"
        assert result['status'] == 'success', f"❌ Task execution result is not success"
        
        navigator_task.status = TaskNodeStatus.ENDED
        end_log = Log(log=f"Task {task_id} completed", prefix="navigator", level="info", ts=time.time())
        navigator_task.logs.append(end_log)
        state_manager.save_single_task(navigator_task)
        
        # Update agent tasks to ENDED
        for agent_name in navigator_task.distributed_tasks.keys():
            agent_task_uuid = navigator_task.distributed_tasks[agent_name]
            agent_task = state_manager.get_single_task(agent_task_uuid)
            if agent_task:
                agent_task.status = TaskNodeStatus.ENDED
                agent_end_log = Log(log=f"Agent {agent_name} completed", prefix=agent_name, level="info", ts=time.time())
                agent_task.logs.append(agent_end_log)
                state_manager.save_single_task(agent_task)
        
        return result
    except AssertionError as e:
        navigator_task.status = TaskNodeStatus.ERROR
        error_log = Log(log=f"Task {task_id} failed", prefix="navigator", level="error", ts=time.time())
        navigator_task.logs.append(error_log)
        state_manager.save_single_task(navigator_task)
        
        # Update agent tasks to ERROR
        for agent_name in navigator_task.distributed_tasks.keys():
            agent_task_uuid = navigator_task.distributed_tasks[agent_name]
            agent_task = state_manager.get_single_task(agent_task_uuid)
            if agent_task:
                agent_task.status = TaskNodeStatus.ERROR
                agent_error_log = Log(log=f"Agent {agent_name} failed", prefix=agent_name, level="error", ts=time.time())
                agent_task.logs.append(agent_error_log)
                state_manager.save_single_task(agent_task)
        
        raise e
    finally:
        # Cleanup custom model temp directory if it was created
        if custom_model_temp_dir and custom_model_temp_dir.exists():
            try:
                logger.info(f"Cleaning up custom model temp directory: {custom_model_temp_dir}")
                shutil.rmtree(custom_model_temp_dir)
                logger.info(f"Successfully cleaned up custom model temp directory")
            except Exception as cleanup_error:
                logger.error(f"Failed to cleanup custom model temp directory {custom_model_temp_dir}: {cleanup_error}")

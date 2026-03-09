# Stellar Server - Comprehensive Documentation & Specifications

## Table of Contents
1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Components](#components)
4. [API Endpoints](#api-endpoints)
5. [WebSocket Endpoints](#websocket-endpoints)
6. [Task Execution Flow](#task-execution-flow)
7. [Celery Task Management](#celery-task-management)
8. [State Management](#state-management)
9. [Deployment](#deployment)
10. [Issues & Optimizations](#issues--optimizations)

---

## Overview

**Stellar** is a distributed federated learning (FL) server that orchestrates machine learning training tasks across multiple clients using various FL frameworks (NVFlare, OpenFL, Flower). The server uses:

- **FastAPI** for REST API and WebSocket communication
- **Celery** with Redis for distributed task execution
- **Redis** for task state persistence and message brokering
- **Docker** for containerized deployment with GPU support
- **stefan_fl** framework for FL framework abstraction

### Key Features
- Multi-framework FL support (NVFlare, OpenFL, Flower, Dummy)
- Real-time task status and metrics streaming via WebSockets
- LLM-powered model codebase generation
- Distributed task execution with server and client workers
- Task state persistence in Redis
- GPU-enabled Docker containers

---

## Architecture

### System Components

```
┌─────────────────────────────────────────────────────────────┐
│                    FastAPI Application                      │
│  (REST API + WebSocket Server on port 1524)                │
└──────────────────────┬──────────────────────────────────────┘
                       │
        ┌──────────────┼──────────────┐
        │              │              │
        ▼              ▼              ▼
┌──────────────┐ ┌──────────┐ ┌──────────────┐
│   Redis      │ │  Celery  │ │ Task State   │
│  (Broker +   │ │  Workers │ │   Manager    │
│  Backend)    │ │          │ │  (Redis)     │
└──────────────┘ └──────────┘ └──────────────┘
        │              │
        └──────┬───────┘
               │
        ┌──────▼──────┐
        │  stefan_fl  │
        │  Framework  │
        │  Adapters   │
        └─────────────┘
```

### Service Architecture

The system consists of multiple Docker services:

1. **stellar** - Main FastAPI application server
2. **redis** - Message broker and result backend
3. **celery-flower** - Celery monitoring dashboard
4. **server-worker** - Celery worker for server-side FL tasks
5. **client-worker_N** - Celery workers for client-side FL tasks

---

## Components

### 1. FastAPI Application (`app.py`)

Main application server providing REST API and WebSocket endpoints.

**Key Responsibilities:**
- Task creation and management
- Model codebase generation
- Real-time status/metrics streaming
- Task download and cleanup
- Device/agent management

**Middleware:**
- `GlobalExceptionMiddleware` - Catches and handles unhandled exceptions

**Lifespan:**
- Initializes `TaskStateManager` on startup
- Sets up default devices

### 2. Celery Application (`celery_app.py`)

Celery configuration for distributed task execution.

**Configuration:**
- Broker: Redis
- Result Backend: Redis
- Serialization: JSON
- Task tracking: Enabled
- Task events: Enabled

### 3. Celery Tasks (`celery_tasks.py`)

Contains all Celery task definitions:

#### `generate_model_codebase`
- Generates ML model codebase using LLM (Cline agent)
- Saves model to models directory with metadata

#### `join_task`
- Client-side task that joins a federated learning workspace
- Installs requirements and extracts workspace

#### `execute_task` (Main Task)
- Orchestrates FL training execution
- Manages client workers
- Updates task state in Redis
- Handles metrics collection
- Supports multiple FL architectures

**Custom Task Class: `CallbackTask`**
- `on_success()` - Updates task status to ENDED on success
- `on_failure()` - Updates task status to ERROR on failure

### 4. Task State Manager (`task_state_manager.py`)

Redis-based state management for tasks.

**Features:**
- Thread-safe task storage
- Pydantic-based serialization/deserialization
- Task indexing for fast lookups
- Device management
- Status updates

**Redis Keys:**
- `stellar:task:{task_id}` - Task data
- `stellar:tasks:index` - Set of all task IDs

### 5. Type Definitions (`fl_types.py`)

Pydantic models for type safety:

- `FLArchitecture` - Supported FL frameworks
- `TaskNodeStatus` - Task/agent status enum
- `Task` - Main task model
- `TaskAgent` - Agent/device model
- `Log` - Log entry model
- `Metric` - Training metric model
- `TaskCreationForm` - Task creation input

### 6. Utilities (`utils.py`)

Helper functions:
- `build_task_object()` - Creates and starts tasks
- `retrive_model()` - Retrieves model metadata
- `get_available_models()` - Lists available models
- `find_task()` - Finds task by ID with Celery status sync
- `install_requirements()` - Installs Python packages

### 7. LLM Integration (`llm.py`)

LLM-powered model generation:
- `call_cline()` - Calls Cline Docker container for code generation
- `locate_model_codebase()` - Finds generated zip file
- `save_model_codebase()` - Saves model with metadata extraction

### 8. Celery Worker (`celery_worker.py`)

CLI for starting Celery workers:
- `server` - Starts server worker
- `client` - Starts client worker with reference token
- `test` - Tests worker connectivity

---

## API Endpoints

### System Endpoints

#### `GET /system/healthcheck`
Health check endpoint.

**Response:** `"Healthy"`

#### `GET /system/ports`
Get available ports.

**Query Parameters:**
- `port` (optional): Port number

#### `GET /system/agents`
Get all registered devices/agents.

**Response:** List of device objects with status and system info

#### `POST /system/clear`
Clear all completed tasks and their outputs.

**Response:** Boolean indicating success

### Task Management Endpoints

#### `GET /tasks`
List all tasks.

**Response:** List of `TaskEntry` objects (id, type, status)

#### `GET /tasks/{task_id}`
Get task details by ID.

**Response:** `Task` object with full details

#### `POST /tasks/{architecture}`
Create a new FL task.

**Path Parameters:**
- `architecture`: One of `nvflare`, `openfl`, `flower`, `dummy`

**Request Body:** `TaskCreationForm`
```json
{
  "agents": ["client1", "client2"],
  "task_info": {
    "architecture": "nvflare",
    "model_name": "my_model",
    "rounds": 10,
    "local_epochs": 1,
    "batch_size": 32,
    "learning_rate": 0.01,
    "server_host": "localhost",
    "server_ports": [8080, 8081, 8082]
  }
}
```

**Response:** Task UUID (string)

#### `POST /tasks/{task_id}/download`
Download task output as zip file.

**Response:** ZIP file download

### Model Management Endpoints

#### `POST /tasks/models/generate`
Generate model codebase using LLM.

**Query Parameters:**
- `model_name`: Name for the model
- `prompt`: Prompt for LLM code generation

**Response:** Task ID for generation task

#### `GET /tasks/models/{task_id}`
Get model generation task status.

**Response:** Task status string

#### `GET /tasks/models`
List all available models.

**Response:** List of model objects

#### `GET /tasks/openfl/models`
List OpenFL models.

**Response:** List of model objects

#### `GET /tasks/nvflare/models`
List NVFlare models.

**Response:** List of model IDs

---

## WebSocket Endpoints

All WebSocket endpoints use JSON for data transmission.

### `WS /ws/system/agents`
Stream device/agent updates.

**Messages:** Device list (JSON array)

**Update Frequency:** 5 seconds

### `WS /ws/tasks/{task_id}/metrics`
Stream training metrics for a task.

**Initial Message:** All existing metrics
**Subsequent Messages:** New metrics as they arrive

**Update Frequency:** 1 second

### `WS /ws/tasks/{task_id}/navigator_status`
Stream navigator (server) status updates.

**Messages:** Status string (PREPARED, STARTED, ENDED, ERROR)

**Update Frequency:** 1 second

### `WS /ws/tasks/{task_id}/navigator_logs`
Stream navigator logs.

**Initial Message:** All existing logs
**Subsequent Messages:** New logs as they arrive

**Update Frequency:** 1 second

### `WS /ws/tasks/{task_id}/agent/{agent_id}/status`
Stream agent status updates.

**Messages:** Status string

**Update Frequency:** 1 second

### `WS /ws/tasks/{task_id}/agent/{agent_id}/logs`
Stream agent logs.

**Initial Message:** All existing logs
**Subsequent Messages:** New logs as they arrive

**Update Frequency:** 1 second

---

## Task Execution Flow

### 1. Task Creation
```
Client → POST /tasks/{architecture}
  ↓
FastAPI creates Task object
  ↓
Task saved to Redis via TaskStateManager
  ↓
Celery task execute_task.apply_async() queued
  ↓
Returns task UUID
```

### 2. Task Execution (Server Worker)
```
Celery worker picks up execute_task
  ↓
Loads task from Redis
  ↓
Validates model and configuration
  ↓
Creates FL framework adapter
  ↓
Extracts model codebase
  ↓
Installs requirements
  ↓
Starts client workers (via ClientsStarterFn)
  ↓
Runs FL training (adapter.run_training())
  ↓
Collects metrics via metrics_update_fn
  ↓
Updates task state in Redis periodically
  ↓
On completion: Updates status to ENDED
```

### 3. Client Worker Execution
```
Server sends join_task to client queue
  ↓
Client worker receives task
  ↓
Installs requirements
  ↓
Extracts workspace
  ↓
Joins FL workspace (FLFrameworkAdapter.join_workspace())
  ↓
Participates in FL training
  ↓
Status updates sent back to server
```

### 4. Status Synchronization
```
Task state stored in Redis (TaskStateManager)
  ↓
Celery task status in Celery result backend (Redis)
  ↓
CallbackTask.on_success/on_failure updates Redis state
  ↓
find_task() syncs Celery status to Redis on read
  ↓
WebSocket endpoints stream Redis state to clients
```

---

## Celery Task Management

### Task Queues

- **`server`** - Server-side FL tasks (execute_task)
- **`client_{reference_token}`** - Client-side tasks (join_task)

### Task Discovery

`get_clients()` function discovers active client workers:
1. Inspects active Celery workers
2. Filters workers starting with `client_`
3. Extracts reference tokens from queue names
4. Returns mapping of client_name → reference_token

### Task Status Lifecycle

```
PENDING → STARTED → SUCCESS/FAILURE
         ↓
    (CallbackTask updates Redis)
         ↓
PREPARED → STARTED → ENDED/ERROR (in Redis)
```

### Status Synchronization Issues

**Current Implementation:**
1. `CallbackTask.on_success/on_failure` updates Redis on completion
2. `find_task()` checks Celery status and syncs to Redis if FAILURE
3. Task state in Redis is primary source of truth for WebSocket streams

**Problems:**
- No periodic sync between Celery and Redis
- Race conditions possible if task fails before Redis update
- `find_task()` only syncs on FAILURE, not other states
- No handling for tasks stuck in PENDING/STARTED states

---

## State Management

### Distributed Task Architecture

**Status:** ✅ Implemented - Active in production

The system uses a distributed task architecture where each worker (navigator/server and each agent/client) has its own task object in Redis. This eliminates write contention and improves scalability.

#### Structure

Each worker (navigator/server and each agent/client) has its own SingleTask object in Redis:

**Navigator Task (Main Task):**
- **Key Format:** `stellar:task:{main_task_uuid}`
- **Data Structure:**
```json
{
  "uuid": "main-task-uuid",
  "device_id": "navigator-id",
  "status": "STARTED",
  "data": {
    "task_info": {
      "architecture": "nvflare",
      "model_name": "my_model",
      "rounds": 10,
      ...
    }
  },
  "logs": [
    {
      "log": "Task started",
      "prefix": "navigator",
      "level": "info",
      "ts": 1234567890.0
    }
  ],
  "metrics": [
    {
      "site": "aggregator",
      "tag": "loss",
      "step": 1,
      "value": 0.5
    }
  ],
  "distributed_tasks": {
    "client1": "client1-task-uuid",
    "client2": "client2-task-uuid"
  }
}
```

**Agent Task (Per Client):**
- **Key Format:** `stellar:task:{agent_task_uuid}`
- **Data Structure:**
```json
{
  "uuid": "client1-task-uuid",
  "device_id": "client1",
  "status": "STARTED",
  "data": {
    "task_info": {
      "parent_task_uuid": "main-task-uuid",
      "agent_id": "client1"
    }
  },
  "logs": [
    {
      "log": "Agent started",
      "prefix": "client1",
      "level": "info",
      "ts": 1234567890.0
    }
  ],
  "metrics": [
    {
      "site": "client1",
      "tag": "loss",
      "step": 1,
      "value": 0.6
    }
  ],
  "distributed_tasks": {}
}
```

#### Benefits

1. **Reduced Contention**
   - Each worker updates only its own task object
   - No conflicts when multiple agents update simultaneously
   - Parallel writes to different Redis keys

2. **Better Scalability**
   - Adding more agents doesn't increase single object size
   - Each agent's data is independent
   - Easier to shard across Redis instances if needed

3. **Improved Performance**
   - Smaller objects = faster serialization/deserialization
   - Less data transferred on each update
   - Better Redis memory utilization

4. **Clearer Ownership**
   - Each worker owns its task object
   - Clear separation of concerns
   - Easier to implement per-worker permissions

5. **Easier Debugging**
   - Can inspect individual worker tasks independently
   - Clearer audit trail per worker
   - Isolated failure tracking

#### Composition Logic

The public `get_task()` API composes the original `Task` structure from distributed SingleTask objects for API compatibility:

```python
def get_task(task_id: str) -> Optional[Task]:
    """
    Get composed task structure from distributed task objects.
    
    1. Load navigator task (main task)
    2. Load all distributed agent tasks from distributed_tasks mapping
    3. Compose into original Task structure for API compatibility
    """
    # Load navigator task
    navigator_task = get_single_task(task_id)  # SingleTask model
    if not navigator_task:
        return None
    
    # Load all agent tasks
    agents = {}
    for agent_id, agent_task_uuid in navigator_task.distributed_tasks.items():
        agent_task = get_single_task(agent_task_uuid)
        if agent_task:
            agents[agent_id] = TaskAgent(
                status=agent_task.status,
                logs=agent_task.logs
            )
    
    # Compose into Task structure
    return Task(
        uuid=navigator_task.uuid,
        navigator_id=navigator_task.device_id,
        navigator_status=navigator_task.status,
        navigator_data=navigator_task.data,
        navigator_logs=navigator_task.logs,
        agents=agents,
        metrics=navigator_task.metrics + [m for agent in agents.values() for m in agent.metrics]
    )
```

#### Implementation Details

- Use existing `SingleTask` Pydantic model (already defined in `fl_types.py`)
- Navigator task's `distributed_tasks` field maps `agent_id → agent_task_uuid`
- Each agent gets its own UUID when task is created
- Metrics can be aggregated from all tasks or kept separate
- WebSocket streams can subscribe to individual task updates

### Task Index

All task IDs are stored in a Redis set: `stellar:tasks:index`

Contains all task UUIDs (navigator + all agents).

Used for:
- Listing all tasks (`list_task_ids()`)
- Task existence checks
- Cleanup operations

### State Updates

State updates follow the distributed architecture:
- Each worker updates only its own task object
- Navigator updates its task + `distributed_tasks` mapping
- Agents update only their own task objects
- Composition happens on read via `get_task()`

**Update Locations:**
1. **Task execution** - Updates during training
2. **CallbackTask** - Updates on success/failure
3. **Client workers** - Updates agent status
4. **Metrics collection** - Appends metrics
5. **Log collection** - Appends logs

**Concurrency:** 
- Redis operations are atomic
- No write contention - each worker updates different Redis keys
- Parallel updates from multiple workers are safe

---

## Deployment

### Docker Compose Services

#### Base Configuration (`docker-compose.yml`)
- **redis**: Redis server (port 6379)
- **stellar**: FastAPI app (port 1524)
- **celery-flower**: Monitoring (port 5555)
- **server-worker**: Server worker

#### Override Configuration (`docker-compose.override.yml`)
- Overrides for development
- Server ports: 8080-8082
- Client workers with reference tokens

### Environment Variables

- `CELERY_BROKER_URL` - Redis broker URL
- `CELERY_RESULT_BACKEND` - Redis result backend URL
- `LLM_MODEL_DIR` - Directory for LLM-generated models
- `SERVER_HOST` - Server hostname/IP
- `SERVER_PORT_RANGE` - Server port range (start-end)

### Dockerfile

- Base: `python:3.11-slim`
- Installs stefan_fl from GitLab
- Installs additional dependencies
- Copies stellar Python files
- Entrypoint: `uvicorn app:app --host 0.0.0.0 --port 1524`

### GPU Support

All services configured with NVIDIA GPU access:
```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: all
          capabilities: [gpu]
```

---

## Issues & Optimizations

### Critical Issues

#### 1. **Celery Task Status Synchronization Gap**

**Problem:**
- Celery task status (in result backend) and Redis task state can become desynchronized
- `find_task()` only syncs FAILURE status, not other states
- No periodic background sync
- Race conditions if task fails before `on_failure` callback executes

**Impact:**
- WebSocket streams may show incorrect status
- API endpoints may return stale status
- Task cleanup may miss failed tasks

**Recommendation:**
```python
# Add periodic sync in find_task()
async def _cb(task_id: str) -> Optional[Task]:
    state_manager = get_task_state_manager()
    task = state_manager.get_task(task_id)
    
    if not task:
        return None
    
    result = AsyncResult(task_id, app=celery_app)
    celery_status = result.status
    
    # Sync all states, not just FAILURE
    status_map = {
        "PENDING": TaskNodeStatus.PREPARED,
        "STARTED": TaskNodeStatus.STARTED,
        "SUCCESS": TaskNodeStatus.ENDED,
        "FAILURE": TaskNodeStatus.ERROR,
        "REVOKED": TaskNodeStatus.ERROR,
        "RETRY": TaskNodeStatus.STARTED,
    }
    
    if celery_status in status_map:
        expected_status = status_map[celery_status]
        if task.navigator_status != expected_status:
            task.navigator_status = expected_status
            if celery_status == "FAILURE":
                error_info = result.info
                error_log = Log(
                    log=str(error_info) if error_info else "Task failed",
                    prefix="navigator",
                    level="error",
                    ts=time.time()
                )
                task.navigator_logs.append(error_log)
            state_manager.save_task(task)
    
    return task
```

#### 2. **Missing Import in utils.py**

**Problem:**
- `find_task()` uses `time.time()` but `time` module is not imported

**Fix:**
```python
import time  # Add to imports
```

#### 3. **No Error Handling for None Task in find_task()**

**Problem:**
- `find_task()` can return None but code assumes task exists when checking Celery status

**Fix:**
```python
async def _cb(task_id: str) -> Optional[Task]:
    state_manager = get_task_state_manager()
    task = state_manager.get_task(task_id)
    
    if not task:
        return None  # Early return
    
    result = AsyncResult(task_id, app=celery_app)
    # ... rest of sync logic
```

#### 4. **Client Worker Error Handling**

**Problem:**
- `ClientsStarterFn._client_fn()` catches exceptions but doesn't propagate to main task
- If client fails, server task may continue thinking client succeeded

**Current Code:**
```python
try:
    result_obj.get(propagate=True)
    # ... success handling
except Exception as e:
    # ... error handling
    # But doesn't raise - task continues
```

**Recommendation:**
- Consider retry logic for transient failures
- Add timeout handling
- Optionally fail main task if critical client fails

#### 5. **No Task Timeout Handling**

**Problem:**
- No timeout for long-running tasks
- Tasks can hang indefinitely
- No way to cancel stuck tasks

**Recommendation:**
```python
@celery_app.task(base=CallbackTask, bind=True, name="celery_tasks.execute_task")
def execute_task(self, task_id: str):
    # Add timeout
    try:
        with timeout(3600):  # 1 hour timeout
            # ... task execution
    except TimeoutError:
        task.navigator_status = TaskNodeStatus.ERROR
        # ... error handling
```

#### 6. **Race Condition in Task State Updates**

**Problem:**
- Multiple threads/processes can update same task simultaneously
- No locking mechanism
- Last write wins, may lose updates

**Current Recommendation:**
- Use Redis transactions (MULTI/EXEC)
- Or use Redis locks (redis-py `Lock`)
- Or use optimistic locking with version numbers

**Solution:**
- ✅ **Distributed task architecture** (see [Distributed Task Architecture](#distributed-task-architecture)) eliminates this issue
- Each worker updates only its own task object
- No contention between workers updating different keys
- Natural isolation of updates

#### 7. **Metrics Update Function Issues**

**Problem:**
- `metrics_update_fn` creates new `last_metrics_i` dict each call
- Should persist across calls to avoid duplicate metrics
- No deduplication logic

**Current Code:**
```python
def metrics_update_fn(metrics: FLTrainingMetrics):
    metrics.last_metrics_i = {}  # Resets every time!
```

**Fix:**
```python
def metrics_update_fn(metrics: FLTrainingMetrics):
    if not hasattr(metrics, 'last_metrics_i') or metrics.last_metrics_i is None:
        metrics.last_metrics_i = {}
    # ... rest of logic
```

#### 8. **No Cleanup for Failed Tasks**

**Problem:**
- Failed tasks remain in Redis indefinitely
- Output files may not be cleaned up
- `/system/clear` only clears SUCCESS tasks

**Recommendation:**
- Add cleanup for ERROR tasks
- Add TTL to Redis keys
- Add background cleanup job

### Performance Optimizations

#### 1. **Distributed Task Architecture** ✅

**Status:** ✅ Implemented

The distributed task architecture has been successfully implemented. Each worker (navigator and agents) now has its own SingleTask object in Redis, eliminating write contention and improving scalability.

**Implementation:**
- Uses `SingleTask` model from `fl_types.py` with `distributed_tasks` field
- Navigator task contains mapping of `agent_id → agent_task_uuid`
- Each agent has separate task UUID and SingleTask object
- `get_task()` composes Task structure from distributed SingleTasks for API compatibility

**Benefits Achieved:**
- ✅ Eliminated write contention
- ✅ Parallel updates from multiple workers
- ✅ Smaller objects = faster operations
- ✅ Better scalability with many agents
- ✅ Clearer ownership and debugging

#### 2. **Reduce Redis Writes**

**Problem:**
- Task state saved to Redis on every metric/log update
- High write frequency can impact performance

**Optimization:**
- Batch updates (save every N updates or every T seconds)
- Use Redis pipelines for multiple operations
- **Note:** Distributed architecture naturally reduces write frequency per key since each worker updates its own object

#### 3. **WebSocket Polling Optimization**

**Problem:**
- WebSocket endpoints poll Redis every 1 second
- Multiple concurrent connections = high Redis load

**Optimization:**
- Use Redis pub/sub for real-time updates
- Subscribe to task updates instead of polling

#### 3. **Task State Caching**

**Problem:**
- `get_task()` reads from Redis every time
- No in-memory caching

**Optimization:**
- Add LRU cache for frequently accessed tasks
- Invalidate on updates

#### 4. **Client Discovery Optimization**

**Problem:**
- `get_clients()` inspects all workers on every call
- Can be slow with many workers

**Optimization:**
- Cache client list with TTL
- Use Redis to store client registry
- Update on worker connect/disconnect events

### Code Quality Improvements

#### 1. **Error Handling**

**Issues:**
- Many `assert` statements that raise AssertionError
- No graceful error handling in some paths
- Error messages not user-friendly

**Recommendation:**
- Replace assertions with proper exceptions
- Add try/except blocks with meaningful error messages
- Return error responses instead of raising

#### 2. **Logging**

**Issues:**
- Inconsistent logging levels
- Some errors only logged, not handled
- No structured logging

**Recommendation:**
- Use structured logging (JSON format)
- Add correlation IDs for request tracing
- Log all state transitions

#### 3. **Type Safety**

**Issues:**
- Some functions lack type hints
- `any` type used in some places

**Recommendation:**
- Add complete type hints
- Use `typing` module properly
- Enable mypy type checking

#### 4. **Configuration Management**

**Issues:**
- Hardcoded values scattered throughout code
- Environment variables accessed directly

**Recommendation:**
- Create configuration class
- Validate configuration on startup
- Use pydantic-settings for config

### Security Considerations

#### 1. **Authentication/Authorization**

**Issues:**
- No authentication on API endpoints
- No authorization checks
- Reference tokens in client workers not validated

**Recommendation:**
- Add API key or JWT authentication
- Implement role-based access control
- Validate reference tokens

#### 2. **Input Validation**

**Issues:**
- Limited validation on task creation
- No sanitization of user inputs
- File paths not validated

**Recommendation:**
- Add comprehensive input validation
- Sanitize file paths
- Validate model names and configurations

#### 3. **Resource Limits**

**Issues:**
- No limits on task execution time
- No limits on number of concurrent tasks
- No limits on resource usage

**Recommendation:**
- Add task queue limits
- Implement resource quotas
- Add rate limiting

### Monitoring & Observability

#### 1. **Metrics Collection**

**Missing:**
- Task execution time metrics
- Error rate metrics
- Queue depth metrics
- Resource usage metrics

**Recommendation:**
- Add Prometheus metrics
- Export metrics endpoint
- Integrate with monitoring system

#### 2. **Distributed Tracing**

**Missing:**
- No request tracing across services
- No correlation IDs

**Recommendation:**
- Add OpenTelemetry instrumentation
- Trace requests from API to Celery to workers

#### 3. **Health Checks**

**Current:**
- Basic health check endpoint

**Enhancement:**
- Add dependency health checks (Redis, Celery)
- Add readiness/liveness probes
- Return detailed health status

---

## Summary

The Stellar server is a well-architected distributed FL orchestration system with:

**Strengths:**
- Clean separation of concerns
- Real-time status streaming
- Multi-framework support
- Docker-based deployment
- Redis-based state management

**Recent Improvements:**
- ✅ **Distributed Task Architecture** - Implemented distributed task structure with separate SingleTask objects per worker
- ✅ **Architecture:** Each worker (navigator + agents) has its own SingleTask object with `distributed_tasks` mapping in navigator task
- ✅ **API Compatibility:** `get_task()` composes Task structure from distributed SingleTasks for API compatibility

**Areas for Improvement:**
- **Critical:** Fix Celery-Redis status synchronization
- **Critical:** Add missing imports and error handling
- **Important:** Implement task timeouts and cancellation
- **Important:** Add proper error handling and recovery
- **Performance:** Optimize Redis writes and WebSocket polling
- **Security:** Add authentication and input validation
- **Observability:** Add comprehensive monitoring

**Priority Actions:**
1. **Task Status Synchronization** - The most critical remaining issue is the status synchronization gap between Celery and Redis, which can lead to inconsistent state and incorrect status reporting. This should be addressed with a comprehensive sync mechanism that handles all task states, not just failures.
2. ✅ **Distributed Task Architecture** - Successfully implemented. The distributed task structure significantly improves scalability, eliminates contention, and enables better parallel processing.


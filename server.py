import asyncio
import json
import logging
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from agents.base_agent import BaseAgent
from schemas.base import AgentCard, Message, MessageRole, Task, TaskState, TaskStatus, TextPart

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="A2A Protocol Server", version="1.0.0")

# Enable CORS for all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Store for agents managed by this server
agents: Dict[str, BaseAgent] = {}

# Store for tasks managed by this server
tasks: Dict[str, Task] = {}

# Event subscription store
task_subscribers: Dict[str, List[asyncio.Queue]] = {}

# Helper function to handle serialization with different Pydantic versions
def serialize_model(model: Any) -> Dict[str, Any]:
    """Serialize a Pydantic model to dict, compatible with both v1 and v2"""
    if hasattr(model, "model_dump"):  # Pydantic v2
        return model.model_dump()
    elif hasattr(model, "dict"):      # Pydantic v1
        return model.dict()
    else:
        # Fallback for custom models
        return {k: v for k, v in model.__dict__.items() if not k.startswith('_')}

# JSON-RPC Request Models
class TaskIdParams(BaseModel):
    id: str
    metadata: Optional[Dict[str, Any]] = None

class TaskQueryParams(BaseModel):
    id: str
    historyLength: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None

class TaskSendParams(BaseModel):
    id: str
    sessionId: Optional[str] = None
    message: Message
    historyLength: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None

class JSONRPCRequest(BaseModel):
    jsonrpc: str = "2.0"
    id: Optional[Union[int, str]] = None
    method: str
    params: Union[TaskIdParams, TaskQueryParams, TaskSendParams]

class JSONRPCResponse(BaseModel):
    jsonrpc: str = "2.0"
    id: Optional[Union[int, str]] = None
    result: Any = None
    error: Optional[Dict[str, Any]] = None

# Register agents with the server
def register_agent(agent: BaseAgent) -> None:
    """Register an agent with the server"""
    agents[agent.id] = agent
    logger.info(f"Registered agent: {agent.name} (ID: {agent.id})")

# Route to serve the Agent Card
@app.get("/.well-known/agent.json")
async def get_agent_card(agent_id: Optional[str] = None) -> AgentCard:
    """Serve the agent card according to the A2A protocol"""
    if not agent_id:
        # If no agent_id specified and we have only one agent, use that one
        if len(agents) == 1:
            agent = next(iter(agents.values()))
        else:
            # For demo purposes, return the first agent if multiple are registered
            if agents:
                agent = next(iter(agents.values()))
            else:
                raise HTTPException(status_code=404, detail="No agent registered")
    elif agent_id not in agents:
        raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")
    else:
        agent = agents[agent_id]

    return agent.get_agent_card()

# Route to send a task without streaming
@app.post("/")
async def jsonrpc_endpoint(request: JSONRPCRequest) -> JSONRPCResponse:
    """Handle JSON-RPC requests according to the A2A protocol"""
    method = request.method

    try:
        if method == "tasks/send":
            return await handle_send_task(request)
        elif method == "tasks/get":
            return await handle_get_task(request)
        elif method == "tasks/cancel":
            return await handle_cancel_task(request)
        else:
            return JSONRPCResponse(
                jsonrpc="2.0",
                id=request.id,
                error={
                    "code": -32601,
                    "message": "Method not found",
                    "data": None
                }
            )
    except Exception as e:
        logger.exception("Error processing request")
        return JSONRPCResponse(
            jsonrpc="2.0",
            id=request.id,
            error={
                "code": -32603,
                "message": "Internal error",
                "data": str(e)
            }
        )

async def handle_send_task(request: JSONRPCRequest) -> JSONRPCResponse:
    """Handle tasks/send method"""
    params = request.params

    # Get agent - for demo purposes, use the first registered agent
    if not agents:
        return JSONRPCResponse(
            jsonrpc="2.0",
            id=request.id,
            error={
                "code": -32603,
                "message": "No agents registered",
                "data": None
            }
        )

    agent = next(iter(agents.values()))

    # Check if this is a new task or an update to an existing task
    task_id = params.id

    if task_id in tasks:
        # Existing task - update it
        task = tasks[task_id]

        # Update the task status if it's in input-required state
        if task.status.state == TaskState.INPUT_REQUIRED:
            # Process the new message
            task.status.message = params.message
            task = agent.process_task(task)
            tasks[task_id] = task
        else:
            # Cannot send to a task that's not in input-required state
            return JSONRPCResponse(
                jsonrpc="2.0",
                id=request.id,
                error={
                    "code": -32603,
                    "message": "Task is not in input-required state",
                    "data": None
                }
            )
    else:
        # New task
        task = Task(
            id=task_id,
            sessionId=params.sessionId or str(uuid.uuid4()),
            status=TaskStatus(
                state=TaskState.SUBMITTED,
                message=params.message,
                timestamp=datetime.utcnow()
            ),
            metadata=params.metadata or {}
        )

        # Process the task
        task = agent.process_task(task)
        tasks[task_id] = task

    return JSONRPCResponse(
        jsonrpc="2.0",
        id=request.id,
        result=task
    )

async def handle_get_task(request: JSONRPCRequest) -> JSONRPCResponse:
    """Handle tasks/get method"""
    params = request.params
    task_id = params.id

    if task_id not in tasks:
        return JSONRPCResponse(
            jsonrpc="2.0",
            id=request.id,
            error={
                "code": -32001,
                "message": "Task not found",
                "data": None
            }
        )

    return JSONRPCResponse(
        jsonrpc="2.0",
        id=request.id,
        result=tasks[task_id]
    )

async def handle_cancel_task(request: JSONRPCRequest) -> JSONRPCResponse:
    """Handle tasks/cancel method"""
    params = request.params
    task_id = params.id

    if task_id not in tasks:
        return JSONRPCResponse(
            jsonrpc="2.0",
            id=request.id,
            error={
                "code": -32001,
                "message": "Task not found",
                "data": None
            }
        )

    task = tasks[task_id]

    # Only tasks in certain states can be canceled
    if task.status.state not in [TaskState.SUBMITTED, TaskState.WORKING, TaskState.INPUT_REQUIRED]:
        return JSONRPCResponse(
            jsonrpc="2.0",
            id=request.id,
            error={
                "code": -32002,
                "message": "Task cannot be canceled",
                "data": None
            }
        )

    # Get the agent that owns this task
    agent_id = next(iter(agents.keys()))  # For demo purposes, use first agent
    agent = agents[agent_id]

    # Update the task status
    cancel_message = Message(
        role=MessageRole.AGENT,
        parts=[TextPart(text="Task canceled by request")]
    )

    task = agent.update_task_status(task_id, TaskState.CANCELED, message=cancel_message)
    tasks[task_id] = task

    return JSONRPCResponse(
        jsonrpc="2.0",
        id=request.id,
        result=task
    )

# Route to handle streaming tasks
@app.post("/stream")
async def stream_endpoint(request: Request) -> StreamingResponse:
    """Handle streaming requests according to the A2A protocol"""
    try:
        data = await request.json()
        jsonrpc_request = JSONRPCRequest(**data)

        if jsonrpc_request.method == "tasks/sendSubscribe":
            return await handle_send_subscribe(jsonrpc_request)
        elif jsonrpc_request.method == "tasks/resubscribe":
            return await handle_resubscribe(jsonrpc_request)
        else:
            # Return error as a normal JSON response
            response = JSONRPCResponse(
                jsonrpc="2.0",
                id=jsonrpc_request.id,
                error={
                    "code": -32601,
                    "message": "Method not found",
                    "data": None
                }
            )
            return JSONResponse(content=serialize_model(response))
    except Exception as e:
        logger.exception("Error processing streaming request")
        response = JSONRPCResponse(
            jsonrpc="2.0",
            id=None,
            error={
                "code": -32603,
                "message": "Internal error",
                "data": str(e)
            }
        )
        return JSONResponse(content=serialize_model(response))

async def handle_send_subscribe(request: JSONRPCRequest) -> StreamingResponse:
    """Handle tasks/sendSubscribe method"""
    params = request.params

    # Create a queue for this subscription
    queue = asyncio.Queue()

    # Same logic as handle_send_task but with streaming
    if not agents:
        response = JSONRPCResponse(
            jsonrpc="2.0",
            id=request.id,
            error={
                "code": -32603,
                "message": "No agents registered",
                "data": None
            }
        )
        await queue.put(f"data: {json.dumps(serialize_model(response))}\n\n")
        return StreamingResponse(stream_generator(queue), media_type="text/event-stream")

    agent = next(iter(agents.values()))
    task_id = params.id

    # Add this queue to the subscribers for this task
    if task_id not in task_subscribers:
        task_subscribers[task_id] = []
    task_subscribers[task_id].append(queue)

    # Create task processing coroutine
    asyncio.create_task(process_task_async(agent, params, task_id, request.id))

    return StreamingResponse(stream_generator(queue), media_type="text/event-stream")

async def process_task_async(agent, params, task_id, request_id):
    """Process a task asynchronously and update subscribers"""
    try:
        if task_id in tasks:
            # Existing task - update it
            task = tasks[task_id]

            # Update the task status if it's in input-required state
            if task.status.state == TaskState.INPUT_REQUIRED:
                # Process the new message
                task.status.message = params.message
                task = agent.process_task(task)
                tasks[task_id] = task

                # Notify subscribers
                await notify_task_update(task_id, task, request_id)
            else:
                # Cannot send to a task that's not in input-required state
                error_response = JSONRPCResponse(
                    jsonrpc="2.0",
                    id=request_id,
                    error={
                        "code": -32603,
                        "message": "Task is not in input-required state",
                        "data": None
                    }
                )
                await notify_error(task_id, error_response)
        else:
            # New task
            task = Task(
                id=task_id,
                sessionId=params.sessionId or str(uuid.uuid4()),
                status=TaskStatus(
                    state=TaskState.SUBMITTED,
                    message=params.message,
                    timestamp=datetime.utcnow()
                ),
                metadata=params.metadata or {}
            )

            # Send initial update
            tasks[task_id] = task
            await notify_task_update(task_id, task, request_id)

            # Process the task
            task = agent.process_task(task)
            tasks[task_id] = task

            # Notify subscribers of the final state
            await notify_task_update(task_id, task, request_id)
    except Exception as e:
        logger.exception(f"Error processing task {task_id}")
        error_response = JSONRPCResponse(
            jsonrpc="2.0",
            id=request_id,
            error={
                "code": -32603,
                "message": "Internal error",
                "data": str(e)
            }
        )
        await notify_error(task_id, error_response)

async def handle_resubscribe(request: JSONRPCRequest) -> StreamingResponse:
    """Handle tasks/resubscribe method"""
    params = request.params
    task_id = params.id

    # Create a queue for this subscription
    queue = asyncio.Queue()

    if task_id not in tasks:
        response = JSONRPCResponse(
            jsonrpc="2.0",
            id=request.id,
            error={
                "code": -32001,
                "message": "Task not found",
                "data": None
            }
        )
        await queue.put(f"data: {json.dumps(serialize_model(response))}\n\n")
        return StreamingResponse(stream_generator(queue), media_type="text/event-stream")

    # Add this queue to the subscribers for this task
    if task_id not in task_subscribers:
        task_subscribers[task_id] = []
    task_subscribers[task_id].append(queue)

    # Send the current state immediately
    task = tasks[task_id]
    response = JSONRPCResponse(
        jsonrpc="2.0",
        id=request.id,
        result={
            "id": task.id,
            "status": task.status,
            "final": task.status.state in [TaskState.COMPLETED, TaskState.FAILED, TaskState.CANCELED]
        }
    )
    await queue.put(f"data: {json.dumps(serialize_model(response))}\n\n")

    return StreamingResponse(stream_generator(queue), media_type="text/event-stream")

async def notify_task_update(task_id: str, task: Task, request_id: Optional[Union[int, str]] = None):
    """Notify all subscribers of a task update"""
    if task_id not in task_subscribers:
        return

    response = JSONRPCResponse(
        jsonrpc="2.0",
        id=request_id,
        result={
            "id": task.id,
            "status": task.status,
            "final": task.status.state in [TaskState.COMPLETED, TaskState.FAILED, TaskState.CANCELED]
        }
    )

    # Use the serialize_model helper function to handle different Pydantic versions
    response_str = json.dumps(serialize_model(response))

    for queue in task_subscribers[task_id]:
        await queue.put(f"data: {response_str}\n\n")

async def notify_error(task_id: str, error_response: JSONRPCResponse):
    """Notify all subscribers of an error"""
    if task_id not in task_subscribers:
        return

    # Use the serialize_model helper function
    response_str = json.dumps(serialize_model(error_response))

    for queue in task_subscribers[task_id]:
        await queue.put(f"data: {response_str}\n\n")
        # Close the stream after error
        await queue.put(None)

async def stream_generator(queue):
    """Generate a stream of server-sent events"""
    try:
        while True:
            message = await queue.get()
            if message is None:
                break
            yield message
    except asyncio.CancelledError:
        # Client disconnected
        pass

def start_server(host: str = "0.0.0.0", port: int = 8000) -> None:
    """Start the A2A protocol server"""
    uvicorn.run(app, host=host, port=port)

if __name__ == "__main__":
    start_server()

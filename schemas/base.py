from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field


class TaskState(str, Enum):
    """Task states as defined in the A2A protocol."""
    SUBMITTED = "submitted"
    WORKING = "working"
    INPUT_REQUIRED = "input-required"
    COMPLETED = "completed"
    CANCELED = "canceled"
    FAILED = "failed"
    UNKNOWN = "unknown"


class PartType(str, Enum):
    """Part types as defined in the A2A protocol."""
    TEXT = "text"
    FILE = "file"
    DATA = "data"


class MessageRole(str, Enum):
    """Role types for messages as defined in the A2A protocol."""
    USER = "user"
    AGENT = "agent"


class Part(BaseModel):
    """Base class for all part types."""
    type: PartType
    metadata: Optional[Dict[str, Any]] = None


class TextPart(Part):
    """Text part as defined in the A2A protocol."""
    type: PartType = PartType.TEXT
    text: str


class FileContent(BaseModel):
    """File content as defined in the A2A protocol."""
    name: Optional[str] = None
    mimeType: Optional[str] = None
    bytes: Optional[str] = None
    uri: Optional[str] = None


class FilePart(Part):
    """File part as defined in the A2A protocol."""
    type: PartType = PartType.FILE
    file: FileContent


class DataPart(Part):
    """Data part as defined in the A2A protocol."""
    type: PartType = PartType.DATA
    data: Dict[str, Any]


class Message(BaseModel):
    """Message as defined in the A2A protocol."""
    role: MessageRole
    parts: List[Union[TextPart, FilePart, DataPart]]
    metadata: Optional[Dict[str, Any]] = None


class Artifact(BaseModel):
    """Artifact as defined in the A2A protocol."""
    name: Optional[str] = None
    description: Optional[str] = None
    parts: List[Union[TextPart, FilePart, DataPart]]
    index: int = 0
    append: Optional[bool] = None
    lastChunk: Optional[bool] = None
    metadata: Optional[Dict[str, Any]] = None


class TaskStatus(BaseModel):
    """Task status as defined in the A2A protocol."""
    state: TaskState
    message: Optional[Message] = None
    timestamp: Optional[datetime] = None


class Task(BaseModel):
    """Task as defined in the A2A protocol."""
    id: str
    sessionId: Optional[str] = None
    status: TaskStatus
    artifacts: Optional[List[Artifact]] = None
    metadata: Optional[Dict[str, Any]] = None


class AgentCapabilities(BaseModel):
    """Agent capabilities as defined in the A2A protocol."""
    streaming: bool = False
    pushNotifications: bool = False
    stateTransitionHistory: bool = False


class AgentProvider(BaseModel):
    """Agent provider information as defined in the A2A protocol."""
    organization: str
    url: Optional[str] = None


class AgentSkill(BaseModel):
    """Agent skill as defined in the A2A protocol."""
    id: str
    name: str
    description: Optional[str] = None
    tags: Optional[List[str]] = None
    examples: Optional[List[str]] = None
    inputModes: Optional[List[str]] = None
    outputModes: Optional[List[str]] = None


class AgentAuthentication(BaseModel):
    """Agent authentication schemas as defined in the A2A protocol."""
    schemes: List[str]
    credentials: Optional[str] = None


class AgentCard(BaseModel):
    """Agent card as defined in the A2A protocol."""
    name: str
    description: Optional[str] = None
    url: str
    provider: Optional[AgentProvider] = None
    version: str
    documentationUrl: Optional[str] = None
    capabilities: AgentCapabilities
    authentication: Optional[AgentAuthentication] = None
    defaultInputModes: List[str] = ["text"]
    defaultOutputModes: List[str] = ["text"]
    skills: List[AgentSkill]


# For backward compatibility
ContentType = PartType

class Content(BaseModel):
    """Legacy content model for backward compatibility."""
    type: ContentType
    value: Any


class LegacyMessage(BaseModel):
    """Legacy message model for backward compatibility."""
    id: str
    sender_id: str
    recipient_id: str
    content: List[Content]
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    in_reply_to: Optional[str] = None
    sequence_num: Optional[int] = None


class FunctionCall(BaseModel):
    """Function call as used in the A2A protocol."""
    function_name: str
    arguments: Dict[str, Any]


class FunctionResult(BaseModel):
    """Function result as used in the A2A protocol."""
    function_name: str
    result: Any
    error: Optional[str] = None


class AgentSpec(BaseModel):
    """Legacy agent specification for backward compatibility."""
    id: str
    name: str
    description: str
    version: str
    supported_protocols: List[str] = []
    supported_functions: List[str] = []


class ConversationSession(BaseModel):
    """Session for tracking conversations between agents."""
    id: str
    participants: List[str]
    created_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    messages: List[LegacyMessage] = []

import logging
import uuid
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import schemas.base
from schemas.base import (
    AgentCapabilities,
    AgentCard,
    AgentProvider,
    AgentSkill,
    AgentSpec,
    Artifact,
    Content,
    ContentType,
    DataPart,
    FilePart,
    MessageRole,
    Task,
    TaskState,
    TaskStatus,
    TextPart,
)
from schemas.base import LegacyMessage as Message

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """Base class for all agents in the A2A protocol."""

    def __init__(
        self,
        id: str = None,
        name: str = None,
        description: str = None,
        version: str = "1.0",
        url: str = "http://localhost:8000"
    ):
        """Initialize the base agent with its identity information."""
        self.id = id or str(uuid.uuid4())
        self.name = name or self.__class__.__name__
        self.description = description or "A2A Protocol Compatible Agent"
        self.version = version
        self.url = url
        self.supported_protocols = ["a2a-2025"]
        self.supported_functions = []
        self.conversation_history: List[Message] = []
        self.tasks: Dict[str, Task] = {}

    def get_spec(self) -> AgentSpec:
        """Get the agent's specification (legacy format)."""
        return AgentSpec(
            id=self.id,
            name=self.name,
            description=self.description,
            version=self.version,
            supported_protocols=self.supported_protocols,
            supported_functions=self.supported_functions
        )

    def get_agent_card(self) -> AgentCard:
        """Get the agent card according to the A2A protocol."""
        return AgentCard(
            name=self.name,
            description=self.description,
            url=self.url,
            provider=AgentProvider(
                organization="Demo Organization"
            ),
            version=self.version,
            capabilities=AgentCapabilities(
                streaming=False,
                pushNotifications=False,
                stateTransitionHistory=True
            ),
            skills=self._get_agent_skills()
        )

    def _get_agent_skills(self) -> List[AgentSkill]:
        """Get the skills supported by this agent."""
        # Default implementation returns a basic conversation skill
        return [
            AgentSkill(
                id="conversation",
                name="Conversation",
                description="Ability to engage in natural language conversation",
                inputModes=["text"],
                outputModes=["text"]
            )
        ]

    def create_message(
        self,
        recipient_id: str,
        content_value: Any,
        content_type: ContentType = ContentType.TEXT,
        in_reply_to: Optional[str] = None
    ) -> Message:
        """Create a legacy message to be sent to another agent."""
        message = Message(
            id=str(uuid.uuid4()),
            sender_id=self.id,
            recipient_id=recipient_id,
            content=[
                Content(type=content_type, value=content_value)
            ],
            timestamp=datetime.utcnow(),
            metadata={},
            in_reply_to=in_reply_to,
            sequence_num=len(self.conversation_history) + 1
        )

        self.conversation_history.append(message)
        return message

    def create_a2a_message(
        self,
        role: MessageRole,
        text_content: str = None,
        data_content: Dict[str, Any] = None,
        file_content: Dict[str, Any] = None,
    ) -> schemas.base.Message:
        """Create a message according to the A2A protocol."""
        parts = []

        if text_content is not None:
            parts.append(TextPart(text=text_content))

        if data_content is not None:
            parts.append(DataPart(data=data_content))

        if file_content is not None:
            parts.append(FilePart(file=file_content))

        return schemas.base.Message(
            role=role,
            parts=parts
        )

    def create_task(self, user_message: Union[str, schemas.base.Message]) -> Task:
        """Create a new task from a user message."""
        task_id = str(uuid.uuid4())

        if isinstance(user_message, str):
            message = self.create_a2a_message(
                role=MessageRole.USER,
                text_content=user_message
            )
        else:
            message = user_message

        task = Task(
            id=task_id,
            sessionId=str(uuid.uuid4()),
            status=TaskStatus(
                state=TaskState.SUBMITTED,
                message=message,
                timestamp=datetime.utcnow()
            ),
            metadata={}
        )

        self.tasks[task_id] = task
        return task

    def update_task_status(self, task_id: str, state: TaskState, message: Optional[schemas.base.Message] = None) -> Task:
        """Update the status of a task."""
        if task_id not in self.tasks:
            raise ValueError(f"Task {task_id} not found")

        task = self.tasks[task_id]
        task.status.state = state
        task.status.timestamp = datetime.utcnow()

        if message is not None:
            task.status.message = message

        return task

    def add_task_artifact(self, task_id: str, artifact: Artifact) -> Task:
        """Add an artifact to a task."""
        if task_id not in self.tasks:
            raise ValueError(f"Task {task_id} not found")

        task = self.tasks[task_id]

        if task.artifacts is None:
            task.artifacts = []

        task.artifacts.append(artifact)
        return task

    def receive_message(self, message: Message) -> None:
        """
        Process an incoming message from another agent (legacy format).

        Args:
            message: The message to process
        """
        # Validate that this agent is the intended recipient
        if message.recipient_id != self.id:
            logger.warning(f"Message {message.id} not intended for this agent (expected: {self.id})")
            return

        # Add to conversation history
        self.conversation_history.append(message)
        logger.debug(f"Agent {self.id} received message {message.id}")

        # Process the message
        self._process_message(message)

    def process_task(self, task: Task) -> Task:
        """
        Process an incoming task according to the A2A protocol.

        Args:
            task: The task to process

        Returns:
            The updated task
        """
        # Store the task if we don't already have it
        if task.id not in self.tasks:
            self.tasks[task.id] = task

        # Update status to working
        task = self.update_task_status(task.id, TaskState.WORKING)

        # Process the task
        try:
            result_task = self._process_task(task)

            # Make sure we have the latest task version
            if result_task is not None:
                self.tasks[task.id] = result_task

            # If processing didn't update the status to a terminal state, mark as completed
            if self.tasks[task.id].status.state not in [TaskState.COMPLETED, TaskState.FAILED, TaskState.CANCELED]:
                self.update_task_status(task.id, TaskState.COMPLETED)

        except Exception as e:
            # Mark as failed if an exception occurred
            error_message = self.create_a2a_message(
                role=MessageRole.AGENT,
                text_content=f"Error processing task: {str(e)}"
            )
            self.update_task_status(task.id, TaskState.FAILED, message=error_message)

        return self.tasks[task.id]

    @abstractmethod
    def _process_task(self, task: Task) -> Optional[Task]:
        """
        Process a task according to the A2A protocol. This method should be implemented by subclasses.

        Args:
            task: The task to process

        Returns:
            The updated task, or None if no updates are needed
        """
        pass

    @abstractmethod
    def _process_message(self, message: Message) -> None:
        """
        Process an incoming message (legacy format). This method should be implemented by subclasses.

        Args:
            message: The message to process
        """
        pass

    @abstractmethod
    def generate_response(self, message: Message) -> Message:
        """
        Generate a response to a message (legacy format). This method should be implemented by subclasses.

        Args:
            message: The message to respond to

        Returns:
            The response message
        """
        pass

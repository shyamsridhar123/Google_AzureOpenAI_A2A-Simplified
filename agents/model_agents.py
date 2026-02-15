import json
import logging
from typing import List, Optional

import schemas.base
from agents.base_agent import BaseAgent
from models.model_implementations import GPT41Model, GPT45Model, GPTO3MiniModel
from schemas.base import Artifact, ContentType, MessageRole, Task, TaskState, TextPart
from schemas.base import LegacyMessage as Message

logger = logging.getLogger(__name__)


class GPT45Agent(BaseAgent):
    """An agent powered by the GPT-4.5 model."""

    def __init__(self, id: str = None, name: str = None):
        description = "An advanced reasoning agent powered by GPT-4.5"
        super().__init__(id, name or "GPT-4.5 Agent", description)
        self.model = GPT45Model()
        self.supported_functions = [
            "analyze_data",
            "complex_reasoning",
            "task_planning"
        ]

    def _process_message(self, message: Message) -> None:
        """Process an incoming legacy message using the GPT-4.5 model."""
        logger.info(f"GPT-4.5 Agent received: {message.id}")
        for content in message.content:
            if content.type == ContentType.TEXT:
                logger.debug(f"Content: {content.value[:100]}...")

    def _process_task(self, task: Task) -> Optional[Task]:
        """
        Process a task according to the A2A protocol using the GPT-4.5 model.

        Args:
            task: The task to process

        Returns:
            The updated task, or None if no updates are needed
        """
        # Extract text from all text parts in the message
        message_text = self._extract_text_from_message(task.status.message)

        # Create prompt from message text
        messages = [
            {"role": "system", "content": f"You are {self.name}, {self.description}. Follow the Agent-to-Agent protocol when communicating."}
        ]

        # Add the user's message
        messages.append({"role": "user", "content": message_text})

        # Generate response
        response_text = self.model.generate_text(messages, temperature=0.7)

        # Create A2A message with response
        response_message = self.create_a2a_message(
            role=MessageRole.AGENT,
            text_content=response_text
        )

        # Create artifact with the response
        artifact = Artifact(
            name="response",
            description="Generated response",
            parts=[TextPart(text=response_text)],
            index=0
        )

        # Update task with response
        task = self.update_task_status(
            task_id=task.id,
            state=TaskState.COMPLETED,
            message=response_message
        )

        # Add artifact to task
        self.add_task_artifact(task.id, artifact)

        return task

    def _extract_text_from_message(self, message: schemas.base.Message) -> str:
        """Extract text content from all text parts in a message."""
        text_parts = []
        data_parts = []

        for part in message.parts:
            if part.type == ContentType.TEXT:
                text_parts.append(part.text)
            elif part.type == ContentType.DATA:
                data_parts.append(f"JSON Data: {json.dumps(part.data, indent=2)}")

        # Combine all parts
        all_parts = text_parts + data_parts
        return "\n".join(all_parts) if all_parts else "No text content provided."

    def generate_response(self, message: Message) -> Message:
        """Generate a response using the GPT-4.5 model (legacy format)."""
        self._format_prompt_from_message(message)

        # Convert conversation history to OpenAI format
        messages = [
            {"role": "system", "content": f"You are {self.name}, {self.description}. Follow the Agent-to-Agent protocol when communicating."}
        ]

        # Add relevant conversation history
        for msg in self.conversation_history[-5:]:  # Just use the last 5 messages for context
            if msg.sender_id == self.id:
                role = "assistant"
            else:
                role = "user"

            for content in msg.content:
                if content.type == ContentType.TEXT:
                    messages.append({"role": role, "content": content.value})

        # Generate the response
        response_text = self.model.generate_text(messages, temperature=0.7)

        # Create and return a new message
        return self.create_message(
            recipient_id=message.sender_id,
            content_value=response_text,
            content_type=ContentType.TEXT,
            in_reply_to=message.id
        )

    def _format_prompt_from_message(self, message: Message) -> str:
        """Format a prompt from a message for the GPT-4.5 model."""
        prompt_parts = []

        for content in message.content:
            if content.type == ContentType.TEXT:
                prompt_parts.append(content.value)
            elif content.type == ContentType.JSON:
                prompt_parts.append(f"JSON Data: {json.dumps(content.value, indent=2)}")
            else:
                prompt_parts.append(f"[Content of type {content.type} not directly processable as text]")

        return "\n".join(prompt_parts)

    def _get_agent_skills(self) -> List[dict]:
        """Get the skills supported by this agent for the A2A protocol."""
        return [
            {
                "id": "analyze_data",
                "name": "Data Analysis",
                "description": "Analyze structured and unstructured data",
                "inputModes": ["text", "data"],
                "outputModes": ["text", "data"]
            },
            {
                "id": "complex_reasoning",
                "name": "Complex Reasoning",
                "description": "Perform complex reasoning tasks with multiple steps",
                "inputModes": ["text"],
                "outputModes": ["text"]
            },
            {
                "id": "task_planning",
                "name": "Task Planning",
                "description": "Plan and break down complex tasks",
                "inputModes": ["text"],
                "outputModes": ["text", "data"]
            }
        ]


class GPTO3MiniAgent(BaseAgent):
    """An agent powered by the GPT-O3 Mini model."""

    def __init__(self, id: str = None, name: str = None):
        description = "An efficient assistant powered by GPT-O3 Mini"
        super().__init__(id, name or "GPT-O3 Mini Agent", description)
        self.model = GPTO3MiniModel()
        self.supported_functions = [
            "answer_questions",
            "summarize_text",
            "basic_reasoning"
        ]

    def _process_message(self, message: Message) -> None:
        """Process an incoming legacy message using the GPT-O3 Mini model."""
        logger.info(f"GPT-O3 Mini Agent received: {message.id}")
        for content in message.content:
            if content.type == ContentType.TEXT:
                logger.debug(f"Content: {content.value[:100]}...")

    def _process_task(self, task: Task) -> Optional[Task]:
        """
        Process a task according to the A2A protocol using the GPT-O3 Mini model.

        Args:
            task: The task to process

        Returns:
            The updated task, or None if no updates are needed
        """
        # Extract text from all text parts in the message
        message_text = self._extract_text_from_message(task.status.message)

        # Create prompt from message text
        messages = [
            {"role": "system", "content": f"You are {self.name}, {self.description}. Be concise and efficient in your responses."}
        ]

        # Add the user's message
        messages.append({"role": "user", "content": message_text})

        # Generate response
        response_text = self.model.generate_text(messages, temperature=0.5, max_tokens=300)

        # Create A2A message with response
        response_message = self.create_a2a_message(
            role=MessageRole.AGENT,
            text_content=response_text
        )

        # Create artifact with the response
        artifact = Artifact(
            name="response",
            description="Generated response",
            parts=[TextPart(text=response_text)],
            index=0
        )

        # Update task with response
        task = self.update_task_status(
            task_id=task.id,
            state=TaskState.COMPLETED,
            message=response_message
        )

        # Add artifact to task
        self.add_task_artifact(task.id, artifact)

        return task

    def _extract_text_from_message(self, message: schemas.base.Message) -> str:
        """Extract text content from all text parts in a message."""
        text_parts = []

        for part in message.parts:
            if part.type == ContentType.TEXT:
                text_parts.append(part.text)

        return "\n".join(text_parts) if text_parts else "No text content provided."

    def generate_response(self, message: Message) -> Message:
        """Generate a response using the GPT-O3 Mini model (legacy format)."""
        self._format_prompt_from_message(message)

        # Convert conversation history to OpenAI format
        messages = [
            {"role": "system", "content": f"You are {self.name}, {self.description}. Be concise and efficient in your responses."}
        ]

        # Add relevant conversation history (only a few for efficiency)
        for msg in self.conversation_history[-3:]:  # Just use the last 3 messages for context
            if msg.sender_id == self.id:
                role = "assistant"
            else:
                role = "user"

            for content in msg.content:
                if content.type == ContentType.TEXT:
                    messages.append({"role": role, "content": content.value})

        # Generate the response
        response_text = self.model.generate_text(messages, temperature=0.5, max_tokens=300)

        # Create and return a new message
        return self.create_message(
            recipient_id=message.sender_id,
            content_value=response_text,
            content_type=ContentType.TEXT,
            in_reply_to=message.id
        )

    def _format_prompt_from_message(self, message: Message) -> str:
        """Format a prompt from a message for the GPT-O3 Mini model."""
        # Similar to the GPT-4.5 agent but simpler for efficiency
        for content in message.content:
            if content.type == ContentType.TEXT:
                return content.value

        return "Please provide a text message."

    def _get_agent_skills(self) -> List[dict]:
        """Get the skills supported by this agent for the A2A protocol."""
        return [
            {
                "id": "answer_questions",
                "name": "Question Answering",
                "description": "Answer factual questions efficiently",
                "inputModes": ["text"],
                "outputModes": ["text"]
            },
            {
                "id": "summarize_text",
                "name": "Text Summarization",
                "description": "Create concise summaries of longer texts",
                "inputModes": ["text"],
                "outputModes": ["text"]
            },
            {
                "id": "basic_reasoning",
                "name": "Basic Reasoning",
                "description": "Perform straightforward reasoning tasks",
                "inputModes": ["text"],
                "outputModes": ["text"]
            }
        ]


class GPT41Agent(BaseAgent):
    """An agent powered by the GPT-4.1 model."""

    def __init__(self, id: str = None, name: str = None):
        description = "A reasoning agent powered by GPT-4.1"
        super().__init__(id, name or "GPT-4.1 Agent", description)
        self.model = GPT41Model()
        self.supported_functions = [
            "basic_reasoning",
            "text_analysis"
        ]

    def _process_message(self, message: Message) -> None:
        """Process an incoming legacy message using the GPT-4.1 model."""
        logger.info(f"GPT-4.1 Agent received: {message.id}")
        for content in message.content:
            if content.type == ContentType.TEXT:
                logger.debug(f"Content: {content.value[:100]}...")

    def _process_task(self, task: Task) -> Optional[Task]:
        """
        Process a task according to the A2A protocol using the GPT-4.1 model.

        Args:
            task: The task to process

        Returns:
            The updated task, or None if no updates are needed
        """
        # Extract text from all text parts in the message
        message_text = self._extract_text_from_message(task.status.message)

        # Create prompt from message text
        messages = [
            {"role": "system", "content": f"You are {self.name}, {self.description}. Provide clear and concise responses."}
        ]

        # Add the user's message
        messages.append({"role": "user", "content": message_text})

        # Generate response
        response_text = self.model.generate_text(messages, temperature=0.6)

        # Create A2A message with response
        response_message = self.create_a2a_message(
            role=MessageRole.AGENT,
            text_content=response_text
        )

        # Create artifact with the response
        artifact = Artifact(
            name="response",
            description="Generated response",
            parts=[TextPart(text=response_text)],
            index=0
        )

        # Update task with response
        task = self.update_task_status(
            task_id=task.id,
            state=TaskState.COMPLETED,
            message=response_message
        )

        # Add artifact to task
        self.add_task_artifact(task.id, artifact)

        return task

    def _extract_text_from_message(self, message: schemas.base.Message) -> str:
        """Extract text content from all text parts in a message."""
        text_parts = []

        for part in message.parts:
            if part.type == ContentType.TEXT:
                text_parts.append(part.text)

        return "\n".join(text_parts) if text_parts else "No text content provided."

    def generate_response(self, message: Message) -> Message:
        """Generate a response using the GPT-4.1 model (legacy format)."""
        self._format_prompt_from_message(message)

        # Convert conversation history to OpenAI format
        messages = [
            {"role": "system", "content": f"You are {self.name}, {self.description}. Provide clear and concise responses."}
        ]

        # Add relevant conversation history (only a few for efficiency)
        for msg in self.conversation_history[-3:]:  # Just use the last 3 messages for context
            if msg.sender_id == self.id:
                role = "assistant"
            else:
                role = "user"

            for content in msg.content:
                if content.type == ContentType.TEXT:
                    messages.append({"role": role, "content": content.value})

        # Generate the response
        response_text = self.model.generate_text(messages, temperature=0.6)

        # Create and return a new message
        return self.create_message(
            recipient_id=message.sender_id,
            content_value=response_text,
            content_type=ContentType.TEXT,
            in_reply_to=message.id
        )

    def _format_prompt_from_message(self, message: Message) -> str:
        """Format a prompt from a message for the GPT-4.1 model."""
        # Similar to the GPT-4.5 agent but simpler for efficiency
        for content in message.content:
            if content.type == ContentType.TEXT:
                return content.value

        return "Please provide a text message."

    def _get_agent_skills(self) -> List[dict]:
        """Get the skills supported by this agent for the A2A protocol."""
        return [
            {
                "id": "basic_reasoning",
                "name": "Basic Reasoning",
                "description": "Perform straightforward reasoning tasks",
                "inputModes": ["text"],
                "outputModes": ["text"]
            },
            {
                "id": "text_analysis",
                "name": "Text Analysis",
                "description": "Analyze and interpret text data",
                "inputModes": ["text"],
                "outputModes": ["text"]
            }
        ]

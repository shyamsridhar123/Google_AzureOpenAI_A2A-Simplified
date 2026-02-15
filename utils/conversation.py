import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from schemas.base import ConversationSession, Message


class ConversationManager:
    """Manages conversations between agents."""

    def __init__(self):
        """Initialize the conversation manager."""
        self.conversations: Dict[str, ConversationSession] = {}

    def create_conversation(self, participant_ids: List[str], metadata: Optional[Dict[str, Any]] = None) -> ConversationSession:
        """
        Create a new conversation session.

        Args:
            participant_ids: List of agent IDs participating in the conversation
            metadata: Optional metadata for the conversation

        Returns:
            The newly created conversation session
        """
        conversation_id = str(uuid.uuid4())
        conversation = ConversationSession(
            id=conversation_id,
            participants=participant_ids,
            created_at=datetime.utcnow(),
            metadata=metadata or {},
            messages=[]
        )

        self.conversations[conversation_id] = conversation
        return conversation

    def get_conversation(self, conversation_id: str) -> Optional[ConversationSession]:
        """
        Get a conversation by ID.

        Args:
            conversation_id: The ID of the conversation to get

        Returns:
            The conversation, or None if not found
        """
        return self.conversations.get(conversation_id)

    def add_message_to_conversation(self, conversation_id: str, message: Message) -> bool:
        """
        Add a message to a conversation.

        Args:
            conversation_id: The ID of the conversation
            message: The message to add

        Returns:
            True if the message was added, False otherwise
        """
        conversation = self.get_conversation(conversation_id)
        if not conversation:
            return False

        # Validate that the message participants are part of the conversation
        if (message.sender_id not in conversation.participants and
                message.sender_id != "system" and
                message.recipient_id not in conversation.participants and
                message.recipient_id != "system"):
            return False

        conversation.messages.append(message)
        return True

    def get_conversation_history(self, conversation_id: str) -> List[Message]:
        """
        Get the message history for a conversation.

        Args:
            conversation_id: The ID of the conversation

        Returns:
            The list of messages in the conversation
        """
        conversation = self.get_conversation(conversation_id)
        if not conversation:
            return []

        return conversation.messages.copy()

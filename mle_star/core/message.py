"""Message protocol for agent communication in MLE-STAR framework."""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional
from enum import Enum
from datetime import datetime
import uuid
import json


class MessageType(Enum):
    """Types of messages exchanged between agents."""

    TASK_REQUEST = "task_request"
    TASK_DECOMPOSITION = "task_decomposition"
    EXECUTION_REQUEST = "execution_request"
    EXECUTION_RESULT = "execution_result"
    VERIFICATION_REQUEST = "verification_request"
    VERIFICATION_RESULT = "verification_result"
    FEEDBACK = "feedback"
    ERROR = "error"
    STATUS_UPDATE = "status_update"


@dataclass
class Message:
    """
    Structured message for agent communication.

    This class defines the communication protocol between agents in the MLE-STAR framework.
    Each message has a unique ID and can reference a parent message for conversation threading.

    Attributes:
        msg_type: Type of the message (e.g., TASK_REQUEST, EXECUTION_RESULT)
        sender: Role of the sending agent (e.g., "planner", "executor")
        receiver: Role of the receiving agent
        content: Dictionary containing the actual message content
        metadata: Additional metadata (timestamps, version info, etc.)
        msg_id: Unique message identifier (auto-generated)
        timestamp: When the message was created (auto-generated)
        parent_msg_id: ID of the parent message (for threading)

    Example:
        >>> msg = Message(
        ...     msg_type=MessageType.TASK_REQUEST,
        ...     sender="workflow",
        ...     receiver="planner",
        ...     content={"task": {"description": "Train a classifier"}},
        ... )
        >>> print(msg.msg_id)
        '550e8400-e29b-41d4-a716-446655440000'
    """

    msg_type: MessageType
    sender: str
    receiver: str
    content: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Auto-generated fields
    msg_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.utcnow)
    parent_msg_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert message to dictionary representation.

        Returns:
            Dictionary containing all message fields.
        """
        return {
            'msg_id': self.msg_id,
            'msg_type': self.msg_type.value,
            'sender': self.sender,
            'receiver': self.receiver,
            'content': self.content,
            'metadata': self.metadata,
            'timestamp': self.timestamp.isoformat(),
            'parent_msg_id': self.parent_msg_id
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Message':
        """
        Create Message instance from dictionary.

        Args:
            data: Dictionary containing message data.

        Returns:
            Message instance.

        Raises:
            ValueError: If required fields are missing or invalid.
        """
        try:
            return cls(
                msg_type=MessageType(data['msg_type']),
                sender=data['sender'],
                receiver=data['receiver'],
                content=data['content'],
                metadata=data.get('metadata', {}),
                msg_id=data.get('msg_id', str(uuid.uuid4())),
                timestamp=datetime.fromisoformat(data['timestamp']) if 'timestamp' in data else datetime.utcnow(),
                parent_msg_id=data.get('parent_msg_id')
            )
        except KeyError as e:
            raise ValueError(f"Missing required field in message data: {e}")
        except ValueError as e:
            raise ValueError(f"Invalid message data: {e}")

    def to_json(self) -> str:
        """
        Serialize message to JSON string.

        Returns:
            JSON string representation of the message.
        """
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> 'Message':
        """
        Deserialize message from JSON string.

        Args:
            json_str: JSON string containing message data.

        Returns:
            Message instance.

        Raises:
            ValueError: If JSON is invalid or missing required fields.
        """
        try:
            data = json.loads(json_str)
            return cls.from_dict(data)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON string: {e}")

    def create_reply(
        self,
        msg_type: MessageType,
        content: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ) -> 'Message':
        """
        Create a reply message to this message.

        This method creates a new message that references this message as its parent,
        establishing a conversation thread.

        Args:
            msg_type: Type of the reply message.
            content: Content of the reply.
            metadata: Optional metadata for the reply.

        Returns:
            New Message instance with this message as parent.

        Example:
            >>> request = Message(...)
            >>> response = request.create_reply(
            ...     msg_type=MessageType.EXECUTION_RESULT,
            ...     content={"status": "success"}
            ... )
            >>> response.parent_msg_id == request.msg_id
            True
        """
        return Message(
            msg_type=msg_type,
            sender=self.receiver,  # Swap sender/receiver
            receiver=self.sender,
            content=content,
            metadata=metadata or {},
            parent_msg_id=self.msg_id
        )

    def __str__(self) -> str:
        """String representation of the message."""
        return (
            f"Message(id={self.msg_id[:8]}..., "
            f"type={self.msg_type.value}, "
            f"from={self.sender}, "
            f"to={self.receiver})"
        )

    def __repr__(self) -> str:
        """Detailed string representation of the message."""
        return (
            f"Message("
            f"msg_id='{self.msg_id}', "
            f"msg_type={self.msg_type}, "
            f"sender='{self.sender}', "
            f"receiver='{self.receiver}', "
            f"timestamp={self.timestamp.isoformat()}, "
            f"parent_msg_id='{self.parent_msg_id}'"
            f")"
        )


class MessageValidator:
    """Validator for Message instances."""

    @staticmethod
    def validate(message: Message) -> tuple[bool, Optional[str]]:
        """
        Validate a message instance.

        Args:
            message: Message to validate.

        Returns:
            Tuple of (is_valid, error_message).

        Example:
            >>> msg = Message(...)
            >>> is_valid, error = MessageValidator.validate(msg)
            >>> if not is_valid:
            ...     print(f"Invalid message: {error}")
        """
        # Check required fields
        if not message.sender or not isinstance(message.sender, str):
            return False, "Sender must be a non-empty string"

        if not message.receiver or not isinstance(message.receiver, str):
            return False, "Receiver must be a non-empty string"

        if not isinstance(message.content, dict):
            return False, "Content must be a dictionary"

        if not isinstance(message.msg_type, MessageType):
            return False, "msg_type must be a MessageType enum"

        # Validate UUID format
        try:
            uuid.UUID(message.msg_id)
        except ValueError:
            return False, "msg_id must be a valid UUID"

        # Validate parent_msg_id if present
        if message.parent_msg_id is not None:
            try:
                uuid.UUID(message.parent_msg_id)
            except ValueError:
                return False, "parent_msg_id must be a valid UUID"

        return True, None


class MessageHistory:
    """
    Maintains a history of messages for conversation tracking.

    This class helps track message exchanges between agents,
    useful for debugging and understanding agent interactions.
    """

    def __init__(self):
        """Initialize empty message history."""
        self.messages: list[Message] = []
        self.message_index: Dict[str, Message] = {}

    def add(self, message: Message) -> None:
        """
        Add a message to the history.

        Args:
            message: Message to add.
        """
        self.messages.append(message)
        self.message_index[message.msg_id] = message

    def get_by_id(self, msg_id: str) -> Optional[Message]:
        """
        Retrieve a message by its ID.

        Args:
            msg_id: Message ID to look up.

        Returns:
            Message if found, None otherwise.
        """
        return self.message_index.get(msg_id)

    def get_thread(self, msg_id: str) -> list[Message]:
        """
        Get the conversation thread for a message.

        Traces back through parent messages to reconstruct the full conversation.

        Args:
            msg_id: ID of the message to start from.

        Returns:
            List of messages in the thread, ordered from oldest to newest.
        """
        thread = []
        current_msg = self.get_by_id(msg_id)

        while current_msg is not None:
            thread.insert(0, current_msg)
            if current_msg.parent_msg_id:
                current_msg = self.get_by_id(current_msg.parent_msg_id)
            else:
                break

        return thread

    def get_by_sender(self, sender: str) -> list[Message]:
        """
        Get all messages from a specific sender.

        Args:
            sender: Sender role to filter by.

        Returns:
            List of messages from the sender.
        """
        return [msg for msg in self.messages if msg.sender == sender]

    def get_by_type(self, msg_type: MessageType) -> list[Message]:
        """
        Get all messages of a specific type.

        Args:
            msg_type: Message type to filter by.

        Returns:
            List of messages of the specified type.
        """
        return [msg for msg in self.messages if msg.msg_type == msg_type]

    def clear(self) -> None:
        """Clear all messages from history."""
        self.messages.clear()
        self.message_index.clear()

    def __len__(self) -> int:
        """Return the number of messages in history."""
        return len(self.messages)

    def __iter__(self):
        """Iterate over messages in chronological order."""
        return iter(self.messages)

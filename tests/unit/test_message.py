"""Unit tests for message protocol."""

import pytest
from datetime import datetime

from mle_star.core.message import (
    Message,
    MessageType,
    MessageValidator,
    MessageHistory
)


@pytest.mark.unit
class TestMessage:
    """Test Message class."""

    def test_message_creation(self, sample_message):
        """Test basic message creation."""
        assert sample_message.msg_type == MessageType.TASK_REQUEST
        assert sample_message.sender == "test_sender"
        assert sample_message.receiver == "test_receiver"
        assert sample_message.content == {"test": "data"}
        assert sample_message.msg_id is not None
        assert isinstance(sample_message.timestamp, datetime)

    def test_message_id_unique(self):
        """Test that each message gets unique ID."""
        msg1 = Message(
            msg_type=MessageType.TASK_REQUEST,
            sender="s1",
            receiver="r1",
            content={}
        )
        msg2 = Message(
            msg_type=MessageType.TASK_REQUEST,
            sender="s1",
            receiver="r1",
            content={}
        )

        assert msg1.msg_id != msg2.msg_id

    def test_message_to_dict(self, sample_message):
        """Test message serialization to dict."""
        data = sample_message.to_dict()

        assert data['msg_type'] == 'TASK_REQUEST'
        assert data['sender'] == 'test_sender'
        assert data['receiver'] == 'test_receiver'
        assert data['content'] == {"test": "data"}
        assert 'msg_id' in data
        assert 'timestamp' in data

    def test_message_from_dict(self, sample_message):
        """Test message deserialization from dict."""
        data = sample_message.to_dict()
        restored = Message.from_dict(data)

        assert restored.msg_type == sample_message.msg_type
        assert restored.sender == sample_message.sender
        assert restored.receiver == sample_message.receiver
        assert restored.content == sample_message.content
        assert restored.msg_id == sample_message.msg_id

    def test_create_reply(self, sample_message):
        """Test creating reply message."""
        reply = sample_message.create_reply(
            msg_type=MessageType.TASK_DECOMPOSITION,
            content={"reply": "data"}
        )

        assert reply.msg_type == MessageType.TASK_DECOMPOSITION
        assert reply.sender == sample_message.receiver  # Swapped
        assert reply.receiver == sample_message.sender  # Swapped
        assert reply.content == {"reply": "data"}
        assert reply.parent_msg_id == sample_message.msg_id

    def test_message_thread(self):
        """Test message threading."""
        msg1 = Message(
            msg_type=MessageType.TASK_REQUEST,
            sender="user",
            receiver="agent",
            content={}
        )

        msg2 = msg1.create_reply(
            msg_type=MessageType.TASK_DECOMPOSITION,
            content={}
        )

        msg3 = msg2.create_reply(
            msg_type=MessageType.EXECUTION_REQUEST,
            content={}
        )

        # Verify thread structure
        assert msg2.parent_msg_id == msg1.msg_id
        assert msg3.parent_msg_id == msg2.msg_id

    def test_message_metadata(self):
        """Test message metadata."""
        msg = Message(
            msg_type=MessageType.TASK_REQUEST,
            sender="s",
            receiver="r",
            content={},
            metadata={"priority": "high", "tags": ["test"]}
        )

        assert msg.metadata["priority"] == "high"
        assert "test" in msg.metadata["tags"]


@pytest.mark.unit
class TestMessageValidator:
    """Test MessageValidator class."""

    def test_validate_valid_message(self, sample_message):
        """Test validation of valid message."""
        validator = MessageValidator()
        result = validator.validate(sample_message)

        assert result['valid'] is True
        assert len(result['issues']) == 0

    def test_validate_missing_fields(self):
        """Test validation catches missing fields."""
        # Create invalid message dict
        invalid_dict = {
            'msg_type': 'TASK_REQUEST',
            # Missing sender, receiver, content
        }

        validator = MessageValidator()
        # This should raise an error when trying to create Message
        with pytest.raises(TypeError):
            Message.from_dict(invalid_dict)

    def test_validate_invalid_message_type(self):
        """Test validation catches invalid message type."""
        msg = Message(
            msg_type=MessageType.TASK_REQUEST,
            sender="s",
            receiver="r",
            content={}
        )

        # Manually corrupt the type
        msg.msg_type = "INVALID_TYPE"

        validator = MessageValidator()
        result = validator.validate(msg)

        assert result['valid'] is False
        assert any('type' in issue.lower() for issue in result['issues'])

    def test_validate_content_size(self):
        """Test validation catches oversized content."""
        huge_content = {"data": "x" * 2_000_000}  # 2MB of data

        msg = Message(
            msg_type=MessageType.TASK_REQUEST,
            sender="s",
            receiver="r",
            content=huge_content
        )

        validator = MessageValidator(max_content_size=1_000_000)
        result = validator.validate(msg)

        assert result['valid'] is False
        assert any('size' in issue.lower() for issue in result['issues'])


@pytest.mark.unit
class TestMessageHistory:
    """Test MessageHistory class."""

    def test_add_message(self):
        """Test adding messages to history."""
        history = MessageHistory()

        msg1 = Message(
            msg_type=MessageType.TASK_REQUEST,
            sender="s1",
            receiver="r1",
            content={}
        )

        msg2 = Message(
            msg_type=MessageType.TASK_DECOMPOSITION,
            sender="s2",
            receiver="r2",
            content={}
        )

        history.add(msg1)
        history.add(msg2)

        assert len(history.messages) == 2
        assert history.messages[0] == msg1
        assert history.messages[1] == msg2

    def test_get_by_id(self):
        """Test retrieving message by ID."""
        history = MessageHistory()

        msg = Message(
            msg_type=MessageType.TASK_REQUEST,
            sender="s",
            receiver="r",
            content={}
        )

        history.add(msg)

        retrieved = history.get_by_id(msg.msg_id)
        assert retrieved == msg

        # Non-existent ID
        assert history.get_by_id("nonexistent") is None

    def test_get_by_sender(self):
        """Test filtering messages by sender."""
        history = MessageHistory()

        msg1 = Message(msg_type=MessageType.TASK_REQUEST, sender="alice", receiver="bob", content={})
        msg2 = Message(msg_type=MessageType.TASK_REQUEST, sender="alice", receiver="charlie", content={})
        msg3 = Message(msg_type=MessageType.TASK_REQUEST, sender="bob", receiver="alice", content={})

        history.add(msg1)
        history.add(msg2)
        history.add(msg3)

        alice_messages = history.get_by_sender("alice")
        assert len(alice_messages) == 2
        assert all(msg.sender == "alice" for msg in alice_messages)

    def test_get_by_receiver(self):
        """Test filtering messages by receiver."""
        history = MessageHistory()

        msg1 = Message(msg_type=MessageType.TASK_REQUEST, sender="alice", receiver="bob", content={})
        msg2 = Message(msg_type=MessageType.TASK_REQUEST, sender="charlie", receiver="bob", content={})
        msg3 = Message(msg_type=MessageType.TASK_REQUEST, sender="bob", receiver="alice", content={})

        history.add(msg1)
        history.add(msg2)
        history.add(msg3)

        bob_messages = history.get_by_receiver("bob")
        assert len(bob_messages) == 2
        assert all(msg.receiver == "bob" for msg in bob_messages)

    def test_get_by_type(self):
        """Test filtering messages by type."""
        history = MessageHistory()

        msg1 = Message(msg_type=MessageType.TASK_REQUEST, sender="s", receiver="r", content={})
        msg2 = Message(msg_type=MessageType.TASK_REQUEST, sender="s", receiver="r", content={})
        msg3 = Message(msg_type=MessageType.ERROR, sender="s", receiver="r", content={})

        history.add(msg1)
        history.add(msg2)
        history.add(msg3)

        task_messages = history.get_by_type(MessageType.TASK_REQUEST)
        assert len(task_messages) == 2

        error_messages = history.get_by_type(MessageType.ERROR)
        assert len(error_messages) == 1

    def test_get_thread(self):
        """Test retrieving message thread."""
        history = MessageHistory()

        msg1 = Message(msg_type=MessageType.TASK_REQUEST, sender="s", receiver="r", content={})
        msg2 = msg1.create_reply(msg_type=MessageType.TASK_DECOMPOSITION, content={})
        msg3 = msg2.create_reply(msg_type=MessageType.EXECUTION_REQUEST, content={})

        history.add(msg1)
        history.add(msg2)
        history.add(msg3)

        # Get thread for msg3 (should include all)
        thread = history.get_thread(msg3.msg_id)
        assert len(thread) == 3

        # Get thread for msg2 (should include msg1 and msg2)
        thread = history.get_thread(msg2.msg_id)
        assert len(thread) == 2

    def test_clear(self):
        """Test clearing history."""
        history = MessageHistory()

        msg = Message(msg_type=MessageType.TASK_REQUEST, sender="s", receiver="r", content={})
        history.add(msg)

        assert len(history.messages) == 1

        history.clear()
        assert len(history.messages) == 0

    def test_history_size_limit(self):
        """Test history size limit."""
        history = MessageHistory(max_size=5)

        # Add 10 messages
        for i in range(10):
            msg = Message(
                msg_type=MessageType.TASK_REQUEST,
                sender=f"s{i}",
                receiver="r",
                content={}
            )
            history.add(msg)

        # Should only keep last 5
        assert len(history.messages) == 5
        assert history.messages[0].sender == "s5"
        assert history.messages[-1].sender == "s9"

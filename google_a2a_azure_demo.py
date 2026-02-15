#!/usr/bin/env python
"""
Google Agent-to-Agent Protocol Demo with Azure OpenAI Models
------------------------------------------------------------
This demo showcases the implementation of Google's Agent-to-Agent (A2A) protocol
using Azure OpenAI's GPT-4.5 Turbo and O3 Mini models.

The script demonstrates:
1. Creation of different AI agents with distinct capabilities
2. A2A protocol-based messaging and task handling
3. Conversation management using the A2A protocol schema
4. Different interaction modes: standard, collaborative, and interactive

Usage:
    python google_a2a_azure_demo.py                     # Run standard demo
    python google_a2a_azure_demo.py --mode=collaborative # Run collaborative mode with 3 agents
    python google_a2a_azure_demo.py --mode=interactive   # Run interactive mode with user input
"""

import argparse
import json
import logging
import os
import uuid
from datetime import datetime
from typing import Any, Dict, Optional

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up argument parser for different modes
parser = argparse.ArgumentParser(description="Google A2A Protocol Demo with Azure OpenAI Models")
parser.add_argument(
    "--mode",
    type=str,
    choices=["standard", "collaborative", "interactive"],
    default="standard",
    help="Demo mode: standard (default), collaborative, or interactive"
)
args = parser.parse_args()

# Ensure logs directory exists
os.makedirs("logs", exist_ok=True)

# Configure logging with timestamp in filename
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = f"logs/google_a2a_azure_demo_{args.mode}_{timestamp}.log"

logging.basicConfig(
    level=logging.DEBUG,  # Changed to DEBUG level for more detailed logs
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("a2a_demo")
logger.info(f"Starting A2A Demo in {args.mode} mode - Log file: {log_file}")

# Import our A2A components
try:
    from agents.model_agents import GPT41Agent, GPT45Agent, GPTO3MiniAgent
    from schemas.base import (
        Content,
        ContentType,
        LegacyMessage,
        MessageRole,
        TaskState,
    )
    from utils.conversation import ConversationManager

    # Log the available TaskState values for debugging
    logger.info(f"Available TaskState values: {[state.value for state in TaskState]}")

except ImportError as e:
    logger.critical(f"Failed to import required modules: {str(e)}")
    raise


def print_box(text: str, width: int = 80) -> None:
    """Print text in a boxed format."""
    lines = text.strip().split('\n')
    lines = [line.strip() for line in lines if line.strip()]

    print("╔" + "═" * (width - 2) + "╗")
    for line in lines:
        if len(line) <= width - 4:
            padding = width - 4 - len(line)
            print(f"║ {line}{' ' * padding} ║")
        else:
            words = line.split()
            current_line = ""
            for word in words:
                if len(current_line) + len(word) + 1 <= width - 4:
                    current_line += " " + word if current_line else word
                else:
                    padding = width - 4 - len(current_line)
                    print(f"║ {current_line}{' ' * padding} ║")
                    current_line = word
            if current_line:
                padding = width - 4 - len(current_line)
                print(f"║ {current_line}{' ' * padding} ║")
    print("╚" + "═" * (width - 2) + "╝")


def print_agent_message(agent_name: str, message_text: str, width: int = 80) -> None:
    """Print an agent's message in a formatted box."""
    print(f"\n{agent_name} says:")
    print_box(message_text, width)
    # Also log the message
    logger.info(f"{agent_name}: {message_text[:100]}...")


def print_conversation_header(title: str, width: int = 100) -> None:
    """Print a visually distinct header for the conversation."""
    print("\n")
    print("┏" + "━" * (width - 2) + "┓")
    padding = (width - len(title) - 4) // 2
    print(f"┃{' ' * padding}【 {title} 】{' ' * (width - len(title) - 4 - padding)}┃")
    print("┗" + "━" * (width - 2) + "┛")


def print_agent_card(agent_name: str, agent_id: str, agent_type: str, width: int = 100) -> None:
    """Print agent information in a card format."""
    print(f"┌{'─' * (width - 2)}┐")
    print(f"│ 🤖 Agent: {agent_name}{' ' * (width - 11 - len(agent_name))}│")
    print(f"│ 🆔 ID: {agent_id[:8]}...{' ' * (width - 16 - len(agent_id[:8]))}│")
    print(f"│ 🧠 Model: {agent_type}{' ' * (width - 11 - len(agent_type))}│")
    print(f"└{'─' * (width - 2)}┘")


def print_message_box(
    sender_name: str,
    recipient_name: str,
    message_text: str,
    message_id: Optional[str] = None,
    in_reply_to: Optional[str] = None,
    timestamp: Optional[Any] = None,
    message_type: str = "TEXT",
    width: int = 100
) -> None:
    """Print a message in a visually distinct box with metadata."""
    # Determine arrow direction based on sender/recipient
    if sender_name == "Human User":
        direction = "➜ "
        color_start = "\033[94m"  # Blue for human messages
    else:
        direction = "➜ "
        color_start = "\033[92m"  # Green for agent messages

    color_end = "\033[0m"

    # Create header with message direction
    header = f"{sender_name} {direction} {recipient_name}"

    # Add timestamp if available
    time_str = ""
    if timestamp:
        if isinstance(timestamp, str):
            time_str = timestamp
        else:
            time_str = timestamp.strftime("%H:%M:%S")

    # Print message box
    print(f"\n{color_start}┌{'─' * (width - 2)}┐{color_end}")
    print(f"{color_start}│ {header}{' ' * (width - len(header) - 3)}│{color_end}")

    # Print metadata line
    metadata = []
    if message_id:
        metadata.append(f"ID: {message_id[:6]}...")
    if in_reply_to:
        metadata.append(f"Reply to: {in_reply_to[:6]}...")
    if time_str:
        metadata.append(f"Time: {time_str}")
    if message_type:
        metadata.append(f"Type: {message_type}")

    if metadata:
        metadata_str = " | ".join(metadata)
        print(f"{color_start}│ {metadata_str}{' ' * (width - len(metadata_str) - 3)}│{color_end}")

    # Print separator line
    print(f"{color_start}├{'─' * (width - 2)}┤{color_end}")

    # Print message content
    lines = message_text.strip().split('\n')
    lines = [line.strip() for line in lines if line.strip()]

    for line in lines:
        if len(line) <= width - 4:
            padding = width - 4 - len(line)
            print(f"{color_start}│ {line}{' ' * padding} │{color_end}")
        else:
            words = line.split()
            current_line = ""
            for word in words:
                if len(current_line) + len(word) + 1 <= width - 4:
                    current_line += " " + word if current_line else word
                else:
                    padding = width - 4 - len(current_line)
                    print(f"{color_start}│ {current_line}{' ' * padding} │{color_end}")
                    current_line = word
            if current_line:
                padding = width - 4 - len(current_line)
                print(f"{color_start}│ {current_line}{' ' * padding} │{color_end}")

    # Print message footer
    print(f"{color_start}└{'─' * (width - 2)}┘{color_end}")

    # Log the message
    logger.info(f"{sender_name} to {recipient_name}: {message_text[:100]}...")


def print_structured_data(title: str, data: Dict[str, Any], width: int = 100) -> None:
    """Print structured data in a formatted box."""
    print(f"\n┌{'─' * (width - 2)}┐")
    print(f"│ 📊 {title}{' ' * (width - len(title) - 6)}│")
    print(f"├{'─' * (width - 2)}┤")

    # Convert data to pretty JSON
    json_str = json.dumps(data, indent=2)

    # Print each line
    for line in json_str.split('\n'):
        if len(line) <= width - 4:
            padding = width - 4 - len(line)
            print(f"│ {line}{' ' * padding} │")
        else:
            # If line is too long, truncate with ellipsis
            print(f"│ {line[:width-8]}... │")

    print(f"└{'─' * (width - 2)}┘")


def print_conversation_flow(conversation: Any, width: int = 100) -> None:
    """Print a visual representation of the conversation flow."""
    print(f"\n┌{'─' * (width - 2)}┐")
    print(f"│ 🔄 Conversation Flow{' ' * (width - 21)}│")
    print(f"├{'─' * (width - 2)}┤")

    for i, message in enumerate(conversation.messages):
        sender = message.sender_id
        recipient = message.recipient_id
        msg_type = message.content[0].type.value if message.content else "UNKNOWN"

        # Truncate IDs for display
        sender_short = sender[:8] + "..." if len(sender) > 8 else sender
        recipient_short = recipient[:8] + "..." if len(recipient) > 8 else recipient

        # Create flow line
        flow = f"{sender_short} ──[{msg_type}]──➤ {recipient_short}"

        # Add reply reference if present
        if message.in_reply_to:
            reply_ref = f"(↩️ {message.in_reply_to[:6]}...)"
            flow = f"{flow} {reply_ref}"

        padding = width - len(flow) - 4
        print(f"│ {i+1:2d}. {flow}{' ' * padding} │")

    print(f"└{'─' * (width - 2)}┘")


def print_protocol_message(
    message: Any,
    sender_name: Optional[str] = None,
    recipient_name: Optional[str] = None,
    width: int = 100,
    display_structured_data: bool = True
) -> None:
    """Print a complete A2A protocol message with all components."""
    if hasattr(message, 'parts'):  # A2A Message format
        # Extract message components
        text_content = None
        data_content = None

        # Get sender and recipient names if not provided
        if not sender_name:
            sender_name = "Agent"
        if not recipient_name:
            recipient_name = "User"

        for part in message.parts:
            if hasattr(part, 'text') and part.text:
                text_content = part.text
            elif hasattr(part, 'data') and part.data:
                data_content = part.data

        # Print the text content
        if text_content:
            print_message_box(
                sender_name=sender_name,
                recipient_name=recipient_name,
                message_text=text_content,
                message_type="A2A",
                width=width
            )

        # Print the data content if present and requested
        if data_content and display_structured_data:
            print_structured_data("Structured Data", data_content, width)

    elif hasattr(message, 'content'):  # Legacy Message format
        # Get sender and recipient names if not provided
        if not sender_name:
            sender_name = message.sender_id
        if not recipient_name:
            recipient_name = message.recipient_id

        # Extract message components
        content_text = None
        if message.content and len(message.content) > 0:
            if message.content[0].type == ContentType.TEXT:
                content_text = message.content[0].value

        # Print the message
        if content_text:
            print_message_box(
                sender_name=sender_name,
                recipient_name=recipient_name,
                message_text=content_text,
                message_id=message.id,
                in_reply_to=message.in_reply_to,
                timestamp=message.timestamp,
                message_type="Legacy",
                width=width
            )
    else:
        # Fallback for unknown message format
        print("\nUnknown message format:", type(message))


def generate_direct_response(agent: Any, prompt_text: str) -> str:
    """Generate a response directly using the agent's model without task mechanism."""
    logger.info(f"Generating direct response from {agent.name}")

    # Format messages for OpenAI API
    messages = [
        {"role": "system", "content": f"You are {agent.name}, {agent.description}. Follow the Agent-to-Agent protocol when communicating."},
        {"role": "user", "content": prompt_text}
    ]

    try:
        # Generate response using the agent's model
        response_text = agent.model.generate_text(messages, temperature=0.7, max_tokens=5000)
        logger.info(f"Response generated successfully from {agent.name}")
        return response_text
    except Exception as e:
        logger.error(f"Error generating response from {agent.name}: {str(e)}", exc_info=True)
        return f"Error generating response: {str(e)}"


def run_agent_protocol_demo() -> None:
    """Run the A2A protocol demo between two different Azure OpenAI agents."""
    # Create conversation header with visual separator
    print_conversation_header("GOOGLE A2A PROTOCOL DEMO WITH AZURE OPENAI MODELS", 100)
    print("Initializing agents...\n")

    # Create our agents
    logger.info("Creating agents...")
    research_agent = GPT45Agent(name="Research Specialist")
    assistant_agent = GPTO3MiniAgent(name="Information Navigator")

    # Display agent cards with visual formatting
    print_agent_card(research_agent.name, research_agent.id, "GPT-4.5 Turbo", 100)
    print_agent_card(assistant_agent.name, assistant_agent.id, "GPT-O3 Mini", 100)

    logger.info(f"Created agents: {research_agent.name} and {assistant_agent.name}")

    # Create a conversation manager to handle A2A protocol compliant conversations
    logger.info("Creating conversation...")
    manager = ConversationManager()
    conversation = manager.create_conversation(
        participant_ids=[research_agent.id, assistant_agent.id],
        metadata={
            "topic": "Google Agent-to-Agent Protocol Implementation",
            "purpose": "Technical demonstration of A2A protocol"
        }
    )

    print(f"\nCreated A2A Protocol Conversation: {conversation.id}")
    print(f"Topic: {conversation.metadata.get('topic')}")
    print(f"Purpose: {conversation.metadata.get('purpose')}")
    print("=" * 100)

    # First exchange using A2A message format
    print("\nStarting A2A protocol research conversation...")
    logger.info("Creating initial research question message...")

    # Create initial request message
    initial_question = """
    As part of my research into agent communication protocols, I need to understand
    the Google Agent-to-Agent (A2A) protocol in more detail. Can you provide information on:

    1. Key components of the A2A protocol
    2. How message exchange is structured
    3. Task handling and artifact management
    4. Advantages over ad-hoc agent communication approaches
    """

    # Create A2A message
    initial_message = research_agent.create_a2a_message(
        role=MessageRole.USER,
        text_content=initial_question
    )

    # Display the initial message with enhanced formatting
    print_protocol_message(
        initial_message,
        sender_name=research_agent.name,
        recipient_name=assistant_agent.name,
        width=100
    )

    # Generate response directly without the task mechanism
    print("\nProcessing message using A2A protocol...")
    logger.info("Generating response from assistant agent...")

    try:
        # Generate response directly using agent's model
        response_text = generate_direct_response(assistant_agent, initial_question)

        # Create A2A message for response
        response_message = assistant_agent.create_a2a_message(
            role=MessageRole.AGENT,
            text_content=response_text
        )

        # Display the response with enhanced formatting
        print_protocol_message(
            response_message,
            sender_name=assistant_agent.name,
            recipient_name=research_agent.name,
            width=100
        )
    except Exception as e:
        logger.error(f"Error generating initial response: {str(e)}", exc_info=True)
        print(f"Error generating response: {str(e)}")
        return

    # Create follow-up question
    print("\nCreating follow-up question using A2A protocol...")
    logger.info("Creating follow-up question message")

    followup_question = """
    Thank you for that overview. I'd like to understand more about the practical implementation.

    Could you specifically explain:
    1. How would you implement the A2A protocol in a real application?
    2. What are the essential schema components needed?
    3. How can Azure OpenAI models like GPT-4.5 and O3 Mini be leveraged in an A2A protocol implementation?
    """

    # Create A2A message for follow-up
    followup_message = research_agent.create_a2a_message(
        role=MessageRole.USER,
        text_content=followup_question
    )

    # Display the follow-up message with enhanced formatting
    print_protocol_message(
        followup_message,
        sender_name=research_agent.name,
        recipient_name=assistant_agent.name,
        width=100
    )

    # Generate response to follow-up
    print("\nProcessing follow-up message using A2A protocol...")
    logger.info("Generating response to follow-up question...")

    try:
        # Generate follow-up response directly
        followup_response = generate_direct_response(assistant_agent, followup_question)

        # Create A2A message for follow-up response
        followup_response_message = assistant_agent.create_a2a_message(
            role=MessageRole.AGENT,
            text_content=followup_response
        )

        # Display the follow-up response with enhanced formatting
        print_protocol_message(
            followup_response_message,
            sender_name=assistant_agent.name,
            recipient_name=research_agent.name,
            width=100
        )
    except Exception as e:
        logger.error(f"Error generating follow-up response: {str(e)}", exc_info=True)
        print(f"Error generating follow-up response: {str(e)}")
        return

    # Demonstrate A2A protocol multi-modal response with data component
    print("\nCreating multi-modal A2A response with schema diagram...")
    logger.info("Creating multi-modal A2A response with schema demonstration")

    # Create a JSON representation of the A2A schema for visualization
    a2a_schema = {
        "message": {
            "id": "unique-message-id",
            "sender_id": "agent-1-id",
            "recipient_id": "agent-2-id",
            "timestamp": "ISO-timestamp",
            "parts": [
                {"type": "text", "text": "Message content"},
                {"type": "data", "data": {"key": "value"}}
            ]
        },
        "task": {
            "id": "task-id",
            "agent_id": "executor-agent-id",
            "status": {
                "state": "working|completed|failed",
                "message": {"/*": "Message object as above*/"}
            },
            "artifacts": [
                {
                    "id": "artifact-id",
                    "name": "Result data",
                    "parts": [
                        {"type": "text", "text": "Artifact description"},
                        {"type": "data", "data": {"result": "value"}}
                    ]
                }
            ]
        }
    }

    # Create a multi-modal response with text and data parts
    try:
        logger.info("Creating schema message with multi-modal content")

        # Pass the dictionary directly for data_content
        schema_message = research_agent.create_a2a_message(
            role=MessageRole.AGENT,
            text_content="""
            Based on my analysis, here's a simplified representation of the A2A protocol schema structure.

            The attached JSON shows the core components of messages and tasks in the protocol.
            This standardized structure enables different agents to communicate reliably
            regardless of their underlying implementation.

            This is how our current conversation is structured behind the scenes.
            """,
            data_content=a2a_schema
        )

        # Display the multi-modal message with enhanced formatting
        print_protocol_message(
            schema_message,
            sender_name=research_agent.name,
            recipient_name=assistant_agent.name,
            width=100,
            display_structured_data=True
        )

        # Convert the A2A message to legacy format and add to conversation
        legacy_schema_message = research_agent.create_message(
            recipient_id=assistant_agent.id,
            content_value=schema_message.parts[0].text,
            content_type=ContentType.TEXT
        )

        # Add schema message to conversation
        manager.add_message_to_conversation(conversation.id, legacy_schema_message)
        logger.info(f"Added schema message to conversation with ID: {legacy_schema_message.id}")

    except Exception as e:
        logger.error(f"Error creating multi-modal message: {str(e)}", exc_info=True)
        print(f"Error creating multi-modal message: {str(e)}")

    # Use the legacy message format for a final exchange
    print("\nDemonstrating legacy message format compatibility...")
    logger.info("Testing legacy message format compatibility")

    try:
        # Generate a message ID to use in case we don't have a schema message ID
        fallback_message_id = str(uuid.uuid4())

        # Create a legacy format message
        legacy_message = assistant_agent.create_message(
            recipient_id=research_agent.id,
            content_value="""
            This demonstrates that our implementation supports both the full A2A protocol
            and a simpler legacy message format for backward compatibility.

            The A2A protocol offers:
            - Standardized message structure
            - Rich content types (text, data, etc.)
            - Task and artifact management
            - Metadata for tracking and organization

            Thank you for exploring the Google A2A protocol implementation with Azure OpenAI models!
            """,
            content_type=ContentType.TEXT,
            # Use legacy_schema_message.id instead of schema_message.id
            in_reply_to=legacy_schema_message.id if 'legacy_schema_message' in locals() else fallback_message_id
        )

        # Add to conversation manager
        manager.add_message_to_conversation(conversation.id, legacy_message)

        # Display the final legacy message with enhanced formatting
        print_protocol_message(
            legacy_message,
            sender_name=assistant_agent.name,
            recipient_name=research_agent.name,
            width=100
        )
    except Exception as e:
        logger.error(f"Error with legacy message format: {str(e)}", exc_info=True)
        print(f"Error with legacy message format: {str(e)}")

    # Display conversation flow with visual representation
    print_conversation_flow(conversation, width=100)

    # Display conversation summary
    logger.info("Preparing conversation summary")
    print("\nA2A Protocol Conversation Summary:")
    print(f"  - Conversation ID: {conversation.id}")
    print(f"  - Total messages: {len(conversation.messages)}")
    print(f"  - Participants: {', '.join([research_agent.name, assistant_agent.name])}")

    print(f"\nDemo completed successfully! Log file: {log_file}")
    print("=" * 100)
    logger.info("Demo completed successfully")


def run_collaborative_mode() -> None:
    """Run the A2A protocol demo in collaborative mode with three agents working together."""
    # Create conversation header with visual separator
    print_conversation_header("GOOGLE A2A PROTOCOL COLLABORATIVE MODE", 100)
    print("Initializing agents for collaborative problem solving...\n")

    # Create our agents with specialized roles
    logger.info("Creating specialized agents for collaboration...")
    research_agent = GPT45Agent(name="Research Specialist")
    planning_agent = GPT41Agent(name="Planning Strategist")
    implementation_agent = GPTO3MiniAgent(name="Implementation Expert")

    # Display agent cards with visual formatting
    print_agent_card(research_agent.name, research_agent.id, "GPT-4.5 Turbo", 100)
    print_agent_card(planning_agent.name, planning_agent.id, "GPT-4.1", 100)
    print_agent_card(implementation_agent.name, implementation_agent.id, "GPT-O3 Mini", 100)

    logger.info(f"Created collaborative agents: {research_agent.name}, {planning_agent.name}, {implementation_agent.name}")

    # Create a conversation manager for multi-agent collaboration
    logger.info("Creating collaborative conversation...")
    manager = ConversationManager()
    conversation = manager.create_conversation(
        participant_ids=[research_agent.id, planning_agent.id, implementation_agent.id],
        metadata={
            "topic": "Collaborative A2A Protocol Implementation",
            "purpose": "Multi-agent collaborative problem solving"
        }
    )

    print(f"\nCreated Collaborative Conversation: {conversation.id}")
    print(f"Topic: {conversation.metadata.get('topic')}")
    print(f"Purpose: {conversation.metadata.get('purpose')}")
    print("=" * 100)

    # Define a complex problem that requires collaboration
    challenge = """
    COLLABORATIVE CHALLENGE: Building a Real-Time Translation System

    Our organization needs to implement a real-time translation system for international
    conferences that leverages the A2A protocol for seamless agent collaboration.

    The system should:
    1. Support real-time speech recognition in multiple languages
    2. Provide accurate translations with cultural context preservation
    3. Have a fallback mechanism for handling ambiguous phrases
    4. Maintain conversation history for context-aware translations

    Each specialist should analyze this problem from their perspective and propose solutions.
    How would you design and implement this system using the A2A protocol?
    """

    print("\nPresenting collaborative challenge...")
    print_box(challenge, 100)

    # Research Agent takes the first approach
    print("\n[PHASE 1: Research Analysis]")
    logger.info("Starting research analysis phase...")

    research_prompt = f"""
    You are the Research Specialist. Analyze the collaborative challenge:

    {challenge}

    Focus on:
    - Required language technologies
    - Similar existing systems
    - Key research findings relevant to the problem
    - Potential challenges and their research-backed solutions

    Provide a comprehensive analysis that the other agents can build upon.
    """

    try:
        # Generate research analysis
        research_response = generate_direct_response(research_agent, research_prompt)

        # Create and display the research message
        research_message = research_agent.create_message(
            recipient_id=planning_agent.id,
            content_value=research_response,
            content_type=ContentType.TEXT
        )
        manager.add_message_to_conversation(conversation.id, research_message)

        # Display with enhanced formatting
        print_protocol_message(
            research_message,
            sender_name=research_agent.name,
            recipient_name=planning_agent.name,
            width=100
        )
    except Exception as e:
        logger.error(f"Error in research phase: {str(e)}", exc_info=True)
        print(f"Error in research phase: {str(e)}")
        return

    # Planning Agent builds on the research
    print("\n[PHASE 2: Strategic Planning]")
    logger.info("Starting strategic planning phase...")

    # Include the research findings in the planning prompt
    planning_prompt = f"""
    You are the Planning Strategist. Review the Research Specialist's findings:

    {research_response}

    Now develop a strategic plan for implementing the real-time translation system:

    1. Create a high-level architecture using the A2A protocol
    2. Define the communication flow between components
    3. Outline how different agents would collaborate
    4. Propose a phased implementation approach

    Focus on how the A2A protocol would facilitate agent collaboration in this context.
    """

    try:
        # Generate planning strategy
        planning_response = generate_direct_response(planning_agent, planning_prompt)

        # Create and display the planning message
        planning_message = planning_agent.create_message(
            recipient_id=implementation_agent.id,
            content_value=planning_response,
            content_type=ContentType.TEXT,
            in_reply_to=research_message.id
        )
        manager.add_message_to_conversation(conversation.id, planning_message)

        # Display with enhanced formatting
        print_protocol_message(
            planning_message,
            sender_name=planning_agent.name,
            recipient_name=implementation_agent.name,
            width=100
        )
    except Exception as e:
        logger.error(f"Error in planning phase: {str(e)}", exc_info=True)
        print(f"Error in planning phase: {str(e)}")
        return

    # Implementation Expert provides technical implementation details
    print("\n[PHASE 3: Technical Implementation]")
    logger.info("Starting technical implementation phase...")

    implementation_prompt = f"""
    You are the Implementation Expert. Based on the research and strategic plan:

    Research findings: {research_response}

    Strategic plan: {planning_response}

    Provide a technical implementation approach:

    1. Specific A2A protocol components and message structures to use
    2. Code examples for key functionality (pseudo-code or Python)
    3. Integration points with Azure OpenAI models
    4. Testing and validation approach

    Your response should be practical and technically focused, showing how the A2A protocol
    would be implemented for this real-time translation system.
    """

    try:
        # Generate implementation details
        implementation_response = generate_direct_response(implementation_agent, implementation_prompt)

        # Create and display the implementation message
        implementation_message = implementation_agent.create_message(
            recipient_id=research_agent.id,
            content_value=implementation_response,
            content_type=ContentType.TEXT,
            in_reply_to=planning_message.id
        )
        manager.add_message_to_conversation(conversation.id, implementation_message)

        # Display with enhanced formatting
        print_protocol_message(
            implementation_message,
            sender_name=implementation_agent.name,
            recipient_name=research_agent.name,
            width=100
        )
    except Exception as e:
        logger.error(f"Error in implementation phase: {str(e)}", exc_info=True)
        print(f"Error in implementation phase: {str(e)}")
        return

    # Final collaborative synthesis
    print("\n[PHASE 4: Collaborative Synthesis]")
    logger.info("Starting collaborative synthesis phase...")

    synthesis_prompt = f"""
    As the Research Specialist, review the complete collaborative process:

    1. Initial research findings
    2. Strategic planning by the Planning Strategist: {planning_response}
    3. Technical implementation by the Implementation Expert: {implementation_response}

    Create a synthesis that:

    1. Highlights how the A2A protocol enabled effective collaboration
    2. Summarizes the proposed solution
    3. Identifies next steps for implementation
    4. Reflects on how this collaborative approach compares to traditional single-agent solutions

    Your synthesis should demonstrate the value of multi-agent collaboration using the A2A protocol.
    """

    try:
        # Generate final synthesis
        synthesis_response = generate_direct_response(research_agent, synthesis_prompt)

        # Create a multi-modal response with the complete solution
        synthesis_data = {
            "solution": {
                "title": "Real-Time Translation System with A2A Protocol",
                "research_phase": {"contributor": research_agent.name, "completed": True},
                "planning_phase": {"contributor": planning_agent.name, "completed": True},
                "implementation_phase": {"contributor": implementation_agent.name, "completed": True},
                "collaboration_metrics": {
                    "messages_exchanged": 3,
                    "phases_completed": 4,
                    "agents_involved": 3
                }
            }
        }

        # Create a final A2A message with both text and structured data
        final_message = research_agent.create_a2a_message(
            role=MessageRole.AGENT,
            text_content=synthesis_response,
            data_content=synthesis_data
        )

        # Add as legacy message to conversation for tracking
        legacy_final_message = research_agent.create_message(
            recipient_id="all",
            content_value=synthesis_response,
            content_type=ContentType.TEXT,
            in_reply_to=implementation_message.id
        )
        manager.add_message_to_conversation(conversation.id, legacy_final_message)

        # Display the final synthesis with enhanced formatting
        print_protocol_message(
            final_message,
            sender_name=research_agent.name,
            recipient_name="All Participants",
            width=100,
            display_structured_data=True
        )
    except Exception as e:
        logger.error(f"Error in synthesis phase: {str(e)}", exc_info=True)
        print(f"Error in synthesis phase: {str(e)}")
        return

    # Display conversation flow with visual representation
    print_conversation_flow(conversation, width=100)

    # Display conversation summary
    logger.info("Preparing collaboration summary")
    print("\nA2A Protocol Collaboration Summary:")
    print(f"  - Conversation ID: {conversation.id}")
    print(f"  - Total messages: {len(conversation.messages)}")
    print(f"  - Participants: {', '.join([research_agent.name, planning_agent.name, implementation_agent.name])}")

    print(f"\nCollaborative demo completed successfully! Log file: {log_file}")
    print("=" * 100)
    logger.info("Collaborative demo completed successfully")


def run_interactive_mode() -> None:
    """Run the A2A protocol demo in interactive mode allowing user to interact with agents."""
    # Create conversation header with visual separator
    print_conversation_header("GOOGLE A2A PROTOCOL INTERACTIVE MODE", 100)
    print("Initializing agents for interactive communication...\n")

    # Create our agents for interactive mode
    logger.info("Creating agents for interactive mode...")
    primary_agent = GPT45Agent(name="AI Assistant")
    analysis_agent = GPTO3MiniAgent(name="Analysis Expert")

    # Display agent cards with visual formatting
    print_agent_card(primary_agent.name, primary_agent.id, "GPT-4.5 Turbo", 100)
    print_agent_card(analysis_agent.name, analysis_agent.id, "GPT-O3 Mini", 100)

    logger.info(f"Created interactive agents: {primary_agent.name}, {analysis_agent.name}")

    # Create a conversation manager for interactive communication
    logger.info("Creating interactive conversation...")
    manager = ConversationManager()
    conversation = manager.create_conversation(
        participant_ids=[primary_agent.id, analysis_agent.id, "human-user"],
        metadata={
            "topic": "Interactive A2A Protocol Exploration",
            "purpose": "Interactive demonstration of A2A protocol"
        }
    )

    print(f"\nCreated Interactive Conversation: {conversation.id}")
    print(f"Topic: {conversation.metadata.get('topic')}")

    # Display help information in an enhanced format
    print_structured_data("Interactive Mode Commands", {
        "/analyze": "Send the last message to the Analysis Expert for deeper insights",
        "/multimodal": "Request a response with both text and structured data",
        "/help": "Show this help message",
        "/quit": "Exit interactive mode"
    }, 100)

    # Track message history for context
    message_history = []
    last_user_message = None

    # Start interactive loop
    while True:
        try:
            # Get user input
            user_input = input("\nYou: ").strip()

            # Handle special commands
            if user_input.lower() == "/quit":
                print("Exiting interactive mode...")
                break
            elif user_input.lower() == "/help":
                print_structured_data("Interactive Mode Commands", {
                    "/analyze": "Send the last message to the Analysis Expert for deeper insights",
                    "/multimodal": "Request a response with both text and structured data",
                    "/help": "Show this help message",
                    "/quit": "Exit interactive mode"
                }, 100)
                continue
            elif user_input.lower() == "/analyze":
                if not last_user_message:
                    print("No message to analyze yet. Please send a message first.")
                    continue

                print(f"\nRequesting analysis from {analysis_agent.name}...")

                analysis_prompt = f"""
                As the Analysis Expert, analyze the following user message in depth:

                USER MESSAGE: "{last_user_message}"

                Provide a comprehensive analysis including:
                1. Key topics and concepts identified
                2. Underlying assumptions or implications
                3. Potential connections to A2A protocol concepts
                4. Any areas that might require further clarification

                Your analysis should be detailed and insightful, focusing on aspects that
                the primary assistant might have missed.
                """

                analysis_response = generate_direct_response(analysis_agent, analysis_prompt)

                # Create and add the analysis message
                analysis_a2a_message = analysis_agent.create_a2a_message(
                    role=MessageRole.AGENT,
                    text_content=analysis_response
                )

                # Convert to legacy format for conversation tracking
                analysis_message = analysis_agent.create_message(
                    recipient_id="human-user",
                    content_value=analysis_response,
                    content_type=ContentType.TEXT
                )
                manager.add_message_to_conversation(conversation.id, analysis_message)

                # Display with enhanced formatting
                print_protocol_message(
                    analysis_a2a_message,
                    sender_name=analysis_agent.name,
                    recipient_name="You",
                    width=100
                )

                message_history.append({"role": "assistant", "content": analysis_response, "name": analysis_agent.name})
                continue
            elif user_input.lower() == "/multimodal":
                if not last_user_message:
                    print("No message context yet. Please send a message first.")
                    continue

                print("\nGenerating multi-modal response...")

                multimodal_prompt = f"""
                Based on our conversation so far, and specifically the user's last message:
                "{last_user_message}"

                Create both a text explanation and a structured data representation related to
                the A2A protocol. Your response should demonstrate how the A2A protocol can represent
                both natural language and structured data in the same message.
                """

                multimodal_text = generate_direct_response(primary_agent, multimodal_prompt)

                # Create structured data about the A2A protocol
                structured_data = {
                    "a2a_protocol_components": {
                        "message_structure": {
                            "id": "unique-identifier",
                            "sender": "agent-id",
                            "recipient": "agent-id",
                            "content_types": ["text", "data", "file"]
                        },
                        "message_types": ["query", "response", "notification", "error"],
                        "content_formats_supported": ["text/plain", "application/json", "image/*"],
                        "conversation_context": {
                            "tracking": True,
                            "history_length": "configurable",
                            "metadata_support": True
                        }
                    },
                    "relevance_to_query": {
                        "topic": last_user_message[:50] + "..." if len(last_user_message) > 50 else last_user_message,
                        "confidence": 0.92,
                        "timestamp": datetime.now().isoformat()
                    }
                }

                # Create a multi-modal A2A message
                multimodal_a2a_message = primary_agent.create_a2a_message(
                    role=MessageRole.AGENT,
                    text_content=multimodal_text,
                    data_content=structured_data
                )

                # Convert to legacy format for conversation tracking
                multimodal_message = primary_agent.create_message(
                    recipient_id="human-user",
                    content_value=multimodal_text,
                    content_type=ContentType.TEXT
                )
                manager.add_message_to_conversation(conversation.id, multimodal_message)

                # Display the multi-modal response with enhanced formatting
                print_protocol_message(
                    multimodal_a2a_message,
                    sender_name=primary_agent.name,
                    recipient_name="You",
                    width=100,
                    display_structured_data=True
                )

                message_history.append({"role": "assistant", "content": multimodal_text, "name": primary_agent.name})
                continue

            # Save the user message for potential analysis
            last_user_message = user_input

            # Create a user message directly using the Message class instead of agent.create_message
            user_message = LegacyMessage(
                id=str(uuid.uuid4()),
                sender_id="human-user",  # Set sender_id to human-user
                recipient_id=primary_agent.id,
                content=[
                    Content(type=ContentType.TEXT, value=user_input)
                ],
                timestamp=datetime.utcnow(),
                metadata={},
                sequence_num=len(message_history) + 1
            )

            # Add message to conversation
            manager.add_message_to_conversation(conversation.id, user_message)

            # Display user message with enhanced formatting
            print_message_box(
                sender_name="You",
                recipient_name=primary_agent.name,
                message_text=user_input,
                message_id=user_message.id,
                timestamp=datetime.now(),
                message_type="Human",
                width=100
            )

            # Add to message history
            message_history.append({"role": "user", "content": user_input})

            # Generate response with context
            context_messages = [{"role": "system", "content": f"You are {primary_agent.name}, {primary_agent.description}. You're participating in an interactive A2A protocol demonstration. Keep responses helpful, informative, and focused on the A2A protocol when relevant."}]

            # Add conversation history for context
            for msg in message_history[-10:]:  # Only use the last 10 messages
                context_messages.append({"role": msg["role"], "content": msg["content"]})

            # Generate response with context
            response_text = primary_agent.model.generate_text(context_messages, temperature=0.7, max_tokens=5000)

            # Create A2A message for response
            response_a2a_message = primary_agent.create_a2a_message(
                role=MessageRole.AGENT,
                text_content=response_text
            )

            # Convert to legacy format for conversation tracking
            response_message = primary_agent.create_message(
                recipient_id="human-user",
                content_value=response_text,
                content_type=ContentType.TEXT,
                in_reply_to=user_message.id
            )
            manager.add_message_to_conversation(conversation.id, response_message)

            # Display the response with enhanced formatting
            print_protocol_message(
                response_a2a_message,
                sender_name=primary_agent.name,
                recipient_name="You",
                width=100
            )

            # Add to message history
            message_history.append({"role": "assistant", "content": response_text, "name": primary_agent.name})

        except KeyboardInterrupt:
            print("\nInterrupted. Exiting interactive mode...")
            break
        except Exception as e:
            logger.error(f"Error in interactive mode: {str(e)}", exc_info=True)
            print(f"\nEncountered an error: {str(e)}")
            print("Type /quit to exit or continue with a new message.")

    # Display conversation flow with visual representation
    print_conversation_flow(conversation, width=100)

    # Display conversation summary
    print("\nInteractive A2A Protocol Session Summary:")
    print(f"  - Conversation ID: {conversation.id}")
    print(f"  - Total messages: {len(conversation.messages)}")
    print(f"  - Messages exchanged: {len(message_history)}")

    print(f"\nInteractive demo completed! Log file: {log_file}")
    print("=" * 100)
    logger.info("Interactive demo completed")

if __name__ == "__main__":
    try:
        logger.info(f"Starting Agent-to-Agent protocol demo in {args.mode} mode")
        if args.mode == "collaborative":
            run_collaborative_mode()
        elif args.mode == "interactive":
            run_interactive_mode()
        else:
            run_agent_protocol_demo()
    except Exception as e:
        logger.exception("Critical error running A2A protocol demo")
        print(f"\nError occurred: {str(e)}")
        print(f"See log file for details: {log_file}")

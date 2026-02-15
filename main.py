
from dotenv import load_dotenv

from agents.model_agents import GPT45Agent, GPTO3MiniAgent
from schemas.base import ContentType
from utils.conversation import ConversationManager
from utils.logging_config import setup_logging

# Load environment variables
load_dotenv()

# Setup logging
setup_logging()


def print_formatted_message(message: str, max_width: int = 80) -> None:
    """Print a formatted message with line wrapping at word boundaries."""
    lines = message.strip().split('\n')
    for line in lines:
        line = line.strip()
        if len(line) <= max_width:
            print(line)
        else:
            words = line.split()
            current_line = ""
            for word in words:
                if len(current_line) + len(word) + 1 <= max_width:
                    if current_line:
                        current_line += " " + word
                    else:
                        current_line = word
                else:
                    print(current_line)
                    current_line = word
            if current_line:
                print(current_line)


def main() -> None:
    print("=" * 80)
    print("Google Agent-to-Agent Protocol Demo")
    print("Using OpenAI GPT-4.5 and GPT-O3 Mini Models")
    print("=" * 80)

    # Create our agents
    gpt45_agent = GPT45Agent(name="Research Assistant")
    gpt_o3_mini_agent = GPTO3MiniAgent(name="Knowledge Navigator")

    print("\nCreated Agents:")
    print(f"  - {gpt45_agent.name} (ID: {gpt45_agent.id})")
    print(f"  - {gpt_o3_mini_agent.name} (ID: {gpt_o3_mini_agent.id})")

    # Create a conversation manager and start a conversation
    conversation_manager = ConversationManager()
    conversation = conversation_manager.create_conversation(
        participant_ids=[gpt45_agent.id, gpt_o3_mini_agent.id],
        metadata={"topic": "A2A Protocol Demo"}
    )

    print(f"\nStarted conversation: {conversation.id}")
    print(f"Topic: {conversation.metadata.get('topic')}")
    print("-" * 80)

    # Initial message from the Research Assistant
    initial_prompt = """
    I'm researching the Agent-to-Agent protocol and need to understand its key components.
    Can you provide a concise explanation of:
    1. The core principles of A2A
    2. How message exchange works
    3. The importance of standardized schemas
    """

    # Create and send the initial message
    initial_message = gpt45_agent.create_message(
        recipient_id=gpt_o3_mini_agent.id,
        content_value=initial_prompt,
        content_type=ContentType.TEXT
    )

    conversation_manager.add_message_to_conversation(conversation.id, initial_message)
    print(f"\nMessage from {gpt45_agent.name}:")
    print_formatted_message(initial_message.content[0].value)
    print("-" * 80)

    # Process the message and generate a response
    gpt_o3_mini_agent.receive_message(initial_message)
    response_message = gpt_o3_mini_agent.generate_response(initial_message)

    # Add the response to the conversation
    conversation_manager.add_message_to_conversation(conversation.id, response_message)
    print(f"\nResponse from {gpt_o3_mini_agent.name}:")
    print_formatted_message(response_message.content[0].value)
    print("-" * 80)

    # Follow-up message from the Research Assistant
    gpt45_agent.receive_message(response_message)
    follow_up_message = gpt45_agent.generate_response(response_message)

    # Add the follow-up to the conversation
    conversation_manager.add_message_to_conversation(conversation.id, follow_up_message)
    print(f"\nFollow-up from {gpt45_agent.name}:")
    print_formatted_message(follow_up_message.content[0].value)
    print("-" * 80)

    # Final response from the Knowledge Navigator
    gpt_o3_mini_agent.receive_message(follow_up_message)
    final_message = gpt_o3_mini_agent.generate_response(follow_up_message)

    # Add the final response to the conversation
    conversation_manager.add_message_to_conversation(conversation.id, final_message)
    print(f"\nFinal response from {gpt_o3_mini_agent.name}:")
    print_formatted_message(final_message.content[0].value)
    print("-" * 80)

    # Print conversation summary
    print("\nConversation Summary:")
    print(f"  - Total messages: {len(conversation.messages)}")
    print(f"  - Participants: {', '.join([gpt45_agent.name, gpt_o3_mini_agent.name])}")
    print("=" * 80)


if __name__ == "__main__":
    main()

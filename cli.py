import argparse

from dotenv import load_dotenv

from agents.model_agents import GPT45Agent, GPTO3MiniAgent
from schemas.base import ContentType
from utils.conversation import ConversationManager
from utils.logging_config import setup_logging

# Load environment variables
load_dotenv()

# Setup logging
setup_logging()


def run_interactive_demo() -> None:
    """Run an interactive demo where the user can chat with the agents."""
    print("=" * 80)
    print("A2A Protocol Interactive Demo")
    print("=" * 80)
    print("\nInitializing agents...")

    # Create agents
    gpt45_agent = GPT45Agent(name="GPT-4.5 Assistant")
    gpt_o3_mini_agent = GPTO3MiniAgent(name="GPT-O3 Mini Assistant")

    # Create conversation manager
    manager = ConversationManager()
    conversation = manager.create_conversation(
        participant_ids=[gpt45_agent.id, gpt_o3_mini_agent.id],
        metadata={"type": "interactive"}
    )

    print("\nAgents ready:")
    print(f"1. {gpt45_agent.name} (ID: {gpt45_agent.id[:8]}...)")
    print(f"2. {gpt_o3_mini_agent.name} (ID: {gpt_o3_mini_agent.id[:8]}...)")

    # Select an agent to interact with
    agent_choice = None
    while agent_choice not in ["1", "2"]:
        agent_choice = input("\nSelect an agent to chat with (1 or 2): ")

    active_agent = gpt45_agent if agent_choice == "1" else gpt_o3_mini_agent
    print(f"\nYou are now chatting with {active_agent.name}")
    print("Type 'exit' to quit, 'switch' to change agents, or 'help' for more commands")
    print("-" * 80)

    # Start interactive loop
    while True:
        # Get user input
        user_input = input("\nYou: ")

        # Handle commands
        if user_input.lower() == "exit":
            print("\nExiting chat. Goodbye!")
            break
        elif user_input.lower() == "switch":
            active_agent = gpt_o3_mini_agent if active_agent.id == gpt45_agent.id else gpt45_agent
            print(f"\nSwitched to {active_agent.name}")
            continue
        elif user_input.lower() == "help":
            print("\nCommands:")
            print("  exit - Quit the program")
            print("  switch - Switch to the other agent")
            print("  help - Show this help message")
            print("  history - Show conversation history")
            continue
        elif user_input.lower() == "history":
            history = manager.get_conversation_history(conversation.id)
            print("\nConversation history:")
            for i, msg in enumerate(history):
                if msg.sender_id != "system":
                    sender = "You" if msg.sender_id not in [gpt45_agent.id, gpt_o3_mini_agent.id] else \
                        (gpt45_agent.name if msg.sender_id == gpt45_agent.id else gpt_o3_mini_agent.name)
                    for content in msg.content:
                        if content.type == ContentType.TEXT:
                            print(f"{i+1}. {sender}: {content.value[:50]}...")
            continue

        # Create a message from the user to the agent
        user_message = active_agent.create_message(
            recipient_id=active_agent.id,
            content_value=user_input,
            content_type=ContentType.TEXT
        )

        # Add the message to the conversation
        manager.add_message_to_conversation(conversation.id, user_message)

        # Have the agent process the message and generate a response
        active_agent.receive_message(user_message)
        response = active_agent.generate_response(user_message)

        # Add the response to the conversation
        manager.add_message_to_conversation(conversation.id, response)

        # Display the response
        print(f"\n{active_agent.name}: {response.content[0].value}")


def run_basic_demo() -> None:
    """Run the basic demo."""
    print("Running basic demo...")
    # Import here to avoid circular imports
    from main import main
    main()


def run_advanced_demo() -> None:
    """Run the advanced demo with function calling."""
    print("Running advanced demo...")
    # Import here to avoid circular imports
    from advanced_demo import FunctionCallingDemo
    demo = FunctionCallingDemo()
    demo.run()


def run_collaboration_demo() -> None:
    """Run the collaboration demo with multiple agents."""
    print("Running collaboration demo...")
    # Import here to avoid circular imports
    from collaboration_demo import MultiAgentCollaborationDemo
    demo = MultiAgentCollaborationDemo()
    demo.run()


def main() -> None:
    parser = argparse.ArgumentParser(description="A2A Protocol Demo CLI")
    parser.add_argument(
        "demo_type",
        choices=["basic", "advanced", "interactive", "collaboration"],
        nargs="?",
        default="interactive",
        help="Type of demo to run"
    )

    args = parser.parse_args()

    if args.demo_type == "basic":
        run_basic_demo()
    elif args.demo_type == "advanced":
        run_advanced_demo()
    elif args.demo_type == "collaboration":
        run_collaboration_demo()
    else:
        run_interactive_demo()


if __name__ == "__main__":
    main()

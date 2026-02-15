import json
import logging
import os
import sys
import traceback
from typing import Any, Dict, List

from dotenv import load_dotenv

# Configure logging with more detailed format
logging.basicConfig(
    level=logging.DEBUG,  # Changed to DEBUG level for more details
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import OpenAI client classes with proper error handling
try:
    import openai
    from openai import AzureOpenAI, OpenAI
    logger.debug(f"Successfully imported OpenAI module version: {openai.__version__}")
    OPENAI_AVAILABLE = True
except ImportError as e:
    logger.warning(f"OpenAI Python library not installed: {str(e)}. Using mock implementation.")
    OPENAI_AVAILABLE = False


# Mock classes for development purposes
class MockCompletionResponse:
    """Mock response for development purposes to avoid API calls"""
    def __init__(self, content):
        self.content = content
        self.function_call = None
        self.tool_calls = []


class MockCompletionChoice:
    """Mock choice object for development"""
    def __init__(self, content):
        self.message = MockCompletionResponse(content)


class MockCompletionResult:
    """Mock completion result for development"""
    def __init__(self, content):
        self.choices = [MockCompletionChoice(content)]


class MockChatCompletions:
    """Mock chat completions class"""
    def create(self, model, messages, temperature=0.7, max_tokens=500, **kwargs):
        """Mock create method that returns predefined responses based on the input"""
        # Extract the last message content if available
        last_message = None
        for m in reversed(messages):
            if m.get("content"):
                last_message = m.get("content")
                break

        # Generate predefined responses based on content keywords
        if last_message and ("Agent-to-Agent Protocol" in last_message or "A2A" in last_message):
            response = """
The Agent-to-Agent (A2A) Protocol is a standardized communication framework enabling AI agents to interact effectively.

Key components include:
1. Message structure with sender/recipient information
2. Content types (text, JSON, binary, etc.)
3. Function calling mechanisms for agent capabilities
4. Conversation session management

The protocol facilitates seamless collaboration between different agent systems.
"""
        elif last_message and "how message exchange works" in last_message.lower():
            response = """
Message exchange in the A2A protocol works through standardized formats:

1. Each message has a unique identifier, sender ID, recipient ID, and timestamp
2. Messages can contain multiple content items of different types
3. Messages can reference previous messages through the in_reply_to field
4. Agents acknowledge receipt and process messages according to their capabilities

This standardization ensures reliable communication between diverse agent systems.
"""
        elif last_message and "key components" in last_message.lower():
            response = """
Thank you for your question about the A2A protocol. Based on your research needs, here are the key points:

1. Core A2A Principles: Standardization, interoperability, security, and extensibility

2. Message Exchange: Uses structured JSON messages with required fields (IDs, content) and optional metadata. Messages flow through secure channels with delivery confirmation.

3. Standardized Schemas: Essential for cross-agent compatibility, enabling agents from different developers to communicate without custom integration work.

Would you like me to elaborate on any specific aspect?
"""
        elif last_message:
            response = f"This is a mock response for demonstration purposes. I received: {last_message[:100]}..."
        else:
            response = "This is a default mock response for demonstration purposes."

        return MockCompletionResult(response.strip())


class MockOpenAIClient:
    """Mock OpenAI client for development purposes"""
    def __init__(self):
        """Initialize with mock chat completions"""
        self.chat = type('obj', (object,), {
            'completions': MockChatCompletions()
        })


class OpenAIModel:
    """Base class for OpenAI model wrappers"""

    def __init__(self, model_name: str, config_prefix: str = None):
        """
        Initialize the model with specific configurations.

        Args:
            model_name: The name of the model to use
            config_prefix: The prefix for environment variables (e.g., "GPT45" or "O3_MINI")
        """
        # Make sure to load environment variables from .env file
        load_dotenv()

        self.model_name = model_name
        self.use_mock = False

        # Log environment information for debugging
        logger.debug(f"Python version: {sys.version}")
        if OPENAI_AVAILABLE:
            logger.debug(f"OpenAI SDK version: {openai.__version__}")

        # Skip real API initialization if OpenAI is not available
        if not OPENAI_AVAILABLE:
            self.client = MockOpenAIClient()
            self.use_mock = True
            logger.info(f"Using mock OpenAI model for {model_name}")
            return

        # Handle proxy environment variables which may cause issues with OpenAI 1.7.2
        proxy_vars = {}
        for var in ["http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"]:
            if var in os.environ:
                proxy_vars[var] = os.environ.pop(var)
                logger.debug(f"Temporarily removed proxy environment variable: {var}")

        # Try to use actual OpenAI API with new client classes
        try:
            # Set up Azure OpenAI if configured
            use_azure = os.getenv("USE_AZURE_OPENAI", "false").lower() == "true"

            if use_azure:
                # Get Azure OpenAI configuration
                if config_prefix:
                    endpoint = os.getenv(f"AZURE_OPENAI_ENDPOINT_{config_prefix}", "")
                    api_key = os.getenv(f"AZURE_OPENAI_KEY_{config_prefix}", "")
                    self.deployment = os.getenv(f"AZURE_OPENAI_DEPLOYMENT_{config_prefix}", model_name)
                else:
                    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "")
                    api_key = os.getenv("AZURE_OPENAI_KEY", "")
                    self.deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", model_name)

                api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")

                # Check if configuration is valid
                if not endpoint or not api_key:
                    raise ValueError(f"Azure OpenAI configuration is incomplete: endpoint={endpoint != ''}, api_key={api_key != ''}")

                logger.debug(f"Initializing Azure OpenAI client with endpoint: {endpoint}, api_version: {api_version}")

                # Create Azure OpenAI client with OpenAI SDK 1.7.2 compatible initialization
                try:
                    # Initialize with only the required parameters for version 1.7.2
                    self.client = AzureOpenAI(
                        api_key=api_key,
                        azure_endpoint=endpoint,
                        api_version=api_version  # Explicitly pass api_version
                    )
                    logger.debug("Azure OpenAI client initialized successfully")
                except TypeError as e:
                    # If initialization fails, try fallback without proxies parameter
                    logger.debug(f"First attempt failed: {str(e)}")
                    # Try the most minimal initialization possible
                    try:
                        # Set environment variables as alternative way to pass credentials
                        os.environ["AZURE_OPENAI_API_KEY"] = api_key
                        os.environ["AZURE_OPENAI_ENDPOINT"] = endpoint
                        # Set OPENAI_API_VERSION which is required by the SDK
                        os.environ["OPENAI_API_VERSION"] = api_version
                        self.client = AzureOpenAI()
                        logger.debug("Azure OpenAI client initialized with environment variables")
                    except Exception as e2:
                        logger.error(f"Failed to initialize Azure OpenAI client with fallback: {str(e2)}")
                        raise
                except Exception as e:
                    logger.error(f"Failed to initialize Azure OpenAI client: {str(e)}")
                    raise

                logger.info(f"Using Azure OpenAI for {model_name} with deployment {self.deployment}")
                self.use_azure = True
            else:
                # Standard OpenAI configuration
                api_key = os.getenv("OPENAI_API_KEY", "")

                if not api_key:
                    raise ValueError("OpenAI API key not found")

                logger.debug("Initializing standard OpenAI client")

                # Create OpenAI client with just the essential parameter
                try:
                    # Initialize with only the required parameter for version 1.7.2
                    self.client = OpenAI(api_key=api_key)
                    logger.debug("Standard OpenAI client initialized successfully")
                except Exception as e:
                    logger.error(f"Failed to initialize standard OpenAI client: {str(e)}")
                    raise

                logger.info(f"Using standard OpenAI for {model_name}")
                self.use_azure = False

            # Test the client with a simple request
            logger.debug("Testing OpenAI client with a simple request...")
            try:
                # For Azure OpenAI
                if self.use_azure:
                    model_id = self.deployment
                else:
                    model_id = self.model_name

                # Log successful client initialization
                logger.debug(f"Initialized client successfully with model: {model_id}")

            except Exception as test_err:
                logger.warning(f"Client test failed: {str(test_err)}")

        except Exception as e:
            logger.error("Error initializing OpenAI client:")
            logger.error(f"Error type: {type(e).__name__}")
            logger.error(f"Error message: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")

            # Fall back to mock implementation
            self.client = MockOpenAIClient()
            self.use_mock = True
            logger.info(f"Using mock OpenAI model for {model_name}")

        # Restore proxy environment variables
        for var, value in proxy_vars.items():
            os.environ[var] = value

    def generate_text(self, messages: List[Dict[str, str]], temperature: float = 0.7, max_tokens: int = 500) -> str:
        """
        Generate text using the OpenAI model.

        Args:
            messages: List of message dictionaries with "role" and "content" keys
            temperature: Controls randomness (0-1)
            max_tokens: Maximum number of tokens to generate

        Returns:
            The generated text response
        """
        try:
            if self.use_mock:
                # Use mock implementation
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                return response.choices[0].message.content
            else:
                # Use new client API style (compatible with openai>=1.0.0)
                # Only pass parameters that are explicitly supported
                completion_kwargs = {
                    "messages": messages,
                }

                # Handle parameter compatibility for specific models
                is_o3_mini = (hasattr(self, 'model_name') and self.model_name == "gpt-35-turbo") or \
                            (hasattr(self, 'deployment') and self.deployment == "o3-mini")

                # Handle max_tokens parameter compatibility
                if is_o3_mini:
                    # For GPT-4o-mini (gpt-35-turbo), use max_completion_tokens
                    completion_kwargs["max_completion_tokens"] = max_tokens
                    logger.debug(f"Using max_completion_tokens={max_tokens} for model: {getattr(self, 'deployment', self.model_name)}")
                    # Note: o3-mini doesn't support temperature parameter in this version
                else:
                    # For other models, use regular parameters
                    completion_kwargs["max_tokens"] = max_tokens
                    completion_kwargs["temperature"] = temperature

                # For Azure, use deployment name as the model
                if self.use_azure:
                    completion_kwargs["model"] = self.deployment
                else:
                    completion_kwargs["model"] = self.model_name

                response = self.client.chat.completions.create(**completion_kwargs)
                return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error generating text: {str(e)}")
            return f"Error: {str(e)}"

    def generate_with_function_calling(
        self,
        messages: List[Dict[str, str]],
        functions: List[Dict[str, Any]],
        temperature: float = 0.7
    ) -> Dict[str, Any]:
        """
        Generate text with function calling capability.

        Args:
            messages: List of message dictionaries
            functions: List of function definitions
            temperature: Controls randomness (0-1)

        Returns:
            Dict containing the response or function call
        """
        try:
            if self.use_mock:
                # Use mock implementation
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=temperature
                )
                return {
                    "type": "message",
                    "content": response.choices[0].message.content
                }
            else:
                # Convert functions to tools format for newer API
                tools = []
                for func in functions:
                    tools.append({
                        "type": "function",
                        "function": {
                            "name": func["name"],
                            "description": func.get("description", ""),
                            "parameters": func.get("parameters", {})
                        }
                    })

                # Only pass parameters that are explicitly supported
                completion_kwargs = {
                    "messages": messages,
                    "temperature": temperature,
                    "tools": tools,
                    "tool_choice": "auto"
                }

                # For Azure, use deployment name as the model
                if self.use_azure:
                    completion_kwargs["model"] = self.deployment
                else:
                    completion_kwargs["model"] = self.model_name

                response = self.client.chat.completions.create(**completion_kwargs)

                # Check if the model decided to call a function
                if response.choices[0].message.tool_calls:
                    tool_call = response.choices[0].message.tool_calls[0]
                    return {
                        "type": "function_call",
                        "function_name": tool_call.function.name,
                        "arguments": json.loads(tool_call.function.arguments)
                    }
                else:
                    return {
                        "type": "message",
                        "content": response.choices[0].message.content
                    }
        except Exception as e:
            logger.error(f"Error with function calling: {str(e)}")
            return {"type": "error", "content": str(e)}

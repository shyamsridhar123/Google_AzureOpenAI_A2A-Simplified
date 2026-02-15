# 🤝 Agent-to-Agent (A2A) Protocol Demo

[![CI](https://github.com/shyamsridhar123/Google_AzureOpenAI_A2A-Simplified/actions/workflows/ci.yml/badge.svg)](https://github.com/shyamsridhar123/Google_AzureOpenAI_A2A-Simplified/actions/workflows/ci.yml)
[![Protocol](https://img.shields.io/badge/Protocol-Google_A2A-4285F4)](https://github.com/google/A2A)
[![Models](https://img.shields.io/badge/Models-Azure_OpenAI-0078D4)](https://learn.microsoft.com/en-us/azure/ai-services/openai/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

This project demonstrates an implementation of the Google Agent-to-Agent (A2A) protocol using Azure OpenAI models GPT 4.1, GPT-4.5 and o3-Mini.

## 📋 Overview

The Agent-to-Agent (A2A) protocol is a standardized communication protocol defined by Google that enables autonomous AI agents to communicate, coordinate, and collaborate effectively. This demo showcases how different AI models can interact in a structured way using this protocol, leveraging Azure OpenAI services for the underlying intelligence.

The implementation provides multiple demo modes to demonstrate different aspects of agent communication and collaboration:
- **✨ Standard Mode**: Basic agent-to-agent communication
- **🤝 Collaborative Mode**: Multiple specialized agents working together on complex tasks
- **👤 Interactive Mode**: Human-in-the-loop interaction with A2A-compatible agents

## Architecture

```mermaid
flowchart TD
    %% Core Components
    Schemas["schemas/base.py<br/>A2A Protocol Schemas"] --> BaseAgent["agents/base_agent.py<br/>Base Agent Class"]
    Schemas --> ConvMgr["utils/conversation.py<br/>Conversation Manager"]
    BaseAgent --> ModelAgents["agents/model_agents.py<br/>GPT45, GPT41, O3Mini Agents"]
    
    %% Azure Integration
    OpenAIAPI["models/openai_model.py<br/>Azure OpenAI API"] --> ModelImpl["models/model_implementations.py<br/>Model-Specific Implementations"]
    ModelImpl --> ModelAgents
    
    %% Demo Application
    ModelAgents --> Demo["google_a2a_azure_demo.py<br/>Main Demo Application"]
    ConvMgr --> Demo
    
    %% Demo Modes
    Demo --> |Standard Mode| TwoAgents["Two-Agent Communication"]
    Demo --> |Collaborative Mode| ThreeAgents["Multi-Agent Workflow"]
    Demo --> |Interactive Mode| UserAgent["Human-Agent Interaction"]
    
    %% Message Flow Highlights
    Agent1(["Research Specialist"]) -- "create_message()" --> Msg[["A2A Protocol Message"]]
    Msg -- "receive_message()" --> Agent2(["Information Navigator"])
    Agent2 -- "generate_response()" --> Resp[["Response Message"]]
    Resp -- "add_to_conversation()" --> Conv[("Managed Conversation")]
    
    %% Styling with standard, high-contrast colors
    classDef core fill:#d4f1f9,stroke:#0275d8,stroke-width:2px,color:#000000
    classDef azure fill:#e8f4f8,stroke:#5bc0de,stroke-width:1px,color:#000000
    classDef demo fill:#ffeeba,stroke:#f0ad4e,stroke-width:2px,color:#000000
    classDef flow fill:#dff0d8,stroke:#5cb85c,stroke-width:1px,color:#000000
    
    class Schemas,BaseAgent,ConvMgr core
    class OpenAIAPI,ModelImpl azure
    class Demo,TwoAgents,ThreeAgents,UserAgent demo
    class Agent1,Msg,Agent2,Resp,Conv flow
```

## ✨ Features

- **🔄 A2A Protocol Implementation**
  - Core protocol components with standardized schemas
  - Structured message format with support for multiple content types (text, data, file)
  - Task management with state transitions (submitted, working, completed, etc.)
  - A2A-compliant agent cards and skill definitions

- **☁️ Azure OpenAI Integration**
  - Seamless integration with Azure OpenAI Services
  - Support for multiple Azure models: GPT-4.5, GPT-4.1, and GPT-O3 Mini
  - Configurable model parameters and deployment options
  - Secure API key management via environment variables

- **🚀 Demonstration Capabilities**
  - Three demo modes: Standard, Collaborative, and Interactive
  - Multi-agent conversations with coordinated workflows
  - Function calling between agents for enhanced capabilities
  - Human-in-the-loop interaction in Interactive mode

- **💻 Developer Experience**
  - Rich terminal visualization of agent interactions
  - Extensive logging for debugging and analysis
  - Clear separation of protocol and implementation layers
  - Practical code examples for educational purposes

## Project Structure

```
├── agents/                  # Agent implementations
│   ├── base_agent.py        # Base agent class implementing A2A protocol
│   └── model_agents.py      # Model-specific agent implementations
├── models/                  # Model wrappers
│   ├── openai_model.py      # Base Azure OpenAI model wrapper
│   └── model_implementations.py  # Specific model implementations
├── schemas/                 # A2A protocol schemas
│   └── base.py              # Core schema definitions
├── utils/                   # Utility functions
│   └── conversation.py      # Conversation management
├── google_a2a_azure_demo.py # Google A2A protocol demo with Azure OpenAI
├── server.py                # Simple A2A protocol server implementation
├── requirements.txt         # Project dependencies
└── .env                     # Environment variables (API keys)
```

## 🛠️ Setup

1. Make sure you have Python 3.8+ installed
2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

3. Configure your `.env` file in the project root directory with your Azure OpenAI API keys:

```
# Azure OpenAI GPT 4.5 Configuration
AZURE_OPENAI_ENDPOINT_GPT45="https://your-resource-name.openai.azure.com"
AZURE_OPENAI_KEY_GPT45="your-api-key-here"
AZURE_OPENAI_MODEL_GPT45="gpt-4.5-preview"
AZURE_OPENAI_DEPLOYMENT_GPT45="gpt-4.5-preview"
AZURE_OPENAI_API_VERSION="2025-03-01-preview"

# Azure OpenAI GPT-O3 Mini Configuration
AZURE_OPENAI_ENDPOINT_O3_MINI="https://your-resource-name.openai.azure.com"
AZURE_OPENAI_KEY_O3_MINI="your-api-key-here"
AZURE_OPENAI_MODEL_O3_MINI="gpt-o3-mini"
AZURE_OPENAI_DEPLOYMENT_O3_MINI="gpt-o3-mini"

# Azure OpenAI GPT-4.1 Configuration (optional for collaborative mode)
AZURE_OPENAI_ENDPOINT_GPT41="https://your-resource-name.openai.azure.com"
AZURE_OPENAI_KEY_GPT41="your-api-key-here"
AZURE_OPENAI_MODEL_GPT41="gpt-4.1-preview"
AZURE_OPENAI_DEPLOYMENT_GPT41="gpt-4.1-preview"

# Logging Configuration (optional)
LOG_LEVEL="INFO"  # Options: DEBUG, INFO, WARNING, ERROR, CRITICAL
LOG_FORMAT="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
```

> **Note:** You can obtain these values from the Azure Portal under your Azure OpenAI resource. Make sure you have deployed the appropriate models in your Azure OpenAI resource.

## 🚀 Running the Demos

### Google A2A Azure Demo

This is the primary demo showcasing the Google A2A protocol implementation with Azure OpenAI models. It supports three different modes:

```bash
# Standard mode - demonstrates basic agent communication
python google_a2a_azure_demo.py --mode standard

# Collaborative mode - demonstrates multi-agent collaboration
python google_a2a_azure_demo.py --mode collaborative

# Interactive mode - allows user interaction with agents
python google_a2a_azure_demo.py --mode interactive
```

## 🎮 Demo Modes Explained

### ✨ Standard Mode
Demonstrates the core A2A protocol concepts between two agents (Research Specialist and Information Navigator). The agents exchange messages with both text and structured data parts, showcasing how different content types can be combined in A2A messages.

### 🤝 Collaborative Mode
Shows three specialized agents (Research Specialist, Planning Strategist, Implementation Expert) working together on a complex problem. This demonstrates how the A2A protocol enables multi-agent collaboration by providing a standardized way for agents with different capabilities to build on each other's work.

### 👤 Interactive Mode
Places the user in the conversation loop with A2A-compatible agents. This demonstrates how humans can interact with A2A agents, including special commands for requesting deeper analysis or multi-modal responses.

## ☁️ Azure OpenAI Integration

This implementation specifically integrates with Azure OpenAI services to provide:

- 🔒 Secure access to powerful AI models hosted in Azure
- 🌎 Regional availability and data residency compliance
- 🏢 Enterprise-grade reliability and scalability
- 🧠 Support for multiple Azure OpenAI models (GPT-4.5, GPT-4.1, o3-Mini)

## 🔄 Comparison with Model Context Protocol (MCP)

While both A2A and MCP are designed to standardize AI interactions, they serve different purposes and operate at different levels:

### 🔌 What is Model Context Protocol (MCP)?

The Model Context Protocol (MCP) is an open protocol developed by Anthropic that standardizes how applications provide context to LLMs and connect them to different data sources and tools. Often described as a "USB-C port for AI applications," MCP enables seamless integration between LLM applications and external data sources or services.

MCP follows a client-server architecture where:
- **🖥️ MCP Hosts**: Programs like Claude Desktop, IDEs, or AI tools that want to access data through MCP
- **🔗 MCP Clients**: Protocol clients that maintain connections with servers
- **⚙️ MCP Servers**: Lightweight programs that expose specific capabilities through the standardized protocol
- **📊 Data Sources**: Local or remote data that MCP servers can securely access

### 📊 Key Differences from MCP

| Feature | 🤝 A2A Protocol | 🔌 MCP Protocol |
|---------|----------------|----------------|
| **Purpose** | Agent collaboration and workflow orchestration | Standardized context provision to LLMs |
| **Architecture** | Multi-agent communication framework | Client-server data access pattern |
| **Focus** | Agent capabilities, tasks, and artifacts | Resources, tools, and prompt templates |
| **Use Case** | Complex multi-agent coordination | Single LLM context enhancement |
| **Complexity** | Higher (agent orchestration, tasks, skills) | Lower (standardized data interfaces) |

#### 1. **🎯 Scope and Purpose**:
   - **A2A**: Focuses on agent-to-agent communication and collaboration, enabling independent AI agents to work together on tasks
   - **MCP**: Acts as a "USB-C port for AI applications" - standardizing how applications provide context to LLMs and connect them to different data sources and tools

#### 2. **🏗️ Architecture Level**:
   - **A2A**: Higher-level protocol for orchestrating multiple autonomous agents in a collaborative workflow
   - **MCP**: Client-server architecture focused on standardizing context exchange between applications and language models

#### 3. **💬 Communication Pattern**:
   - **A2A**: Multi-directional communication (agent-to-agent, agent-to-user, agent-to-system) with emphasis on message exchange
   - **MCP**: Primarily connects host applications (clients) to data sources and tools (servers) to provide context to LLMs

#### 4. **🧩 Component Focus**:
   - **A2A**: Emphasizes agent capabilities, skills, conversation sessions, tasks, and artifacts
   - **MCP**: Emphasizes resources (exposing data/content), tools (performing actions), and prompts (reusable templates)

#### 5. **⚙️ Implementation Complexity**:
   - **A2A**: More complex with multi-agent orchestration, skill discovery, and task management
   - **MCP**: Focused on creating standardized connections to data sources and tools for LLM context

### 🔄 Potential for Integration

[![Integration](https://img.shields.io/badge/Integration-Complementary_Protocols-8A2BE2)](https://modelcontextprotocol.io)

A complementary approach might integrate both protocols:
- Use **🔌 MCP** for providing context and tools to individual agents (standardizing the data/tool connections)
- Use **🤝 A2A** for orchestrating collaboration between multiple agents (standardizing agent communication)

This implementation demonstrates how A2A can be used as a higher-level protocol for agent coordination, while MCP-like patterns could be implemented within each agent to standardize how they access external data and tools.

## 🔌 Extending the Demo

You can extend this demo by:

- Adding more agent types with different capabilities
- Implementing more complex function calling scenarios
- Creating a web interface for visualization
- Adding support for more content types (images, audio, etc.)
- Implementing A2A protocol extensions
- Integrating with Azure services like Azure AI Search or Azure Cognitive Services

## 📚 Resources

- [Google Agent-to-Agent Protocol](https://github.com/google/A2A)
- [Google A2A Documentation](https://google.github.io/A2A/)
- [Azure OpenAI Documentation](https://learn.microsoft.com/en-us/azure/ai-services/openai/)
- [Azure OpenAI API Reference](https://learn.microsoft.com/en-us/azure/ai-services/openai/reference)

## 📜 License

MIT

## ⚠️ Disclaimer

This project is intended for educational purposes only. It demonstrates a simplified and more practical implementation of the Google Agent-to-Agent protocol using the latest Azure OpenAI models. This implementation is not affiliated with or endorsed by Google, and it may deviate from the official protocol specification in certain aspects to prioritize practical demonstrations and ease of understanding. The code showcases how the protocol can be adapted and implemented in real-world scenarios with Azure OpenAI services.

While efforts have been made to adhere to the core principles of the A2A protocol, this project focuses on practical applications rather than complete specification compliance. Users interested in the official protocol should refer to the [Google A2A Repository](https://github.com/google/A2A) and [documentation](https://google.github.io/A2A/).
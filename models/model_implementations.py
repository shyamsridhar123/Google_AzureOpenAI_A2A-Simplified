from typing import List

from models.openai_model import OpenAIModel


class GPT45Model(OpenAIModel):
    """Wrapper for GPT-4.5 model (Azure: gpt-4-turbo)"""

    def __init__(self) -> None:
        # Changed model name to gpt-4-turbo which is Azure compatible
        # gpt-4.5-preview is automatically mapped to gpt-4-turbo deployment
        super().__init__(model_name="gpt-4-turbo", config_prefix="GPT45")
        self.capabilities: List[str] = ["text_generation", "function_calling", "advanced_reasoning"]
        self.description: str = "GPT-4 Turbo is a large language model with advanced reasoning capabilities."

    def get_capabilities(self) -> List[str]:
        return self.capabilities


class GPT41Model(OpenAIModel):
    """Wrapper for GPT-4.1 model (Azure: gpt-4-1106)"""

    def __init__(self) -> None:
        # Using gpt-4-1106 which is the Azure model name for GPT-4.1
        super().__init__(model_name="gpt-4-1106", config_prefix="GPT41")
        self.capabilities: List[str] = ["text_generation", "function_calling", "reasoning", "long_context"]
        self.description: str = "GPT-4.1 is a large language model with strong reasoning and long context handling."

    def get_capabilities(self) -> List[str]:
        return self.capabilities


class GPTO3MiniModel(OpenAIModel):
    """Wrapper for GPT-O3 Mini model (Azure: gpt-35-turbo)"""

    def __init__(self) -> None:
        # Changed model name to gpt-35-turbo which is Azure compatible
        # gpt-o3-mini is automatically mapped to gpt-35-turbo deployment
        super().__init__(model_name="gpt-35-turbo", config_prefix="O3_MINI")
        self.capabilities: List[str] = ["text_generation", "function_calling", "efficient_processing"]
        self.description: str = "GPT-3.5 Turbo is a compact, efficient language model for lighter workloads."

    def get_capabilities(self) -> List[str]:
        return self.capabilities

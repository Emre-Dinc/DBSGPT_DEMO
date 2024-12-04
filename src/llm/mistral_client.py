from ctransformers import AutoModelForCausalLM
from typing import Dict, Optional
from src.utils.path_utils import get_config_path
import yaml
import os


class MistralClient:
    def __init__(self, config_path: str = None):
        # Load configuration
        config_path = config_path or get_config_path()
        with open(config_path, "r") as file:
            self.config = yaml.safe_load(file)["model"]

        # Initialize model
        self.model = self._init_model()

    def _init_model(self) -> AutoModelForCausalLM:
        """Initialize the Mistral model"""
        model_path = os.path.abspath(self.config["path"])

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")

        return AutoModelForCausalLM.from_pretrained(
            model_path,
            model_type="mistral",
            context_length=self.config["context_length"],
            gpu_layers=50,
        )

    def _create_prompt(self, query: str, context: Optional[str] = None) -> str:
        """Create a well-structured prompt for the model"""
        # Base system prompt to encourage concise, direct responses
        system_prompt = "You are a helpful AI assistant. Provide direct, concise answers without disclaimers or apologies."

        if context:
            # If we have context, use it in a structured way
            prompt = f"""[INST] {system_prompt}

Context: {context}

Question: {query}

Provide a concise answer based on the context above. [/INST]"""
        else:
            # For direct questions, keep it simple
            prompt = f"""[INST] {system_prompt}

Question: {query}

Provide a concise answer in one line. [/INST]"""

        return prompt

    def generate_response(
        self,
        query: str,
        context: Optional[str] = None,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
    ) -> str:
        """Generate a response using the Mistral model"""
        # Create well-structured prompt
        prompt = self._create_prompt(query, context)

        # Generate response with appropriate parameters
        response = self.model(
            prompt,
            max_new_tokens=max_new_tokens or self.config.get("max_tokens", 2048),
            temperature=temperature or self.config.get("temperature", 0.7),
            top_p=top_p or self.config.get("top_p", 0.95),
            stop=["</s>", "[/INST]"],
        )

        # Clean up the response
        response = response.strip()
        for stop_token in ["</s>", "[/INST]"]:
            if response.endswith(stop_token):
                response = response[: -len(stop_token)].strip()

        return response

    def __del__(self):
        """Cleanup when object is destroyed"""
        pass

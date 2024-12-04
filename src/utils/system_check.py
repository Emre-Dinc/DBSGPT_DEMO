from typing import Dict, Tuple
import sys
import os

# Use absolute imports
from src.llm.mistral_client import MistralClient
from src.db.milvus_client import MilvusClient


class SystemCheck:
    @staticmethod
    def check_milvus() -> Tuple[bool, str]:
        """Test Milvus connection"""
        try:
            milvus_client = MilvusClient()
            return True, "Milvus connection successful!"
        except Exception as e:
            return False, f"Milvus connection failed: {str(e)}"

    @staticmethod
    def check_model() -> Tuple[bool, str, str]:
        """Test Mistral model loading and basic inference"""
        try:
            llm = MistralClient()
            test_query = "What is 2+2? Answer in one word."
            response = llm.generate_response(test_query)
            return True, "Model loaded successfully!", response
        except Exception as e:
            return False, f"Model loading failed: {str(e)}", ""

    @staticmethod
    def run_all_checks() -> Dict[str, Dict[str, any]]:
        """Run all system checks and return results"""
        results = {"milvus": {}, "model": {}}

        # Check Milvus
        milvus_success, milvus_message = SystemCheck.check_milvus()
        results["milvus"] = {
            "status": "✅" if milvus_success else "❌",
            "message": milvus_message,
        }

        # Check Model
        model_success, model_message, test_response = SystemCheck.check_model()
        results["model"] = {
            "status": "✅" if model_success else "❌",
            "message": model_message,
            "test_response": test_response if model_success else None,
        }

        return results

import unittest
import sys
import os
from src.utils.system_check import SystemCheck


class TestSystemSetup(unittest.TestCase):
    def setUp(self):
        """Setup test case"""
        self.system_check = SystemCheck()

    def test_milvus_connection(self):
        """Test if Milvus connection works"""
        success, message = self.system_check.check_milvus()
        self.assertTrue(success, f"Milvus connection failed: {message}")

    def test_model_loading(self):
        """Test if model loads and can generate responses"""
        success, message, response = self.system_check.check_model()
        self.assertTrue(success, f"Model loading failed: {message}")
        self.assertIsNotNone(response, "Model should generate a response")
        self.assertIsInstance(response, str, "Response should be a string")

    def test_complete_system(self):
        """Test entire system setup"""
        results = self.system_check.run_all_checks()

        # Check Milvus
        self.assertEqual(
            results["milvus"]["status"],
            "✅",
            f"Milvus check failed: {results['milvus']['message']}",
        )

        # Check Model
        self.assertEqual(
            results["model"]["status"],
            "✅",
            f"Model check failed: {results['model']['message']}",
        )


if __name__ == "__main__":
    # Print separator for better visibility
    print("\n" + "=" * 50)
    print("Running System Setup Tests")
    print("=" * 50 + "\n")

    # Run tests
    unittest.main(verbosity=2)

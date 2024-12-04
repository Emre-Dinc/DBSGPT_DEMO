from src.db.milvus_client import MilvusClient
from src.llm.mistral_client import MistralClient
from src.utils.path_utils import get_config_path
import time
import os
import traceback


def test_milvus():
    print("\nTesting Milvus Connection...")
    try:
        client = MilvusClient()
        print("✅ Milvus connection successful!")

        # Test insert
        test_embedding = [0.1] * 384
        client.insert(
            content="This is a test document",
            embedding=test_embedding,
            metadata={"source": "test", "type": "document"},
        )
        print("✅ Insert successful!")

        # Test search
        results = client.search(test_embedding)
        print("✅ Search successful!")
        print(f"Found {len(results)} results")
        if results:
            print(f"First result: {results[0]}")
        return True

    except Exception as e:
        print(f"❌ Milvus test failed: {str(e)}")
        print("Detailed error:")
        print(traceback.format_exc())
        return False


def test_mistral():
    print("\nTesting Mistral Model...")
    try:
        client = MistralClient()
        print("✅ Model loaded successfully!")

        # Test cases with expected concise responses
        test_cases = [
            ("What is 2+2? Answer in one word.", "Expected: A single number"),
            (
                "What color is the sky on a clear day? Answer in one word.",
                "Expected: A single color",
            ),
            (
                "Is Python a programming language? Answer yes or no.",
                "Expected: Yes/No only",
            ),
        ]

        for query, expected in test_cases:
            print(f"\nQuery: {query}")
            print(f"{expected}")

            start_time = time.time()
            response = client.generate_response(query)
            end_time = time.time()

            print(f"Response: {response}")
            print(f"Generation time: {end_time - start_time:.2f} seconds")

        # Test with context
        context = "The capital of France is Paris. It is known as the City of Light."
        query = "What is the capital of France?"
        print("\nTesting with context:")
        response = client.generate_response(query, context=context)
        print(f"Response: {response}")

        return True

    except Exception as e:
        print(f"❌ Mistral test failed: {str(e)}")
        print("Detailed error:")
        print(traceback.format_exc())
        return False


if __name__ == "__main__":
    config_path = get_config_path()
    print(f"Using config from: {config_path}")

    if not os.path.exists(config_path):
        print(f"❌ Config file not found at {config_path}")
        exit(1)

    print("Running quick tests...")
    milvus_success = test_milvus()
    mistral_success = test_mistral()

    if milvus_success and mistral_success:
        print("\n✅ All tests passed successfully!")
    else:
        print("\n❌ Some tests failed. Please check the output above.")

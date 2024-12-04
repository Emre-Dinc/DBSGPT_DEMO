from sentence_transformers import SentenceTransformer
from src.db.milvus_client import MilvusClient
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_milvus_data():
    # Initialize components
    client = MilvusClient()
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Test queries
    test_queries = [
        "exam schedule",
        "visa requirements",
        "library access",
        "medical services",
        "student card"
    ]

    print("\nTesting Milvus Search:")
    print("=" * 80)

    for query in test_queries:
        print(f"\nResults for query: '{query}'")
        print("-" * 40)

        # Generate embedding for query
        query_embedding = model.encode([query])[0].tolist()

        # Search
        results = client.search(query_embedding, limit=3)

        # Display results
        for i, result in enumerate(results, 1):
            print(f"\n{i}. Score: {result['score']:.4f}")
            print(f"Q: {result['content'].split('A:')[0].strip()[2:]}")  # Remove 'Q: ' prefix
            print(f"Section: {result['metadata']['section']}")
            print(f"Category: {result['metadata']['category']}")

    # Get collection stats
    stats = client.collection.num_entities
    print(f"\nTotal documents in collection: {stats}")
    print("=" * 80)


if __name__ == "__main__":
    test_milvus_data()
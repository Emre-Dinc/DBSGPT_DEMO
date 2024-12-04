import unittest
from pathlib import Path
from src.db.data_loader import DataLoader


class TestDataLoader(unittest.TestCase):
    def setUp(self):
        """Set up test environment"""
        self.test_dir = Path(__file__).parent
        self.processed_dir = self.test_dir / "test_data" / "processed"
        self.loader = DataLoader()

        print(f"\nLooking for processed PDFs in: {self.processed_dir}")
        print(f"Files found: {list(self.processed_dir.glob('*.json'))}")

    def test_load_single_file(self):
        """Test loading a single processed PDF"""
        # Find first PDF file
        pdf_files = list(self.processed_dir.glob("*.json"))
        if not pdf_files:
            self.skipTest("No processed PDFs available for testing")

        test_file = pdf_files[0]
        print(f"\nTesting with file: {test_file}")

        # Process and load the file
        self.loader.process_and_load(str(test_file))

        # Verify data was loaded by searching
        sample_query = "library"
        results = self.loader.milvus_client.search(
            self.loader.embedding_model.encode([sample_query])[0].tolist()
        )

        self.assertTrue(len(results) > 0, "No results found after loading data")
        print(f"\nFound {len(results)} results for test query")
        print("First result:", results[0])

    def test_load_directory(self):
        """Test loading all PDFs in directory"""
        self.loader.load_directory(str(self.processed_dir))

        # Verify data was loaded
        sample_queries = ["exam", "visa", "library", "medical"]
        for query in sample_queries:
            results = self.loader.milvus_client.search(
                self.loader.embedding_model.encode([query])[0].tolist()
            )
            print(f"\nResults for '{query}': {len(results)}")
            if results:
                print("Top result:", results[0])

            self.assertTrue(len(results) > 0, f"No results found for query '{query}'")


if __name__ == '__main__':
    unittest.main(verbosity=2)

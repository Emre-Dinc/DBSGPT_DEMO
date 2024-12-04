import unittest
from pathlib import Path
import json
from src.data_processing.text_chunker import TextChunker, Chunk


class TestTextChunker(unittest.TestCase):
    def setUp(self):
        """Set up test environment"""
        self.chunker = TextChunker()
        self.test_dir = Path(__file__).parent
        self.processed_dir = self.test_dir / "test_data" / "processed"
        print(f"\nLooking for FAQs in: {self.processed_dir}")

    def test_faq_processing(self):
        """Test processing of FAQ content"""
        # Load the existing FAQ file
        faq_file = self.processed_dir / "FAQs from Students at Dublin Business School.pdf.json"
        self.assertTrue(faq_file.exists(), f"FAQ file not found at {faq_file}")

        with open(faq_file, 'r') as f:
            processed_pdf = json.load(f)

        chunks = self.chunker.process_document(processed_pdf)

        # Basic validation
        self.assertTrue(len(chunks) > 0, "No chunks were created")
        print(f"\nTotal chunks created: {len(chunks)}")

        # Print all chunks for inspection
        print("\nAll chunks created:")
        for i, chunk in enumerate(chunks):
            print(f"\nChunk {i}:")
            print(f"Section: {chunk.metadata['section']}")
            print(f"Question: {chunk.metadata['question']}")
            print(f"Category: {chunk.metadata['category']}")
            if i < 3:  # Print full content only for first 3 chunks
                print(f"Content: {chunk.content[:200]}...")

        # Save all chunks for inspection
        output_dir = self.test_dir / "test_data" / "chunks"
        output_dir.mkdir(parents=True, exist_ok=True)

        for i, chunk in enumerate(chunks):
            output_file = output_dir / f"chunk_{i}.json"
            with open(output_file, 'w') as f:
                json.dump({
                    "content": chunk.content,
                    "metadata": chunk.metadata
                }, f, indent=2)

        # Content validation
        for i, chunk in enumerate(chunks):
            # Skip table of contents pages
            if "Click the question below" in chunk.content:
                continue

            self.assertIn("Q:", chunk.content, f"Question marker missing from chunk {i}")
            self.assertIn("A:", chunk.content, f"Answer marker missing from chunk {i}")
            0.8
            0.79

            parts = chunk.content.split("A:")
            question = parts[0].replace("Q:", "").strip()
            answer = parts[1].strip()

            # Question validation
            self.assertTrue(len(question) > 0, f"Empty question in chunk {i}")
            self.assertTrue(question.endswith("?") or "QUERIES" in question,
                            f"Invalid question format in chunk {i}: {question}")

            # Answer validation
            self.assertTrue(len(answer) > 10,
                            f"Answer too short in chunk {i} for question: {question}")

            # Metadata validation
            self.assertIn("section", chunk.metadata,
                          f"Missing section in chunk {i}")
            self.assertTrue(len(chunk.metadata["section"]) > 0,
                            f"Empty section in chunk {i}")


if __name__ == '__main__':
    unittest.main(verbosity=2)
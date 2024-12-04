import unittest
import unittest
from pathlib import Path
import json
import re
from typing import List, Dict, Any
from src.data_processing.text_chunker import TextChunker, Chunk


class TestTextChunker(unittest.TestCase):
    def setUp(self):
        """Set up test environment"""
        self.chunker = TextChunker()
        self.test_dir = Path(__file__).parent
        self.processed_dir = self.test_dir / "test_data" / "processed"
        self.chunks_dir = self.test_dir / "test_data" / "chunks"

        # Ensure test directories exist
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.chunks_dir.mkdir(parents=True, exist_ok=True)

    def test_process_faq(self):
        """Test processing of FAQ document"""
        # Find FAQ file
        faq_files = list(self.processed_dir.glob("*FAQ*.json"))
        self.assertTrue(faq_files, "No FAQ files found in the processed directory")
        faq_file = faq_files[0]

        print(f"\nProcessing file: {faq_file}")

        with open(faq_file, "r") as f:
            processed_pdf = json.load(f)

        # Process all pages to extract QA pairs
        chunks = []
        current_section = "general"

        for page in processed_pdf["pages"]:
            content = page["content"]
            print(
                f"Processing page content:\n{content[:500]}...\n{'-' * 40}"
            )  # Print first 500 chars for inspection

            # Skip table of contents page
            if "Click the question below to land on the answer you need" in content:
                print("Skipping table of contents page.")
                continue

            # Find section headers
            section_match = re.search(r"\n([A-Z][A-Za-z &]+)(?:\n|:)", content)
            if section_match:
                current_section = section_match.group(1).strip()
                print(f"Detected section: {current_section}")

            # Find question-answer pairs with multiline support
            qa_pattern = r"(?:^|\n)(.*?\?)\s*((?:(?!\n.*?\?).)*)"
            matches = list(re.finditer(qa_pattern, content, re.DOTALL))

            if not matches:
                print(f"No matches found for page content: {content[:500]}...")
                continue

            for match in matches:
                question = match.group(1).strip()
                answer = match.group(2).strip().replace("\n", " ")
                print(f"Matched Q: {question}")
                print(f"Matched A: {answer}")

                if len(answer) < 10:
                    print(f"Skipping short answer: {answer}")
                    continue

                chunks.append(
                    {"section": current_section, "question": question, "answer": answer}
                )

        # Save chunks to JSON file for verification
        output_file = self.chunks_dir / "faq_chunks.json"
        with open(output_file, "w") as f:
            json.dump(chunks, f, indent=4)

        print(f"Chunks saved to {output_file}")
        self.assertTrue(chunks, "No chunks were extracted from the FAQ document")


if __name__ == "__main__":
    unittest.main()

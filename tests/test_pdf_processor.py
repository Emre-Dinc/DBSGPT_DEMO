import unittest
import os
from src.data_processing.pdf_processor import PDFProcessor, ProcessedPDF
from pathlib import Path


class TestPDFProcessor(unittest.TestCase):
    def setUp(self):
        """Set up test environment"""
        # Get the absolute path to the test directory
        self.test_dir = Path(os.path.dirname(os.path.abspath(__file__)))
        self.test_input_dir = self.test_dir / "test_data" / "raw"
        self.test_output_dir = self.test_dir / "test_data" / "processed"

        # Create test directories if they don't exist
        self.test_input_dir.mkdir(parents=True, exist_ok=True)
        self.test_output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize processor with test directories
        self.processor = PDFProcessor(
            input_dir=str(self.test_input_dir), output_dir=str(self.test_output_dir)
        )

        # Print the actual path being used (for debugging)
        print(f"\nLooking for PDFs in: {self.test_input_dir}")
        # List all files in the directory (for debugging)
        print("Files found:", list(self.test_input_dir.glob("*.pdf")))

    def test_directories_creation(self):
        """Test if directories are created properly"""
        self.assertTrue(self.test_input_dir.exists())
        self.assertTrue(self.test_output_dir.exists())

    def test_pdf_processing(self):
        """Test PDF processing with a sample PDF"""
        # Find all PDF files in the test directory
        pdf_files = list(self.test_input_dir.glob("*.pdf"))

        if not pdf_files:
            self.skipTest("No sample PDF available for testing")

        # Test the first PDF found
        test_pdf_path = pdf_files[0]
        print(f"\nProcessing PDF: {test_pdf_path}")

        result = self.processor.process_single_pdf(str(test_pdf_path))

        # Check if processing was successful
        self.assertIsNotNone(result)
        self.assertIsInstance(result, ProcessedPDF)
        self.assertTrue(len(result.pages) > 0)

        # Check if output file was created
        output_file = self.test_output_dir / f"{result.filename}.json"
        self.assertTrue(output_file.exists())

    def test_directory_processing(self):
        """Test processing of multiple PDFs"""
        # Process all PDFs in the test directory
        results = self.processor.process_directory()

        # Get actual PDF count
        pdf_count = len(list(self.test_input_dir.glob("*.pdf")))
        print(f"\nFound {pdf_count} PDFs in directory")

        # Check if all PDFs were processed
        self.assertEqual(len(results), pdf_count)


if __name__ == "__main__":
    unittest.main(verbosity=2)

from typing import List, Dict, Optional
import os
import logging
from pathlib import Path
import pdfplumber
import json
from dataclasses import dataclass, asdict
from datetime import datetime


@dataclass
class PDFPage:
    """Represents a single page from a PDF document"""

    page_number: int
    content: str
    metadata: Dict


@dataclass
class ProcessedPDF:
    """Represents a processed PDF document"""

    filename: str
    total_pages: int
    pages: List[PDFPage]
    metadata: Dict
    processed_at: str


class PDFProcessor:
    def __init__(self, input_dir: str = "data/raw", output_dir: str = "data/processed"):
        """
        Initialize PDF processor

        Args:
            input_dir: Directory containing PDF files
            output_dir: Directory to save processed files
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self._setup_directories()
        self._setup_logging()

    def _setup_directories(self):
        """Create necessary directories if they don't exist"""
        self.input_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )
        self.logger = logging.getLogger(__name__)

    def process_single_pdf(self, pdf_path: str) -> Optional[ProcessedPDF]:
        """
        Process a single PDF file

        Args:
            pdf_path: Path to the PDF file

        Returns:
            ProcessedPDF object if successful, None otherwise
        """
        try:
            self.logger.info(f"Processing PDF: {pdf_path}")
            pdf_pages = []

            with pdfplumber.open(pdf_path) as pdf:
                # Extract basic metadata
                metadata = {
                    "title": os.path.basename(pdf_path),
                    "pages": len(pdf.pages),
                    "source_path": pdf_path,
                }

                # Process each page
                for page_num, page in enumerate(pdf.pages, 1):
                    text = page.extract_text()
                    if text:
                        # Extract page-specific metadata
                        page_metadata = {
                            "page_number": page_num,
                            "width": page.width,
                            "height": page.height,
                            "has_text": bool(text.strip()),
                        }

                        pdf_pages.append(
                            PDFPage(
                                page_number=page_num,
                                content=text,
                                metadata=page_metadata,
                            )
                        )

            # Create ProcessedPDF object
            processed_pdf = ProcessedPDF(
                filename=os.path.basename(pdf_path),
                total_pages=len(pdf_pages),
                pages=pdf_pages,
                metadata=metadata,
                processed_at=datetime.now().isoformat(),
            )

            # Save processed content
            self._save_processed_pdf(processed_pdf)

            self.logger.info(f"Successfully processed PDF: {pdf_path}")
            return processed_pdf

        except Exception as e:
            self.logger.error(f"Error processing PDF {pdf_path}: {str(e)}")
            return None

    def process_directory(self) -> List[ProcessedPDF]:
        """
        Process all PDFs in the input directory

        Returns:
            List of ProcessedPDF objects
        """
        processed_pdfs = []
        pdf_files = list(self.input_dir.glob("*.pdf"))

        self.logger.info(f"Found {len(pdf_files)} PDF files to process")

        for pdf_path in pdf_files:
            result = self.process_single_pdf(str(pdf_path))
            if result:
                processed_pdfs.append(result)

        self.logger.info(f"Successfully processed {len(processed_pdfs)} PDFs")
        return processed_pdfs

    def _save_processed_pdf(self, processed_pdf: ProcessedPDF):
        """Save processed PDF content to output directory"""
        output_file = self.output_dir / f"{processed_pdf.filename}.json"

        # Convert to dictionary format
        pdf_dict = asdict(processed_pdf)

        # Save to JSON file
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(pdf_dict, f, ensure_ascii=False, indent=2)

        self.logger.info(f"Saved processed content to {output_file}")

    def get_text_from_pages(self, processed_pdf: ProcessedPDF) -> str:
        """
        Get concatenated text from all pages

        Args:
            processed_pdf: ProcessedPDF object

        Returns:
            Concatenated text from all pages
        """
        return "\n\n".join([page.content for page in processed_pdf.pages])

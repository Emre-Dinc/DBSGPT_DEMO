from typing import List, Dict, Any
from dataclasses import dataclass
import re
import logging


@dataclass
class Chunk:
    """Represents a chunk of text with its metadata"""
    content: str
    metadata: Dict[str, Any]
RAG

class TextChunker:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.valid_sections = [
            "General Queries",
            "Library Queries",
            "International & VISA Queries",
            "Disabilities Office Queries",
            "Student Life Queries",
            "Medical Information",
            "FINANCE QUERIES"
        ]

    def _clean_text(self, text: str) -> str:
        """Clean text by removing headers and unnecessary formatting"""
        # Remove standard headers and footers
        text = re.sub(r'StudentFAQGuide\s*\nDublinBusinessSchool\s*\nLiveDocument\s*\n_{20,}', '', text)
        text = re.sub(r'\n_{20,}\n', '\n', text)
        # Remove page numbers
        text = re.sub(r'\s+\d+\s*$', '', text, flags=re.MULTILINE)
        return text.strip()

    def _split_into_sections(self, text: str) -> List[Dict[str, Any]]:
        """Split text into sections"""
        sections = []
        lines = text.split('\n')
        current_section = None
        current_content = []

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Check if line is a section header
            is_section = any(section in line for section in self.valid_sections)
            if is_section:
                # Save previous section
                if current_section and current_content:
                    sections.append({
                        'section': current_section,
                        'content': '\n'.join(current_content)
                    })
                # Start new section
                current_section = next((s for s in self.valid_sections if s in line), None)
                current_content = []
            else:
                current_content.append(line)

        # Add last section
        if current_section and current_content:
            sections.append({
                'section': current_section,
                'content': '\n'.join(current_content)
            })

        return sections

    def _extract_qa_pairs(self, content: str) -> List[Dict[str, str]]:
        """Extract question-answer pairs from content"""
        qa_pairs = []
        lines = content.split('\n')

        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if not line:
                i += 1
                continue

            # Check if line is a question
            is_question = (
                    line.endswith('?') or
                    line.startswith(
                        ('How', 'What', 'Where', 'When', 'Why', 'Who', 'Can', 'Will', 'Do', 'I', 'Which')) or
                    re.match(r'^[A-Z][^.!?]*\??$', line)
            )

            if is_question:
                question = line if line.endswith('?') else line + '?'
                answer_lines = []

                # Collect answer lines until next question or section
                i += 1
                while i < len(lines):
                    next_line = lines[i].strip()
                    if not next_line:
                        i += 1
                        continue

                    # Stop if we hit next question or section
                    if (next_line.endswith('?') or
                            next_line.startswith(
                                ('How', 'What', 'Where', 'When', 'Why', 'Who', 'Can', 'Will', 'Do', 'I', 'Which')) or
                            any(section in next_line for section in self.valid_sections)):
                        break

                    answer_lines.append(next_line)
                    i += 1

                answer = ' '.join(answer_lines).strip()
                if answer and len(answer) > 20:  # Minimum answer length
                    qa_pairs.append({
                        'question': question,
                        'answer': answer
                    })
            else:
                i += 1

        return qa_pairs

    def process_document(self, processed_pdf: Dict[str, Any]) -> List[Chunk]:
        """Process document into chunks"""
        # Skip table of contents pages
        content = "\n\n".join(page["content"] for page in processed_pdf["pages"][2:])

        # Clean text
        content = self._clean_text(content)

        # Split into sections
        sections = self._split_into_sections(content)

        # Process sections into chunks
        chunks = []
        chunk_index = 0

        for section in sections:
            qa_pairs = self._extract_qa_pairs(section['content'])

            for qa in qa_pairs:
                metadata = {
                    "chunk_index": chunk_index,
                    "source_file": processed_pdf["metadata"]["title"],
                    "document_type": "faq",
                    "question": qa['question'],
                    "section": section['section'],
                    "category": self._determine_category(qa['question'], qa['answer']),
                    "extracted_info": self._extract_metadata(qa['answer'])
                }

                chunks.append(Chunk(
                    content=f"Q: {qa['question']}\nA: {qa['answer']}",
                    metadata=metadata
                ))
                chunk_index += 1

        return chunks

    def _determine_category(self, question: str, answer: str) -> str:
        """Determine category of QA pair"""
        text = (question + " " + answer).lower()

        categories = {
            "academic": [
                r'\b(?:exam|lecture|module|assignment|grade|timetable|class|course|study|academic)\b',
                r'\b(?:programme|semester|assessment)\b'
            ],
            "administrative": [
                r'\b(?:student card|fee|document|letter|transcript|parchment)\b',
                r'\b(?:admin|registration|application|upload|process)\b'
            ],
            "international": [
                r'\b(?:visa|immigration|international|stamp 2|ppsn)\b',
                r'\b(?:foreign|overseas)\b'
            ],
            "facilities": [
                r'\b(?:library|computer|parking|room|campus|building)\b',
                r'\b(?:facility|access|equipment|space)\b'
            ],
            "student_life": [
                r'\b(?:club|society|event|social|experience)\b',
                r'\b(?:activity|sport|student|life)\b'
            ],
            "medical": [
                r'\b(?:sick|doctor|gp|medical|health|emergency)\b',
                r'\b(?:hospital|clinic|treatment|condition)\b'
            ],
            "technical": [
                r'\b(?:moodle|computer|online|access|password)\b',
                r'\b(?:login|technical|it|system|software)\b'
            ]
        }

        for category, patterns in categories.items():
            if any(re.search(pattern, text) for pattern in patterns):
                return category

        return "general"

    def _extract_metadata(self, text: str) -> Dict[str, List[str]]:
        """Extract metadata from text"""

        def clean_url(url: str) -> str:
            return url.rstrip('.,')

        return {
            "urls": [clean_url(url) for url in re.findall(
                r'(?:https?://[^\s<>"]+|www\.[^\s<>"]+|[a-zA-Z0-9-]+\.dbs\.ie(?:/[^\s<>"]*)?)',
                text
            )],
            "emails": re.findall(
                r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
                text
            ),
            "phone_numbers": re.findall(
                r'\b(?:\+\d{1,3}[-.\s]?)?\d{2,4}[-.\s]?\d{3,4}[-.\s]?\d{3,4}\b',
                text
            ),
            "locations": re.findall(
                r'(?:Room|Building|Floor|Campus|Street)\s+[A-Za-z0-9-]+(?:[.\s][A-Za-z0-9-]+)*',
                text,
                re.IGNORECASE
            ),
            "deadlines": re.findall(
                r'\b\d{1,2}(?:st|nd|rd|th)?\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}\b',
                text
            ),
            "fees": re.findall(r'€\d+(?:[.,]\d{2})?', text)
        }
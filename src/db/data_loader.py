from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
from src.db.milvus_client import MilvusClient
from pathlib import Path
import json
import logging
import yaml


class ChunkLoader:
    def __init__(self, config_path: str = None):
        # Setup logging
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)

        # Load configuration
        config_path = config_path or "config/config.yaml"
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Initialize components
        self.embedding_model = SentenceTransformer(
            self.config['embedding']['model_name']
        )
        self.milvus_client = MilvusClient(config_path)

    def reconnect_milvus(self):
        """Create fresh Milvus connection"""
        self.milvus_client = MilvusClient()

    def load_chunks(self, chunks_dir: str) -> None:
        """Load all chunk files from directory"""
        chunks_dir = Path(chunks_dir)
        chunk_files = list(chunks_dir.glob("chunk_*.json"))
        total_chunks = len(chunk_files)

        self.logger.info(f"Found {total_chunks} chunk files to load")

        # Process chunks in batches
        batch_size = self.config['embedding']['batch_size']
        current_batch = []

        for i, chunk_file in enumerate(chunk_files, 1):
            try:
                with open(chunk_file, 'r') as f:
                    chunk_data = json.load(f)
                    current_batch.append(chunk_data)

                if len(current_batch) >= batch_size or i == total_chunks:
                    self._process_batch(current_batch)
                    self.logger.info(f"Processed chunks {i - len(current_batch) + 1}-{i} of {total_chunks}")
                    current_batch = []

            except Exception as e:
                self.logger.error(f"Error processing {chunk_file}: {e}")
                continue

    def _process_batch(self, chunk_batch: List[Dict]) -> None:
        """Process and insert a batch of chunks"""
        # Prepare texts for embedding
        texts = []
        valid_chunks = []

        for chunk in chunk_batch:
            content = chunk["content"]
            if not content.startswith('Q:'):
                continue

            qa_parts = content.split('\nA:', 1)
            if len(qa_parts) != 2:
                continue

            question = qa_parts[0][2:].strip()  # Remove 'Q: '
            answer = qa_parts[1].strip()

            if len(answer) < 10:  # Skip chunks with very short answers
                continue

            texts.append(f"{question} {answer}")
            valid_chunks.append(chunk)

        if not texts:
            return

        # Generate embeddings
        embeddings = self.embedding_model.encode(texts)

        # Insert each chunk with its embedding
        for chunk, embedding in zip(valid_chunks, embeddings):
            try:
                self.milvus_client.insert(
                    content=chunk["content"],
                    embedding=embedding.tolist(),
                    metadata=chunk["metadata"]
                )
            except Exception as e:
                self.logger.error(f"Error inserting chunk: {e}")
                # Try to reconnect and retry once
                try:
                    self.reconnect_milvus()
                    self.milvus_client.insert(
                        content=chunk["content"],
                        embedding=embedding.tolist(),
                        metadata=chunk["metadata"]
                    )
                except Exception as e2:
                    self.logger.error(f"Retry failed: {e2}")


if __name__ == "__main__":
    # Create loader
    loader = ChunkLoader()

    # Drop existing collection if needed
    response = input("Drop existing collection? (y/n): ").lower()
    if response == 'y':
        loader.milvus_client.collection.drop()
        loader.logger.info("Dropped existing collection")
        loader.reconnect_milvus()
        loader.logger.info("Created new collection")

    # Load chunks
    chunks_dir = Path(__file__).parent.parent.parent / "tests" / "test_data" / "chunks"
    loader.logger.info(f"Loading chunks from: {chunks_dir}")
    loader.load_chunks(str(chunks_dir))
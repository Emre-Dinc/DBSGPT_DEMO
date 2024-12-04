from typing import List, Dict, Any
from src.db.milvus_client import MilvusClient
from src.llm.mistral_client import MistralClient
from sentence_transformers import SentenceTransformer
import logging


class QueryHandler:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.logger = logging.getLogger(__name__)
        self.embedding_model = SentenceTransformer(model_name)
        self.milvus_client = MilvusClient()
        self.mistral_client = MistralClient()

    def process_query(self, query: str, top_k: int = 3) -> Dict[str, Any]:
        try:
            query_embedding = self.embedding_model.encode([query])[0].tolist()

            search_results = self.milvus_client.search(
                query_embedding=query_embedding,
                limit=top_k,
                query=query
            )

            # Format context for Mistral
            context = self._format_context(search_results)

            # Generate response with Mistral
            response = self.mistral_client.generate_response(
                query=query,
                context=context,  # Pass as context parameter
                max_new_tokens=None,  # Use default from config
                temperature=None,  # Use default from config
                top_p=None  # Use default from config
            )

            return {
                'query': query,
                'response': response,
                'sources': self._format_sources(search_results)
            }

        except Exception as e:
            self.logger.error(f"Error processing query: {e}")
            raise

    def _format_context(self, search_results: List[Dict]) -> str:
        contexts = []
        for result in search_results:
            content = result['content']
            if 'Q:' in content and 'A:' in content:
                contexts.append(content)

        return "\n\n".join(contexts)

    def _format_sources(self, search_results: List[Dict]) -> List[Dict]:
        sources = []
        for result in search_results:
            sources.append({
                'question': result['content'].split('\nA:')[0].replace('Q:', '').strip(),
                'section': result['metadata'].get('section', ''),
                'category': result['metadata'].get('category', ''),
                'score': result['score']
            })
        return sources
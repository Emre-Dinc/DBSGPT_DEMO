from typing import List, Optional, Dict, Any
from pymilvus import (
    connections,
    Collection,
    FieldSchema,
    CollectionSchema,
    DataType,
    utility,
)
from src.utils.path_utils import get_config_path
import yaml
import json
import logging
import numpy as np


class MilvusClient:
    def __init__(self, config_path: str = None):
        self.logger = logging.getLogger(__name__)

        config_path = config_path or get_config_path()
        with open(config_path, "r") as file:
            self.config = yaml.safe_load(file)["milvus"]

        self.host = self.config["host"]
        self.port = self.config["port"]
        self.collection_name = self.config["collection_name"]

        self.priority_terms = {
            'exam': {
                'direct': ['exam', 'examination', 'test', 'assessment'],
                'schedule': ['schedule', 'timetable', 'date', 'time', 'when'],
                'context': ['online', 'sit', 'take', 'repeat']
            },
            'visa': {
                'direct': ['visa', 'permit', 'immigration', 'stamp'],
                'requirements': ['requirement', 'document', 'need', 'must', 'necessary'],
                'process': ['apply', 'application', 'submit', 'renew', 'extend']
            },
            'library': {
                'direct': ['library', 'study room', 'book'],
                'access': ['access', 'enter', 'use', 'visit', 'open'],
                'services': ['resource', 'material', 'database', 'research']
            },
            'medical': {
                'direct': ['medical', 'health', 'healthcare', 'treatment'],
                'providers': ['doctor', 'gp', 'physician', 'clinic', 'hospital'],
                'services': ['service', 'appointment', 'care', 'consultation']
            },
            'student': {
                'direct': ['student card', 'id card', 'identification'],
                'card': ['card', 'id', 'badge', 'photo'],
                'status': ['active', 'current', 'registered', 'valid']
            }
        }

        self._connect()
        self.collection = self._init_collection()

    def _connect(self) -> None:
        try:
            try:
                connections.disconnect("default")
            except:
                pass

            connections.connect(
                alias="default",
                host=self.host,
                port=self.port
            )
            self.logger.info("Successfully connected to Milvus")
        except Exception as e:
            self.logger.error(f"Failed to connect to Milvus: {e}")
            raise

    def _init_collection(self) -> Collection:
        try:
            if utility.has_collection(self.collection_name):
                collection = Collection(self.collection_name)
                self.logger.info(f"Using existing collection: {self.collection_name}")
                return collection

            fields = [
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
                FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=384),
                FieldSchema(name="metadata", dtype=DataType.JSON),
            ]

            schema = CollectionSchema(
                fields=fields,
                description="School documents collection"
            )

            collection = Collection(name=self.collection_name, schema=schema)
            self.logger.info(f"Created new collection: {self.collection_name}")

            index_params = {
                "metric_type": "COSINE",
                "index_type": "IVF_SQ8",
                "params": {
                    "nlist": 2048,
                    "sq8_force": 1
                },
            }
            collection.create_index(field_name="embedding", index_params=index_params)
            self.logger.info("Created index on embedding field")

            return collection

        except Exception as e:
            self.logger.error(f"Failed to initialize collection: {e}")
            raise

    def insert(self, content: str, embedding: List[float], metadata: Dict[str, Any]) -> None:
        try:
            data = [
                [content],
                [embedding],
                [json.dumps(metadata)],
            ]

            self.collection.insert(data)
        except Exception as e:
            self.logger.error(f"Failed to insert document: {e}")
            try:
                self._connect()
                self.collection = self._init_collection()
                self.collection.insert(data)
            except Exception as e2:
                self.logger.error(f"Retry failed: {e2}")
                raise

    def _normalize_score(self, score: float, min_val: float = 0.65, max_val: float = 0.98) -> float:
        return min_val + score * (max_val - min_val)

    def search(self, query_embedding: List[float], limit: int = 5, query: str = "") -> List[Dict[str, Any]]:
        try:
            self.collection.load()

            search_params = {
                "metric_type": "L2",
                "params": {"nprobe": 16}
            }

            raw_results = self.collection.search(
                data=[query_embedding],
                anns_field="embedding",
                param=search_params,
                limit=limit * 4,
                output_fields=["content", "metadata"]
            )

            hits = []
            for hits_i in raw_results:
                for hit in hits_i:
                    content = hit.entity.get("content")
                    metadata = json.loads(hit.entity.get("metadata"))

                    qa_parts = content.split('\nA:', 1)
                    question = qa_parts[0].replace('Q:', '').strip()
                    answer = qa_parts[1].strip() if len(qa_parts) > 1 else ""

                    try:
                        vector_sim = 1.0 / (1.0 + np.clip(float(hit.score), 0, 10) * 1.5)
                    except (TypeError, ValueError, ZeroDivisionError):
                        vector_sim = 0.0

                    topic_rel = self._calculate_topic_relevance(query, question, answer)
                    meta_rel = self._calculate_metadata_relevance(query, metadata)
                    direct_match = 0.15 if self._has_direct_match(query, question) else 0

                    raw_score = (
                            0.40 * topic_rel +
                            0.25 * vector_sim +
                            0.20 * meta_rel +
                            direct_match
                    )

                    final_score = self._normalize_score(raw_score)

                    hits.append({
                        "content": content,
                        "metadata": metadata,
                        "score": final_score
                    })

            hits.sort(key=lambda x: x["score"], reverse=True)
            return hits[:limit]

        except Exception as e:
            self.logger.error(f"Search failed: {e}")
            try:
                self._connect()
                self.collection = self._init_collection()
                return self.search(query_embedding, limit, query)
            except Exception as e2:
                self.logger.error(f"Retry failed: {e2}")
                raise

    def _has_direct_match(self, query: str, text: str) -> bool:
        query_terms = set(query.lower().split())
        text_terms = set(text.lower().split())
        return len(query_terms & text_terms) / max(len(query_terms), 1) > 0.5

    def _calculate_topic_relevance(self, query: str, question: str, answer: str) -> float:
        query = query.lower()
        question = question.lower()
        answer = answer.lower()

        max_score = 0
        for topic, term_groups in self.priority_terms.items():
            if any(term in query for group in term_groups.values() for term in group):
                topic_score = 0

                direct_matches = sum(term in question for term in term_groups['direct'])
                topic_score += direct_matches * 0.8

                for idx, (group_name, terms) in enumerate(term_groups.items()):
                    if group_name != 'direct':
                        weight = 0.4 / max(idx + 1, 1)
                        matches = sum(term in question or term in answer for term in terms)
                        topic_score += matches * weight

                max_score = max(max_score, min(topic_score, 1.0))

        return max_score if max_score > 0 else 0.2

    def _calculate_metadata_relevance(self, query: str, metadata: Dict) -> float:
        query = query.lower()
        category = metadata.get("category", "").lower()
        section = metadata.get("section", "").lower()

        category_score = 0
        for topic, term_groups in self.priority_terms.items():
            if any(term in query for group in term_groups.values() for term in group):
                if any(term in category for term in term_groups['direct']):
                    category_score += 0.8
                elif any(term in category for group in term_groups.values() for term in group):
                    category_score += 0.4

                if any(term in section for term in term_groups['direct']):
                    category_score += 0.4
                elif any(term in section for group in term_groups.values() for term in group):
                    category_score += 0.2

        return min(category_score, 1.0)

    def delete(self, filter_params: Dict[str, Any]) -> None:
        try:
            expr = " and ".join([f'json_contains(metadata, "{v}", "{k}")'
                                 for k, v in filter_params.items()])
            self.collection.delete(expr)
            self.logger.info(f"Deleted documents matching filter: {filter_params}")
        except Exception as e:
            self.logger.error(f"Failed to delete documents: {e}")
            raise

    def __del__(self):
        try:
            connections.disconnect("default")
            self.logger.info("Disconnected from Milvus")
        except:
            pass
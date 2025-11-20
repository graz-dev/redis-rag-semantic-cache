"""
Redis Cache Manager for Knowledge Index and Semantic Cache Index.

This module handles:
- Connection to Redis Stack
- Management of two vector indexes:
  1. Knowledge Index: Stores document embeddings
  2. Cache Index: Stores query-response pairs for semantic caching
"""

import os
from typing import List, Optional, Dict, Any
import redis
from redis.commands.search.field import VectorField, TextField, NumericField
from redis.commands.search.indexDefinition import IndexDefinition, IndexType
from redis.commands.search.query import Query


class CacheManager:
    """Manages Redis connections and vector indexes for RAG and semantic caching."""
    
    # Index names
    KNOWLEDGE_INDEX = "knowledge_idx"
    CACHE_INDEX = "cache_idx"
    
    # Vector dimension for Gemini embeddings
    # Note: embedding-001 produces 768-dimensional vectors
    # This can be overridden via environment variable VECTOR_DIM
    VECTOR_DIM = int(os.getenv("VECTOR_DIM", "768"))
    
    def __init__(self, redis_url: str, vector_dim: Optional[int] = None):
        """
        Initialize the cache manager with Redis connection.
        
        Args:
            redis_url: Redis connection URL (e.g., redis://localhost:6379)
            vector_dim: Vector dimension (defaults to VECTOR_DIM class variable)
        """
        self.vector_dim = vector_dim or self.VECTOR_DIM
        self.redis_client = redis.from_url(redis_url, decode_responses=False)
        self._ensure_indexes()
    
    def _ensure_indexes(self):
        """Create Redis vector indexes if they don't exist."""
        try:
            # Check and create Knowledge Index
            try:
                self.redis_client.ft(self.KNOWLEDGE_INDEX).info()
            except redis.exceptions.ResponseError:
                # Index doesn't exist, create it
                schema = (
                    VectorField(
                        "embedding",
                        "FLAT",
                        {
                            "TYPE": "FLOAT32",
                            "DIM": self.vector_dim,
                            "DISTANCE_METRIC": "COSINE",
                        },
                    ),
                    TextField("text"),
                    TextField("source"),
                    TextField("chunk_id"),
                )
                definition = IndexDefinition(prefix=[f"{self.KNOWLEDGE_INDEX}:"], index_type=IndexType.HASH)
                self.redis_client.ft(self.KNOWLEDGE_INDEX).create_index(
                    fields=schema, definition=definition
                )
            
            # Check and create Cache Index
            try:
                self.redis_client.ft(self.CACHE_INDEX).info()
            except redis.exceptions.ResponseError:
                # Index doesn't exist, create it
                schema = (
                    VectorField(
                        "query_embedding",
                        "FLAT",
                        {
                            "TYPE": "FLOAT32",
                            "DIM": self.vector_dim,
                            "DISTANCE_METRIC": "COSINE",
                        },
                    ),
                    TextField("query"),
                    TextField("response"),
                    NumericField("timestamp"),
                )
                definition = IndexDefinition(prefix=[f"{self.CACHE_INDEX}:"], index_type=IndexType.HASH)
                self.redis_client.ft(self.CACHE_INDEX).create_index(
                    fields=schema, definition=definition
                )
        except Exception as e:
            raise ConnectionError(f"Failed to initialize Redis indexes: {e}")
    
    def add_document_chunk(
        self, 
        chunk_id: str, 
        text: str, 
        embedding: List[float], 
        source: str
    ):
        """
        Add a document chunk to the Knowledge Index.
        
        Args:
            chunk_id: Unique identifier for the chunk
            text: The text content of the chunk
            embedding: Vector embedding of the text
            source: Source file/path of the document
        """
        key = f"{self.KNOWLEDGE_INDEX}:{chunk_id}"
        
        # Prepare embedding as bytes
        embedding_bytes = self._float_list_to_bytes(embedding)
        
        self.redis_client.hset(
            key,
            mapping={
                "text": text,
                "embedding": embedding_bytes,
                "source": source,
                "chunk_id": chunk_id,
            }
        )
    
    def search_knowledge(
        self, 
        query_embedding: List[float], 
        top_k: int = 5,
        score_threshold: float = 0.0
    ) -> List[Dict[str, Any]]:
        """
        Search the Knowledge Index for similar document chunks.
        
        Args:
            query_embedding: Vector embedding of the query
            top_k: Number of results to return
            score_threshold: Minimum similarity score (0.0 to 1.0)
            
        Returns:
            List of dictionaries containing text, source, chunk_id, and score
        """
        query_embedding_bytes = self._float_list_to_bytes(query_embedding)
        
        # Create vector query
        base_query = f"*=>[KNN {top_k} @embedding $vec AS score]"
        query = Query(base_query).return_fields("text", "source", "chunk_id", "score").sort_by("score").paging(0, top_k).dialect(2)
        
        results = self.redis_client.ft(self.KNOWLEDGE_INDEX).search(
            query, query_params={"vec": query_embedding_bytes}
        )
        
        # Convert results and filter by threshold
        # Note: Redis returns distance (lower is better), we convert to similarity (higher is better)
        # For COSINE: similarity = 1 - distance
        filtered_results = []
        for doc in results.docs:
            distance = float(doc.score)
            similarity = 1 - distance  # Convert distance to similarity
            if similarity >= score_threshold:
                filtered_results.append({
                    "text": doc.text,
                    "source": doc.source,
                    "chunk_id": doc.chunk_id,
                    "score": similarity
                })
        
        return filtered_results
    
    def search_cache(
        self, 
        query_embedding: List[float], 
        similarity_threshold: float = 0.9,
        top_k: int = 1
    ) -> Optional[Dict[str, Any]]:
        """
        Search the Cache Index for similar queries.
        
        Args:
            query_embedding: Vector embedding of the query
            similarity_threshold: Minimum similarity score to consider a cache hit
            top_k: Number of results to check
            
        Returns:
            Dictionary with query, response, and score if found, None otherwise
        """
        query_embedding_bytes = self._float_list_to_bytes(query_embedding)
        
        # Create vector query
        base_query = f"*=>[KNN {top_k} @query_embedding $vec AS score]"
        query = Query(base_query).return_fields("query", "response", "score").sort_by("score").paging(0, top_k).dialect(2)
        
        results = self.redis_client.ft(self.CACHE_INDEX).search(
            query, query_params={"vec": query_embedding_bytes}
        )
        
        if results.docs:
            # Check the top result
            doc = results.docs[0]
            distance = float(doc.score)
            similarity = 1 - distance  # Convert distance to similarity
            
            if similarity >= similarity_threshold:
                return {
                    "query": doc.query,
                    "response": doc.response,
                    "score": similarity
                }
        
        return None
    
    def add_to_cache(
        self, 
        query: str, 
        query_embedding: List[float], 
        response: str
    ):
        """
        Add a query-response pair to the Cache Index.
        
        Args:
            query: The user's query
            query_embedding: Vector embedding of the query
            response: The generated response
        """
        import time
        
        # Generate a unique key based on timestamp and query hash
        cache_id = f"{int(time.time())}_{hash(query) % 1000000}"
        key = f"{self.CACHE_INDEX}:{cache_id}"
        
        query_embedding_bytes = self._float_list_to_bytes(query_embedding)
        
        self.redis_client.hset(
            key,
            mapping={
                "query": query,
                "query_embedding": query_embedding_bytes,
                "response": response,
                "timestamp": int(time.time()),
            }
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the Redis indexes.
        
        Returns:
            Dictionary with index statistics
        """
        try:
            # Get Knowledge Index stats
            knowledge_info = self.redis_client.ft(self.KNOWLEDGE_INDEX).info()
            knowledge_count = int(knowledge_info.get("num_docs", 0))
            
            # Get Cache Index stats
            cache_info = self.redis_client.ft(self.CACHE_INDEX).info()
            cache_count = int(cache_info.get("num_docs", 0))
            
            # Check connection
            self.redis_client.ping()
            connected = True
        except Exception as e:
            connected = False
            knowledge_count = 0
            cache_count = 0
        
        return {
            "connected": connected,
            "knowledge_index_count": knowledge_count,
            "cache_index_count": cache_count,
        }
    
    def _float_list_to_bytes(self, float_list: List[float]) -> bytes:
        """Convert a list of floats to bytes for Redis storage."""
        import struct
        return struct.pack(f"{len(float_list)}f", *float_list)
    
    def close(self):
        """Close the Redis connection."""
        if self.redis_client:
            self.redis_client.close()


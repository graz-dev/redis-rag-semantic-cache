"""
Redis Cache Manager for Knowledge Index and Semantic Cache Index.

This module handles:
- Connection to Redis Stack via RedisVL
- Management of two vector indexes:
  1. Knowledge Index: Stores document chunks (via SearchIndex)
  2. Cache Index: Stores query-response pairs (via SemanticCache)
"""

import os
from typing import List, Optional, Dict, Any
from redisvl.index import SearchIndex
from redisvl.extensions.llmcache import SemanticCache


from dummy_vectorizer import DummyVectorizer

class CacheManager:
    """Manages Redis connections and vector indexes for RAG and semantic caching using RedisVL."""
    
    def __init__(self, redis_url: str, vector_dim: Optional[int] = None):
        """
        Initialize the cache manager with Redis connection.
        
        Args:
            redis_url: Redis connection URL (e.g., redis://localhost:6379)
            vector_dim: Vector dimension (defaults to 768 if not provided)
        """
        self.redis_url = redis_url
        self.vector_dim = vector_dim or int(os.getenv("VECTOR_DIM", "768"))
        
        # Initialize Knowledge Index from schema
        # We assume schema.yaml is in the same directory or we can define it dict-based here
        # For robustness, let's define it dict-based to avoid file path issues
        self.knowledge_schema = {
            "index": {
                "name": "knowledge_idx",
                "prefix": "knowledge_idx:"
            },
            "fields": [
                {"name": "text", "type": "text"},
                {"name": "source", "type": "text"},
                {"name": "chunk_id", "type": "text"},
                {
                    "name": "embedding",
                    "type": "vector",
                    "attrs": {
                        "algorithm": "hnsw",
                        "distance_metric": "cosine",
                        "dims": self.vector_dim,
                        "datatype": "float32"
                    }
                }
            ]
        }
        
        self.knowledge_index = SearchIndex.from_dict(self.knowledge_schema)
        self.knowledge_index.connect(redis_url=self.redis_url)
        
        # Create index if it doesn't exist
        if not self.knowledge_index.exists():
            self.knowledge_index.create(overwrite=True)
            
        # Initialize Semantic Cache
        # We pass a dummy vectorizer because we handle embedding generation externally
        self.semantic_cache = SemanticCache(
            name="cache_idx",
            redis_url=self.redis_url,
            distance_threshold=0.1, # Default threshold, can be overridden in check()
            vectorizer=DummyVectorizer(dims=self.vector_dim),
            overwrite=True # Force overwrite if schema mismatch
        )
    
    def _float_list_to_bytes(self, float_list: List[float]) -> bytes:
        """Convert a list of floats to bytes for Redis storage."""
        import struct
        return struct.pack(f"{len(float_list)}f", *float_list)

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
        # Convert embedding to bytes to avoid redis-py DataError
        embedding_bytes = self._float_list_to_bytes(embedding)
        
        record = {
            "chunk_id": chunk_id,
            "text": text,
            "embedding": embedding_bytes,
            "source": source
        }
        # RedisVL expects a list of dicts for load
        self.knowledge_index.load([record])
    
    def add_documents(self, records: List[Dict[str, Any]]):
        """
        Bulk add document chunks to the Knowledge Index.
        
        Args:
            records: List of dictionaries with keys: chunk_id, text, embedding, source
        """
        # Convert embeddings to bytes for all records
        for record in records:
            if isinstance(record.get("embedding"), list):
                record["embedding"] = self._float_list_to_bytes(record["embedding"])
                
        self.knowledge_index.load(records)

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
        from redisvl.query import VectorQuery
        
        query = VectorQuery(
            vector=query_embedding,
            vector_field_name="embedding",
            return_fields=["text", "source", "chunk_id"],
            num_results=top_k
        )
        
        results = self.knowledge_index.query(query)
        
        # Convert results and filter by threshold
        filtered_results = []
        for doc in results:
            # RedisVL returns distance by default for HNSW/Cosine? 
            # Actually RedisVL normalizes this usually, but let's check the vector_distance field
            # If using cosine distance, similarity = 1 - distance
            
            distance = float(doc.get("vector_distance", 1.0))
            similarity = 1 - distance
            
            if similarity >= score_threshold:
                filtered_results.append({
                    "text": doc.get("text"),
                    "source": doc.get("source"),
                    "chunk_id": doc.get("chunk_id"),
                    "score": similarity
                })
        
        return filtered_results
    
    def check_cache(
        self, 
        query_embedding: List[float], 
        similarity_threshold: float = 0.9
    ) -> Optional[Dict[str, Any]]:
        """
        Check the Semantic Cache for a similar query.
        
        Args:
            query_embedding: Vector embedding of the query
            similarity_threshold: Minimum similarity score to consider a cache hit
            
        Returns:
            Dictionary with response and score if found, None otherwise
        """
        # SemanticCache.check() usually takes text prompt or vector.
        # If we pass vector, it returns the response list if found.
        # However, RedisVL SemanticCache.check() returns a list of results or None?
        # Let's look at standard usage. 
        # actually check() returns the response text list if hit, or empty list if miss.
        
        # We need to be careful about the threshold. 
        # SemanticCache uses distance_threshold. 
        # similarity_threshold 0.9 means distance_threshold 0.1
        distance_threshold = 1.0 - similarity_threshold
        
        # We temporarily update the threshold of the instance or pass it if supported
        self.semantic_cache.set_threshold(distance_threshold)
        
        # check() returns List[Dict] usually with 'response', 'payload', etc?
        # Wait, looking at RedisVL docs/code:
        # check(vector=...) returns List[Dict] of results.
        
        results = self.semantic_cache.check(vector=query_embedding, num_results=1)
        
        if results:
            # results is a list of dicts, e.g. [{'response': '...', 'vector_distance': ...}]
            result = results[0]
            return {
                "response": result["response"],
                "score": 1 - float(result.get("vector_distance", 0.0)),
                # We might not get the original query back easily unless we stored it in metadata/payload
                # But for now let's just return what we have
                "query": "cached_query" # Placeholder or if we stored it
            }
            
        return None
    
    def store_cache(
        self, 
        query: str, 
        query_embedding: List[float], 
        response: str
    ):
        """
        Store a query-response pair in the Semantic Cache.
        
        Args:
            query: The user's query
            query_embedding: Vector embedding of the query
            response: The generated response
        """
        self.semantic_cache.store(
            prompt=query,
            response=response,
            vector=query_embedding
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the Redis indexes.
        
        Returns:
            Dictionary with index statistics
        """
        try:
            knowledge_info = self.knowledge_index.info()
            knowledge_count = knowledge_info.get("num_docs", 0)
            
            # SemanticCache doesn't expose info() directly easily, but it uses an underlying SearchIndex
            # accessible via self.semantic_cache.index
            cache_info = self.semantic_cache.index.info()
            cache_count = cache_info.get("num_docs", 0)
            
            connected = True
        except Exception:
            connected = False
            knowledge_count = 0
            cache_count = 0
        
        return {
            "connected": connected,
            "knowledge_index_count": knowledge_count,
            "cache_index_count": cache_count,
        }
    
    def close(self):
        """Close the Redis connection."""
        # RedisVL manages connections internally, usually no explicit close needed for the client object
        # but we can try to disconnect if exposed
        if hasattr(self.knowledge_index, "disconnect"):
            self.knowledge_index.disconnect()

"""
RAG Engine using Google Gemini for embeddings and LLM generation.

This module handles:
- Document loading and chunking
- Embedding generation using Gemini
- RAG-based answer generation
- Semantic caching integration
"""

import os
from typing import List, Dict, Any, Optional
from pathlib import Path
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from cache_manager import CacheManager


class RAGEngine:
    """RAG Engine for document processing and query answering."""
    
    def __init__(
        self,
        cache_manager: CacheManager,
        google_api_key: str,
        model_name: str = "gemini-pro",
        embedding_model: str = "models/embedding-001",
        cache_threshold: float = 0.9
    ):
        """
        Initialize the RAG Engine.
        
        Args:
            cache_manager: Instance of CacheManager for Redis operations
            google_api_key: Google API key for Gemini
            model_name: Gemini model name for generation
            embedding_model: Gemini embedding model name
            cache_threshold: Similarity threshold for semantic cache hits
        """
        self.cache_manager = cache_manager
        self.model_name = model_name
        self.embedding_model = embedding_model
        self.cache_threshold = cache_threshold
        
        # Initialize Gemini embeddings and LLM
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model=embedding_model,
            google_api_key=google_api_key
        )
        self.llm = ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=google_api_key,
            temperature=0.7
        )
        
        # Text splitter for chunking documents
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
    
    def load_documents(self, path: str) -> int:
        """
        Load documents from a file or directory and store them in Redis.
        
        Args:
            path: Path to a file or directory
            
        Returns:
            Number of chunks loaded
        """
        path_obj = Path(path)
        
        if not path_obj.exists():
            raise FileNotFoundError(f"Path not found: {path}")
        
        # Collect all document files
        documents = []
        if path_obj.is_file():
            documents.extend(self._load_file(path_obj))
        elif path_obj.is_dir():
            for file_path in path_obj.rglob("*"):
                if file_path.is_file() and self._is_supported_file(file_path):
                    documents.extend(self._load_file(file_path))
        
        if not documents:
            raise ValueError(f"No supported documents found in: {path}")
        
        # Split documents into chunks
        all_chunks = []
        for doc in documents:
            chunks = self.text_splitter.split_documents([doc])
            all_chunks.extend(chunks)
        
        # Embed and store chunks
        total_chunks = len(all_chunks)
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
        ) as progress:
            task = progress.add_task(
                f"[cyan]Processing {total_chunks} chunks...", 
                total=total_chunks
            )
            
            for i, chunk in enumerate(all_chunks):
                # Generate embedding
                embedding = self.embeddings.embed_query(chunk.page_content)
                
                # Generate unique chunk ID
                chunk_id = f"{hash(chunk.metadata.get('source', 'unknown'))}_{i}"
                
                # Store in Redis
                self.cache_manager.add_document_chunk(
                    chunk_id=chunk_id,
                    text=chunk.page_content,
                    embedding=embedding,
                    source=str(chunk.metadata.get("source", "unknown"))
                )
                
                progress.update(task, advance=1)
        
        return total_chunks
    
    def _load_file(self, file_path: Path) -> List[Document]:
        """Load a single file and return Document objects."""
        try:
            if file_path.suffix.lower() == ".pdf":
                loader = PyPDFLoader(str(file_path))
            elif file_path.suffix.lower() in [".txt", ".md"]:
                loader = TextLoader(str(file_path), encoding="utf-8")
            else:
                return []
            
            documents = loader.load()
            # Update metadata with absolute path
            for doc in documents:
                doc.metadata["source"] = str(file_path.absolute())
            
            return documents
        except Exception as e:
            print(f"Warning: Failed to load {file_path}: {e}")
            return []
    
    def _is_supported_file(self, file_path: Path) -> bool:
        """Check if a file type is supported."""
        supported_extensions = {".txt", ".md", ".pdf"}
        return file_path.suffix.lower() in supported_extensions
    
    def query(
        self, 
        user_query: str, 
        top_k: int = 5
    ) -> Dict[str, Any]:
        """
        Process a user query with semantic caching and RAG.
        
        Args:
            user_query: The user's question
            top_k: Number of document chunks to retrieve for RAG
            
        Returns:
            Dictionary with response, source (cache or rag), and metadata
        """
        # Step 1: Embed the query
        query_embedding = self.embeddings.embed_query(user_query)
        
        # Step 2: Check semantic cache first
        cache_result = self.cache_manager.search_cache(
            query_embedding=query_embedding,
            similarity_threshold=self.cache_threshold
        )
        
        if cache_result:
            # Cache hit - return cached response
            return {
                "response": cache_result["response"],
                "source": "cache",
                "similarity": cache_result["score"],
                "cached_query": cache_result["query"]
            }
        
        # Step 3: Cache miss - perform RAG
        # Search knowledge base
        knowledge_results = self.cache_manager.search_knowledge(
            query_embedding=query_embedding,
            top_k=top_k
        )
        
        if not knowledge_results:
            return {
                "response": "I couldn't find any relevant information in the knowledge base to answer your question.",
                "source": "rag",
                "context_chunks": []
            }
        
        # Build context from retrieved chunks
        context = "\n\n".join([result["text"] for result in knowledge_results])
        
        # Generate answer using LLM
        prompt = f"""You are a helpful assistant that answers questions based on the provided context.

Context:
{context}

Question: {user_query}

Please provide a comprehensive answer based on the context above. If the context doesn't contain enough information to answer the question, say so."""

        response = self.llm.invoke(prompt).content
        
        # Step 4: Cache the query-response pair
        self.cache_manager.add_to_cache(
            query=user_query,
            query_embedding=query_embedding,
            response=response
        )
        
        return {
            "response": response,
            "source": "rag",
            "context_chunks": knowledge_results,
            "num_chunks_used": len(knowledge_results)
        }


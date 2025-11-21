"""
RAG Engine supporting both Google Gemini and Ollama for embeddings and LLM generation.

This module handles:
- Document loading and chunking
- Embedding generation (Gemini or Ollama)
- RAG-based answer generation
- Semantic caching integration
"""

import os
from typing import List, Dict, Any, Optional
from pathlib import Path
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.schema.embeddings import Embeddings
from langchain.schema.language_model import BaseLanguageModel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from cache_manager import CacheManager
from cost_calculator import CostCalculator

# Conditional imports
try:
    from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

try:
    # Try langchain-ollama first (preferred)
    from langchain_ollama import OllamaEmbeddings, ChatOllama
    OLLAMA_AVAILABLE = True
except ImportError:
    try:
        # Fallback to langchain-community
        from langchain_community.embeddings import OllamaEmbeddings
        from langchain_community.chat_models import ChatOllama
        OLLAMA_AVAILABLE = True
    except ImportError:
        OLLAMA_AVAILABLE = False


class RAGEngine:
    """RAG Engine for document processing and query answering."""
    
    def __init__(
        self,
        cache_manager: CacheManager,
        provider: str = "gemini",  # "gemini" or "ollama"
        google_api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        embedding_model: Optional[str] = None,
        ollama_base_url: str = "http://localhost:11434",
        cache_threshold: float = 0.9
    ):
        """
        Initialize the RAG Engine.
        
        Args:
            cache_manager: Instance of CacheManager for Redis operations
            provider: "gemini" or "ollama"
            google_api_key: Google API key for Gemini (required if provider="gemini")
            model_name: Model name for generation (provider-specific)
            embedding_model: Embedding model name (provider-specific)
            ollama_base_url: Base URL for Ollama API (default: http://localhost:11434)
            cache_threshold: Similarity threshold for semantic cache hits
        """
        self.cache_manager = cache_manager
        self.provider = provider.lower()
        self.cache_threshold = cache_threshold
        
        # Initialize embeddings and LLM based on provider
        if self.provider == "gemini":
            if not GEMINI_AVAILABLE:
                raise ImportError("langchain-google-genai is not installed. Install it with: pip install langchain-google-genai")
            if not google_api_key:
                raise ValueError("google_api_key is required when using Gemini provider")
            
            self.model_name = model_name or os.getenv("GEMINI_MODEL", "gemini-pro")
            self.embedding_model = embedding_model or os.getenv("GEMINI_EMBEDDING_MODEL", "models/embedding-001")
            
            self.embeddings: Embeddings = GoogleGenerativeAIEmbeddings(
                model=self.embedding_model,
                google_api_key=google_api_key
            )
            self.llm: BaseLanguageModel = ChatGoogleGenerativeAI(
                model=self.model_name,
                google_api_key=google_api_key,
                temperature=0.7
            )
            
        elif self.provider == "ollama":
            if not OLLAMA_AVAILABLE:
                raise ImportError("langchain-ollama is not installed. Install it with: pip install langchain-ollama")
            
            # Default Ollama models
            self.model_name = model_name or os.getenv("OLLAMA_MODEL", "llama3.1:8b")
            self.embedding_model = embedding_model or os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")
            
            self.embeddings: Embeddings = OllamaEmbeddings(
                model=self.embedding_model,
                base_url=ollama_base_url
            )
            self.llm: BaseLanguageModel = ChatOllama(
                model=self.model_name,
                base_url=ollama_base_url,
                temperature=0.7
            )
        else:
            raise ValueError(f"Unsupported provider: {provider}. Use 'gemini' or 'ollama'")
        
        # Text splitter for chunking documents
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        
        # Initialize cost calculator
        self.cost_calculator = CostCalculator()
    
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
        records_to_load = []
        
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
                
                # Prepare record for bulk load
                records_to_load.append({
                    "chunk_id": chunk_id,
                    "text": chunk.page_content,
                    "embedding": embedding,
                    "source": str(chunk.metadata.get("source", "unknown"))
                })
                
                progress.update(task, advance=1)
        
        # Bulk load into Redis
        if records_to_load:
            self.cache_manager.add_documents(records_to_load)
        
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
        cache_result = self.cache_manager.check_cache(
            query_embedding=query_embedding,
            similarity_threshold=self.cache_threshold
        )
        
        if cache_result:
            # Cache hit - return cached response
            # Calculate cost for current model (only embedding, no LLM cost)
            cost_info = self.cost_calculator.calculate_query_cost(
                provider=self.provider,
                model_name=self.model_name,
                embedding_model=self.embedding_model,
                query=user_query,
                response=cache_result["response"],
                use_cache=True
            )
            
            # For cache hit, we need to calculate what the cost would have been WITHOUT cache
            # to show the savings. So we calculate costs as if there was no cache.
            # We'll need context for proper calculation - get it from knowledge base
            knowledge_results = self.cache_manager.search_knowledge(
                query_embedding=query_embedding,
                top_k=5
            )
            context = "\n\n".join([result["text"] for result in knowledge_results]) if knowledge_results else ""
            
            # Calculate costs for all models as if there was NO cache (to show what it would cost)
            all_costs = self.cost_calculator.calculate_all_models_cost(
                query=user_query,
                context=context,
                response=cache_result["response"],
                use_cache=False  # Calculate as if no cache to show potential cost
            )
            
            return {
                "response": cache_result["response"],
                "source": "cache",
                "similarity": cache_result["score"],
                "cached_query": cache_result.get("query", "unknown"),
                "cost": cost_info,
                "all_costs": all_costs
            }
        
        # Step 3: Cache miss - perform RAG
        # Search knowledge base
        knowledge_results = self.cache_manager.search_knowledge(
            query_embedding=query_embedding,
            top_k=top_k
        )
        
        if not knowledge_results:
            # Still calculate cost for embedding
            cost_info = self.cost_calculator.calculate_query_cost(
                provider=self.provider,
                model_name=self.model_name,
                embedding_model=self.embedding_model,
                query=user_query,
                use_cache=False
            )
            
            # Calculate costs for all models
            all_costs = self.cost_calculator.calculate_all_models_cost(
                query=user_query,
                use_cache=False
            )
            
            return {
                "response": "I couldn't find any relevant information in the knowledge base to answer your question.",
                "source": "rag",
                "context_chunks": [],
                "cost": cost_info,
                "all_costs": all_costs
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
        
        # Calculate cost for current model
        cost_info = self.cost_calculator.calculate_query_cost(
            provider=self.provider,
            model_name=self.model_name,
            embedding_model=self.embedding_model,
            query=user_query,
            context=context,
            response=response,
            use_cache=False
        )
        
        # Calculate costs for all models
        all_costs = self.cost_calculator.calculate_all_models_cost(
            query=user_query,
            context=context,
            response=response,
            use_cache=False
        )
        
        # Step 4: Cache the query-response pair
        self.cache_manager.store_cache(
            query=user_query,
            query_embedding=query_embedding,
            response=response
        )
        
        return {
            "response": response,
            "source": "rag",
            "context_chunks": knowledge_results,
            "num_chunks_used": len(knowledge_results),
            "cost": cost_info,
            "all_costs": all_costs
        }

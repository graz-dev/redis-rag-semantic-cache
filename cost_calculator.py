"""
Cost Calculator for tracking API usage costs.

This module calculates costs for LLM API calls and embeddings based on
token usage and configured pricing in costs-config.json.
"""

import json
import os
from typing import Dict, List, Optional, Tuple
from pathlib import Path


class CostCalculator:
    """Calculate API costs based on token usage and configured pricing."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the cost calculator.
        
        Args:
            config_path: Path to costs-config.json file. If None, looks for it in the same directory.
        """
        if config_path is None:
            config_path = Path(__file__).parent / "costs-config.json"
        
        self.config_path = Path(config_path)
        self.config = self._load_config()
    
    def _load_config(self) -> Dict:
        """Load cost configuration from JSON file."""
        if not self.config_path.exists():
            # Return default empty config if file doesn't exist
            return {"providers": {}}
        
        try:
            with open(self.config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Warning: Could not load cost config: {e}")
            return {"providers": {}}
    
    def _get_model_config(self, provider: str, model_name: str) -> Optional[Dict]:
        """Get configuration for a specific model."""
        providers = self.config.get("providers", {})
        provider_config = providers.get(provider.lower(), {})
        models = provider_config.get("models", {})
        return models.get(model_name)
    
    def _get_embedding_config(self, provider: str, embedding_model: str) -> Optional[Dict]:
        """Get configuration for a specific embedding model."""
        providers = self.config.get("providers", {})
        provider_config = providers.get(provider.lower(), {})
        embedding_models = provider_config.get("embedding_models", {})
        return embedding_models.get(embedding_model)
    
    def estimate_tokens(self, text: str) -> int:
        """
        Estimate token count for a text string.
        Uses a simple approximation: ~4 characters per token for English text.
        For more accurate counting, you'd need tiktoken or similar.
        
        Args:
            text: Input text
            
        Returns:
            Estimated token count
        """
        # Simple approximation: ~4 chars per token
        # This is a rough estimate; actual tokenization varies by model
        return len(text) // 4
    
    def calculate_llm_cost(
        self, 
        provider: str, 
        model_name: str, 
        input_tokens: int, 
        output_tokens: int
    ) -> Tuple[float, str]:
        """
        Calculate cost for LLM generation.
        
        Args:
            provider: Provider name (gemini, openai, ollama)
            model_name: Model name
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            
        Returns:
            Tuple of (cost, currency)
        """
        model_config = self._get_model_config(provider, model_name)
        
        if not model_config:
            return 0.0, "USD"
        
        input_cost_per_1k = model_config.get("input_cost_per_1k_tokens", 0.0)
        output_cost_per_1k = model_config.get("output_cost_per_1k_tokens", 0.0)
        currency = model_config.get("currency", "USD")
        
        input_cost = (input_tokens / 1000.0) * input_cost_per_1k
        output_cost = (output_tokens / 1000.0) * output_cost_per_1k
        
        total_cost = input_cost + output_cost
        return total_cost, currency
    
    def calculate_embedding_cost(
        self, 
        provider: str, 
        embedding_model: str, 
        tokens: int
    ) -> Tuple[float, str]:
        """
        Calculate cost for embedding generation.
        
        Args:
            provider: Provider name (gemini, openai, ollama)
            embedding_model: Embedding model name
            tokens: Number of tokens
            
        Returns:
            Tuple of (cost, currency)
        """
        embedding_config = self._get_embedding_config(provider, embedding_model)
        
        if not embedding_config:
            return 0.0, "USD"
        
        cost_per_1k = embedding_config.get("cost_per_1k_tokens", 0.0)
        currency = embedding_config.get("currency", "USD")
        
        cost = (tokens / 1000.0) * cost_per_1k
        return cost, currency
    
    def get_all_models(self) -> Dict[str, List[str]]:
        """
        Get all configured models grouped by provider.
        
        Returns:
            Dictionary mapping provider names to lists of model names
        """
        all_models = {}
        providers = self.config.get("providers", {})
        
        for provider_name, provider_config in providers.items():
            models = provider_config.get("models", {})
            all_models[provider_name] = list(models.keys())
        
        return all_models
    
    def get_all_embedding_models(self) -> Dict[str, List[str]]:
        """
        Get all configured embedding models grouped by provider.
        
        Returns:
            Dictionary mapping provider names to lists of embedding model names
        """
        all_models = {}
        providers = self.config.get("providers", {})
        
        for provider_name, provider_config in providers.items():
            embedding_models = provider_config.get("embedding_models", {})
            all_models[provider_name] = list(embedding_models.keys())
        
        return all_models
    
    def calculate_all_models_cost(
        self,
        query: str,
        context: str = "",
        response: str = "",
        use_cache: bool = False
    ) -> List[Dict]:
        """
        Calculate costs for all configured models.
        
        Args:
            query: User query
            context: Retrieved context (for RAG)
            response: Generated response
            use_cache: Whether cache was used
            
        Returns:
            List of cost dictionaries for all models
        """
        all_costs = []
        all_models = self.get_all_models()
        all_embedding_models = self.get_all_embedding_models()
        
        # Estimate tokens once
        query_tokens = self.estimate_tokens(query)
        context_tokens = self.estimate_tokens(context) if context else 0
        response_tokens = self.estimate_tokens(response) if response else 0
        
        # Calculate costs for each provider and model combination
        for provider, models in all_models.items():
            # Get default embedding model for this provider
            embedding_models = all_embedding_models.get(provider, [])
            default_embedding = embedding_models[0] if embedding_models else ""
            
            for model_name in models:
                # Calculate embedding cost (use first available embedding model for provider)
                embedding_cost = 0.0
                currency = "USD"
                
                if default_embedding:
                    embedding_cost, currency = self.calculate_embedding_cost(
                        provider, default_embedding, query_tokens
                    )
                
                # Calculate LLM cost (only if not using cache)
                llm_cost = 0.0
                input_tokens = 0
                output_tokens = 0
                
                if not use_cache:
                    input_tokens = query_tokens + context_tokens + 100
                    output_tokens = response_tokens
                    llm_cost, currency = self.calculate_llm_cost(
                        provider, model_name, input_tokens, output_tokens
                    )
                
                total_cost = embedding_cost + llm_cost
                
                all_costs.append({
                    "provider": provider,
                    "model": model_name,
                    "embedding_model": default_embedding,
                    "use_cache": use_cache,
                    "query_tokens": query_tokens,
                    "context_tokens": context_tokens,
                    "response_tokens": response_tokens,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "embedding_cost": embedding_cost,
                    "llm_cost": llm_cost,
                    "total_cost": total_cost,
                    "currency": currency,
                    "savings": llm_cost if use_cache else 0.0
                })
        
        return all_costs
    
    def calculate_query_cost(
        self,
        provider: str,
        model_name: str,
        embedding_model: str,
        query: str,
        context: str = "",
        response: str = "",
        use_cache: bool = False
    ) -> Dict:
        """
        Calculate total cost for a query-answer pair.
        
        Args:
            provider: Provider name
            model_name: LLM model name
            embedding_model: Embedding model name
            query: User query
            context: Retrieved context (for RAG)
            response: Generated response
            use_cache: Whether cache was used (no LLM cost if True)
            
        Returns:
            Dictionary with cost breakdown
        """
        # Estimate tokens
        query_tokens = self.estimate_tokens(query)
        context_tokens = self.estimate_tokens(context) if context else 0
        response_tokens = self.estimate_tokens(response) if response else 0
        
        # Calculate embedding cost (always needed for query)
        embedding_cost, currency = self.calculate_embedding_cost(
            provider, embedding_model, query_tokens
        )
        
        # Calculate LLM cost (only if not using cache)
        llm_cost = 0.0
        input_tokens = 0
        output_tokens = 0
        
        if not use_cache:
            # Input tokens = query + context + prompt overhead (~100 tokens)
            input_tokens = query_tokens + context_tokens + 100
            output_tokens = response_tokens
            llm_cost, currency = self.calculate_llm_cost(
                provider, model_name, input_tokens, output_tokens
            )
        
        total_cost = embedding_cost + llm_cost
        
        return {
            "provider": provider,
            "model": model_name,
            "embedding_model": embedding_model,
            "use_cache": use_cache,
            "query_tokens": query_tokens,
            "context_tokens": context_tokens,
            "response_tokens": response_tokens,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "embedding_cost": embedding_cost,
            "llm_cost": llm_cost,
            "total_cost": total_cost,
            "currency": currency,
            "savings": llm_cost if use_cache else 0.0  # Savings = LLM cost that was avoided
        }


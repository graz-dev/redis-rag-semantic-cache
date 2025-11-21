"""
CLI interface for the RAG application using Typer and Rich.

Commands:
- load_documents: Load and index documents
- info: Display system information
- chat: Interactive chat session with semantic caching
"""

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.prompt import Prompt
from rich.markdown import Markdown
from rich import print as rprint
from typing import Optional
import os
from dotenv import load_dotenv

from cache_manager import CacheManager
from rag_engine import RAGEngine
from cost_calculator import CostCalculator

# Load environment variables
load_dotenv()

app = typer.Typer(help="RAG CLI Application with Semantic Caching")
console = Console()


def get_cache_manager() -> CacheManager:
    """Initialize and return CacheManager instance."""
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
    return CacheManager(redis_url)


def display_provider_info(rag_engine: RAGEngine):
    """Display provider and model information."""
    provider = rag_engine.provider.upper()
    llm_model = rag_engine.model_name
    embedding_model = rag_engine.embedding_model
    
    console.print(
        Panel(
            f"[bold]Provider:[/bold] [cyan]{provider}[/cyan]\n"
            f"[bold]LLM Model:[/bold] [green]{llm_model}[/green]\n"
            f"[bold]Embedding Model:[/bold] [green]{embedding_model}[/green]",
            title="Model Configuration",
            border_style="blue"
        )
    )


def format_cost(cost: float) -> str:
    """Format cost for display."""
    if cost == 0.0:
        return "$0.00"
    elif cost < 0.0001:
        return f"${cost:.6f}"
    elif cost < 0.01:
        return f"${cost:.4f}"
    else:
        return f"${cost:.4f}"


def display_cost_info(cost_info: dict, all_costs: list = None):
    """
    Display cost information for a query.
    Shows costs and savings for all configured models.
    
    Args:
        cost_info: Cost info for the current model (used to get use_cache flag)
        all_costs: List of cost info for all models (for comparison)
    """
    if not all_costs:
        return
    
    use_cache = cost_info.get("use_cache", False)
    
    # Display comparison table with all models
    display_cost_comparison_table(all_costs, use_cache)


def display_cost_comparison_table(all_costs: list, use_cache: bool):
    """
    Display a comparison table showing LLM costs and savings for all configured models.
    When cache hit: shows what the LLM cost would have been without cache, and savings.
    When no cache: shows the actual LLM cost for each model.
    """
    if not all_costs:
        return
    
    currency = all_costs[0].get("currency", "USD")
    
    # Create comparison table - showing LLM costs and savings
    if use_cache:
        title = "💰 LLM Cost Comparison (All Models) - Cache Hit!"
        # When cache hit, we need to calculate what the cost would have been
        # The llm_cost in all_costs is already calculated as if there was no cache
    else:
        title = "💰 LLM Cost Comparison (All Models)"
    
    table = Table(title=title, show_header=True, header_style="bold magenta")
    table.add_column("Provider", style="cyan", no_wrap=True)
    table.add_column("Model", style="green")
    
    if use_cache:
        # When cache hit: show cost without cache and savings
        table.add_column("Cost (No Cache)", justify="right", style="yellow")
        table.add_column("Cost (With Cache)", justify="right", style="dim")
        table.add_column("Savings", justify="right", style="green")
    else:
        # When no cache: show actual cost
        table.add_column("LLM Cost", justify="right", style="yellow")
    
    # Sort by LLM cost
    sorted_costs = sorted(all_costs, key=lambda x: x.get("llm_cost", 0.0))
    
    for cost in sorted_costs:
        provider = cost.get("provider", "").upper()
        model = cost.get("model", "")
        llm_cost_without_cache = cost.get("llm_cost", 0.0)
        llm_cost_with_cache = 0.0  # With cache, LLM cost is 0
        savings = llm_cost_without_cache - llm_cost_with_cache  # Savings = cost without cache - cost with cache
        
        provider_display = f"[cyan]{provider}[/cyan]"
        model_display = f"[green]{model}[/green]"
        
        if use_cache:
            row_data = [
                provider_display,
                model_display,
                f"[yellow]{format_cost(llm_cost_without_cache)}[/yellow] {currency}",
                f"[dim]{format_cost(llm_cost_with_cache)}[/dim] {currency}",
                f"[green]{format_cost(savings)}[/green] {currency}",
            ]
        else:
            row_data = [
                provider_display,
                model_display,
                f"[yellow]{format_cost(llm_cost_without_cache)}[/yellow] {currency}",
            ]
        
        table.add_row(*row_data)
    
    console.print(table)
    
    # Show token info (use first cost entry for token info)
    first_cost = all_costs[0]
    query_tokens = first_cost.get("query_tokens", 0)
    context_tokens = first_cost.get("context_tokens", 0)
    response_tokens = first_cost.get("response_tokens", 0)
    
    if query_tokens > 0 or context_tokens > 0 or response_tokens > 0:
        console.print(
            f"[dim]Tokens: Query={query_tokens}, Context={context_tokens}, Response={response_tokens}[/dim]"
        )
    
    # Show cache hit indicator
    if use_cache:
        console.print(
            f"[green]💡 Cache Hit! Cost (No Cache) shows what it would cost without cache. "
            f"Savings = Cost (No Cache) - Cost (With Cache)[/green]"
        )




def get_rag_engine(cache_manager: CacheManager) -> RAGEngine:
    """Initialize and return RAGEngine instance."""
    # Determine provider (default to gemini if not specified)
    provider = os.getenv("LLM_PROVIDER", "gemini").lower()
    cache_threshold = float(os.getenv("CACHE_SIMILARITY_THRESHOLD", "0.9"))
    
    if provider == "ollama":
        # Ollama configuration
        ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        model_name = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
        embedding_model = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")
        
        return RAGEngine(
            cache_manager=cache_manager,
            provider="ollama",
            model_name=model_name,
            embedding_model=embedding_model,
            ollama_base_url=ollama_base_url,
            cache_threshold=cache_threshold
        )
    else:
        # Gemini configuration (default)
        google_api_key = os.getenv("GOOGLE_API_KEY")
        if not google_api_key:
            raise ValueError(
                "GOOGLE_API_KEY not found in environment variables. "
                "Either set GOOGLE_API_KEY for Gemini, or set LLM_PROVIDER=ollama to use Ollama."
            )
        
        model_name = os.getenv("GEMINI_MODEL", "gemini-pro")
        embedding_model = os.getenv("GEMINI_EMBEDDING_MODEL", "models/embedding-001")
        
        return RAGEngine(
            cache_manager=cache_manager,
            provider="gemini",
            google_api_key=google_api_key,
            model_name=model_name,
            embedding_model=embedding_model,
            cache_threshold=cache_threshold
        )


@app.command()
def load_documents(
    path: str = typer.Argument(..., help="Path to file or directory to load")
):
    """
    Load documents from a file or directory and index them in Redis.
    
    Supports: .txt, .md, .pdf files
    """
    try:
        console.print(f"[cyan]Loading documents from: {path}[/cyan]")
        
        cache_manager = get_cache_manager()
        rag_engine = get_rag_engine(cache_manager)
        
        # Display provider and model information
        display_provider_info(rag_engine)
        console.print()  # Empty line for spacing
        
        num_chunks = rag_engine.load_documents(path)
        
        console.print(
            Panel(
                f"[green]✓ Successfully loaded {num_chunks} document chunks into the Knowledge Index![/green]",
                title="Success",
                border_style="green"
            )
        )
        
    except FileNotFoundError as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)
    except ValueError as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Unexpected error: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def info():
    """
    Display system information and statistics.
    """
    try:
        cache_manager = get_cache_manager()
        stats = cache_manager.get_stats()
        
        # Get configuration
        provider = os.getenv("LLM_PROVIDER", "gemini").lower()
        cache_threshold = os.getenv("CACHE_SIMILARITY_THRESHOLD", "0.9")
        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
        
        # Create info table
        table = Table(title="System Information", show_header=True, header_style="bold magenta")
        table.add_column("Setting", style="cyan", no_wrap=True)
        table.add_column("Value", style="green")
        
        # Connection status
        status_icon = "✓" if stats["connected"] else "✗"
        status_text = "[green]Connected[/green]" if stats["connected"] else "[red]Disconnected[/red]"
        table.add_row("Redis Connection", f"{status_icon} {status_text}")
        table.add_row("Redis URL", redis_url)
        
        # Provider and model information
        table.add_row("LLM Provider", provider.upper())
        if provider == "ollama":
            model_name = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
            embedding_model = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")
            ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
            table.add_row("Ollama Model", model_name)
            table.add_row("Embedding Model", embedding_model)
            table.add_row("Ollama Base URL", ollama_base_url)
        else:
            model_name = os.getenv("GEMINI_MODEL", "gemini-pro")
            embedding_model = os.getenv("GEMINI_EMBEDDING_MODEL", "models/embedding-001")
            table.add_row("Gemini Model", model_name)
            table.add_row("Embedding Model", embedding_model)
        
        table.add_row("Cache Threshold", cache_threshold)
        
        # Index statistics
        table.add_row("Knowledge Index Chunks", str(stats["knowledge_index_count"]))
        table.add_row("Cache Index Entries", str(stats["cache_index_count"]))
        
        console.print(table)
        
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def chat():
    """
    Start an interactive chat session with semantic caching.
    
    Type 'exit' or 'quit' to end the session.
    """
    try:
        cache_manager = get_cache_manager()
        rag_engine = get_rag_engine(cache_manager)
        
        # Display provider and model information
        display_provider_info(rag_engine)
        console.print()  # Empty line for spacing
        
        # Verify Redis connection
        stats = cache_manager.get_stats()
        if not stats["connected"]:
            console.print("[red]Error: Cannot connect to Redis. Please check your REDIS_URL.[/red]")
            raise typer.Exit(1)
        
        # Check if knowledge base has documents
        if stats["knowledge_index_count"] == 0:
            console.print(
                "[yellow]Warning: Knowledge Index is empty. "
                "Load some documents first using 'load_documents' command.[/yellow]"
            )
        
        console.print(
            Panel(
                "[bold cyan]RAG Chat Session[/bold cyan]\n"
                "[dim]Type your questions. Type 'exit' or 'quit' to end.[/dim]",
                title="Welcome",
                border_style="cyan"
            )
        )
        
        while True:
            try:
                # Get user input
                user_query = Prompt.ask("\n[bold cyan]You[/bold cyan]")
                
                # Check for exit commands
                if user_query.lower() in ["exit", "quit", "q"]:
                    console.print("[cyan]Goodbye![/cyan]")
                    break
                
                if not user_query.strip():
                    continue
                
                # Process query
                console.print("[dim]Thinking...[/dim]")
                result = rag_engine.query(user_query)
                
                # Display result with appropriate styling
                if result["source"] == "cache":
                    # Cache hit
                    console.print(
                        Panel(
                            f"[green]✓ Cache Hit[/green] "
                            f"(Similarity: {result.get('similarity', 0):.2%})\n"
                            f"[dim]Original query: {result.get('cached_query', 'N/A')}[/dim]",
                            title="Semantic Cache",
                            border_style="green"
                        )
                    )
                    console.print(Markdown(result["response"]))
                    
                    # Display cost information
                    if result.get("cost"):
                        console.print()  # Empty line
                        display_cost_info(
                            result["cost"], 
                            all_costs=result.get("all_costs")
                        )
                else:
                    # Fresh generation
                    console.print(
                        Panel(
                            "[blue]✓ Generated via RAG[/blue]",
                            title="Fresh Generation",
                            border_style="blue"
                        )
                    )
                    console.print(Markdown(result["response"]))
                    
                    # Optionally show context chunks used
                    if result.get("context_chunks"):
                        num_chunks = result.get("num_chunks_used", 0)
                        console.print(f"[dim]Used {num_chunks} context chunk(s)[/dim]")
                    
                    # Display cost information
                    if result.get("cost"):
                        console.print()  # Empty line
                        display_cost_info(
                            result["cost"],
                            all_costs=result.get("all_costs")
                        )
                
            except KeyboardInterrupt:
                console.print("\n[cyan]Goodbye![/cyan]")
                break
            except Exception as e:
                console.print(f"[red]Error processing query: {e}[/red]")
        
    except ValueError as e:
        console.print(f"[red]Configuration Error: {e}[/red]")
        console.print("[yellow]Please ensure GOOGLE_API_KEY is set in your .env file.[/yellow]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()


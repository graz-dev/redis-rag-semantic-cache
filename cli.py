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

# Load environment variables
load_dotenv()

app = typer.Typer(help="RAG CLI Application with Semantic Caching")
console = Console()


def get_cache_manager() -> CacheManager:
    """Initialize and return CacheManager instance."""
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
    return CacheManager(redis_url)


def get_rag_engine(cache_manager: CacheManager) -> RAGEngine:
    """Initialize and return RAGEngine instance."""
    google_api_key = os.getenv("GOOGLE_API_KEY")
    if not google_api_key:
        raise ValueError("GOOGLE_API_KEY not found in environment variables")
    
    model_name = os.getenv("GEMINI_MODEL", "gemini-pro")
    embedding_model = os.getenv("GEMINI_EMBEDDING_MODEL", "models/embedding-001")
    cache_threshold = float(os.getenv("CACHE_SIMILARITY_THRESHOLD", "0.9"))
    
    return RAGEngine(
        cache_manager=cache_manager,
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
        model_name = os.getenv("GEMINI_MODEL", "gemini-pro")
        embedding_model = os.getenv("GEMINI_EMBEDDING_MODEL", "models/embedding-001")
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
        
        # Model information
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
        console.print(
            Panel(
                "[bold cyan]RAG Chat Session[/bold cyan]\n"
                "[dim]Type your questions. Type 'exit' or 'quit' to end.[/dim]",
                title="Welcome",
                border_style="cyan"
            )
        )
        
        cache_manager = get_cache_manager()
        rag_engine = get_rag_engine(cache_manager)
        
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


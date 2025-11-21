# RAG CLI Application with Semantic Caching

A simple Retrieval-Augmented Generation (RAG) CLI application with semantic caching capabilities, supporting both **Google Gemini** (cloud) and **Ollama** (local) models, built with **RedisVL**, Redis Stack, and LangChain.

## Quick Start

1. **Start Redis Stack** (required):
   ```bash
   # Using Docker Compose (recommended)
   docker-compose up -d
   
   # Or using Docker directly
   docker run -d --name redis-stack -p 6379:6379 -p 8001:8001 redis/redis-stack:latest
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment**:
   Create a `.env` file (see Configuration section)
   - For Gemini: Set `LLM_PROVIDER=gemini` and `GOOGLE_API_KEY`
   - For Ollama: Set `LLM_PROVIDER=ollama` (requires Ollama installed)

4. **Load documents and chat**:
   ```bash
   python main.py load-documents /path/to/documents
   python main.py chat
   ```

## Features

- **Dual Redis Indexes**: Separate indexes for knowledge base and semantic cache
- **Semantic Caching**: Automatically cache and retrieve similar queries to reduce LLM calls
- **Cost Tracking**: Real-time cost calculation and display for each query-answer pair
- **Cache Savings**: Visual display of cost savings when cache hits occur
- **Document Loading**: Support for PDF, TXT, and Markdown files
- **Interactive Chat**: Beautiful CLI interface with Rich formatting
- **Progress Tracking**: Visual progress bars during document processing
- **Multi-Provider Support**: Works with Google Gemini (cloud) and Ollama (local)

## Architecture

### Knowledge Index (RedisVL)
Stores embeddings of uploaded documents using `redisvl.index.SearchIndex`, enabling semantic search over your knowledge base.

### Semantic Cache (RedisVL)
Uses `redisvl.extensions.llmcache.SemanticCache` to store query-response pairs. When a similar query is detected (above similarity threshold), the cached response is returned immediately, skipping LLM generation.

### Semantic Caching Flow

1. User asks a question
2. Query is embedded using the configured embedding model (Gemini or Ollama)
3. **Cache Check**: Search Cache Index for similar queries
4. **Cache Hit**: If similarity ≥ threshold (default 0.9), return cached response
5. **Cache Miss**: 
   - Search Knowledge Index for relevant context
   - Generate answer using the configured LLM (Gemini or Ollama)
   - Cache the query-response pair for future use

## Prerequisites

- Python 3.8+
- Redis Stack (with vector search capabilities)
- **Either:**
  - Google Gemini API Key (for cloud-based LLM)
  - **OR** Ollama installed locally (for local LLM)

### Setting up Ollama (Optional - for Local Models)

If you want to use local models instead of Google Gemini, install and set up Ollama:

**Install Ollama:**

Visit [ollama.ai](https://ollama.ai) and download Ollama for your operating system, or use:

```bash
# macOS/Linux
curl -fsSL https://ollama.ai/install.sh | sh

# Or download from https://ollama.ai/download
```

**Install Ollama Python Package:**

```bash
# Install langchain-ollama (may require resolving version conflicts)
pip install langchain-ollama

# If you encounter dependency conflicts with langchain 0.3.0, try:
pip install langchain-ollama --no-deps
pip install langchain-core>=0.2.20,<0.3.0  # Install compatible version
```

**Pull Recommended Models:**

For the best RAG experience, we recommend:

```bash
# Pull the embedding model (required for semantic search)
ollama pull nomic-embed-text

# Pull a language model (choose one based on your hardware)
ollama pull llama3.1:8b      # Recommended: Good balance (8GB RAM)
ollama pull mistral          # Excellent quality (16GB+ RAM)
ollama pull llama3:8b        # Fast and efficient (8GB RAM)
ollama pull gemma2:9b        # Google's open model (9GB RAM)
```

**Verify Ollama is Running:**

```bash
# Check if Ollama is running
ollama list

# Test the API
curl http://localhost:11434/api/tags
```

**Model Recommendations for RAG:**

| Model | Size | RAM Required | Best For |
|-------|------|--------------|----------|
| `llama3.1:8b` | 8B | 8GB+ | **Recommended**: Best balance of quality and speed |
| `mistral` | 7B | 16GB+ | High quality responses, slightly slower |
| `llama3:8b` | 8B | 8GB+ | Fast responses, good quality |
| `gemma2:9b` | 9B | 9GB+ | Google's open model, good quality |

**Embedding Model:**
- `nomic-embed-text` (768 dimensions) - **Recommended**: Optimized for semantic search, same dimension as Gemini embeddings

### Setting up Redis Stack

#### Prerequisites

- [Docker](https://www.docker.com/get-started) installed on your system

#### Running Redis Stack with Docker


A `docker-compose.yml` file is provided for easier management:

```bash
# Start Redis Stack
docker-compose up -d

# Stop Redis Stack
docker-compose down

# View logs
docker-compose logs -f redis-stack
```

This method includes:
- Persistent data storage (volume)
- Health checks
- Automatic restart on failure

**Verify Redis is running:**

```bash
# Check if the container is running
docker ps | grep redis-stack

# Test Redis connection
docker exec -it redis-stack redis-cli ping
# Should return: PONG
```

**Access RedisInsight (Web UI):**

Open your browser and navigate to:
```
http://localhost:8001
```

RedisInsight provides a graphical interface to:
- View and manage your Redis data
- Monitor index statistics
- Execute Redis commands
- Inspect vector indexes


## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd redis-rag-semantic-cache
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```


1. Create a `.env` file in the project root:
```bash
# Copy the example file
cp env.example .env

# Then edit .env and configure your provider
```

**Configuration Options:**

**Option A: Using Google Gemini (Cloud)**
```env
LLM_PROVIDER=gemini
GOOGLE_API_KEY=your_google_api_key_here
REDIS_URL=redis://localhost:6379
```

**Option B: Using Ollama (Local)**
```env
LLM_PROVIDER=ollama
OLLAMA_MODEL=llama3.1:8b
OLLAMA_EMBEDDING_MODEL=nomic-embed-text
REDIS_URL=redis://localhost:6379
```

**Note:** See `env.example` for detailed explanations of all configuration options, including:
- Provider selection (Gemini vs Ollama)
- Model recommendations
- Redis URL formats for different setups
- Advanced configuration options

## Usage

### Load Documents

Load a single file or entire directory:

```bash
python main.py load-documents path/to/document.pdf
python main.py load-documents path/to/documents/
```

Supported formats: `.txt`, `.md`, `.pdf`

### View System Information

```bash
python main.py info
```

Displays:
- Redis connection status
- Active LLM provider (Gemini or Ollama)
- Active models (LLM and embedding)
- Number of documents in Knowledge Index
- Number of cached entries in Cache Index

### Interactive Chat

Start an interactive chat session:

```bash
python main.py chat
```

Features:
- **Cache Hit**: Displays in green with similarity score
- **Fresh Generation**: Displays in blue, indicating RAG-based generation
- Type `exit`, `quit`, or `q` to end the session


## Configuration

### Environment Variables

**Provider Selection:**
- `LLM_PROVIDER` (optional): Choose `"gemini"` or `"ollama"` (default: `"gemini"`)

**Gemini Configuration (if LLM_PROVIDER=gemini):**
- `GOOGLE_API_KEY` (required): Your Google Gemini API key
- `GEMINI_MODEL` (optional): Gemini model for generation (default: `gemini-pro`)
- `GEMINI_EMBEDDING_MODEL` (optional): Embedding model (default: `models/embedding-001`)

**Ollama Configuration (if LLM_PROVIDER=ollama):**
- `OLLAMA_BASE_URL` (optional): Ollama API URL (default: `http://localhost:11434`)
- `OLLAMA_MODEL` (optional): Ollama model for generation (default: `llama3.1:8b`)
- `OLLAMA_EMBEDDING_MODEL` (optional): Embedding model (default: `nomic-embed-text`)

**Common Configuration:**
- `REDIS_URL` (optional): Redis connection URL (default: `redis://localhost:6379`)
- `CACHE_SIMILARITY_THRESHOLD` (optional): Cache hit threshold 0.0-1.0 (default: `0.9`)
- `VECTOR_DIM` (optional): Vector dimension (default: `768` for both Gemini and nomic-embed-text)

### Adjusting Cache Sensitivity

Lower the `CACHE_SIMILARITY_THRESHOLD` (e.g., 0.8) to cache more aggressively, or raise it (e.g., 0.95) for stricter matching.

### Cost Tracking

The application tracks LLM API costs for each query-answer pair during chat sessions:

- **Cost Configuration**: Edit `costs-config.json` to configure pricing for different models
- **Real-time Display**: Costs are displayed after each query showing:
  - LLM cost projection for ALL configured models (what it would cost if using each model)
  - Token usage
  - Savings when cache hits occur (shows savings for each model)
- **Cache Savings**: When a cache hit occurs, the system shows savings for each model (the LLM cost that was avoided)
- **Cost Display**: Only LLM costs are shown (embedding costs are excluded)
- **Comparison Table**: Shows costs and savings for all models configured in `costs-config.json`

**Note**: 
- Cost calculations use token estimation (~4 characters per token)
- The table shows what the cost would be if you were using each model
- When cache hits, savings are calculated as the LLM cost that was avoided for each model
- Costs are only displayed during chat sessions, not during document loading
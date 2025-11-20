# RAG CLI Application with Semantic Caching

A powerful Retrieval-Augmented Generation (RAG) CLI application with semantic caching capabilities, built with Google Gemini, Redis Stack, and LangChain.

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
   Create a `.env` file with your `GOOGLE_API_KEY` and `REDIS_URL`

4. **Load documents and chat**:
   ```bash
   python main.py load-documents /path/to/documents
   python main.py chat
   ```

## Features

- **Dual Redis Indexes**: Separate indexes for knowledge base and semantic cache
- **Semantic Caching**: Automatically cache and retrieve similar queries to reduce LLM calls
- **Document Loading**: Support for PDF, TXT, and Markdown files
- **Interactive Chat**: Beautiful CLI interface with Rich formatting
- **Progress Tracking**: Visual progress bars during document processing

## Architecture

### Knowledge Index
Stores embeddings of uploaded documents, enabling semantic search over your knowledge base.

### Cache Index (Semantic Cache)
Stores query-response pairs. When a similar query is detected (above similarity threshold), the cached response is returned immediately, skipping LLM generation.

### Semantic Caching Flow

1. User asks a question
2. Query is embedded using Gemini
3. **Cache Check**: Search Cache Index for similar queries
4. **Cache Hit**: If similarity ≥ threshold (default 0.9), return cached response
5. **Cache Miss**: 
   - Search Knowledge Index for relevant context
   - Generate answer using Gemini LLM
   - Cache the query-response pair for future use

## Prerequisites

- Python 3.8+
- Redis Stack (with vector search capabilities)
- Google Gemini API Key

### Setting up Redis Stack

#### Prerequisites

- [Docker](https://www.docker.com/get-started) installed on your system

#### Running Redis Stack with Docker

**Option 1: Using Docker Compose (Recommended)**

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

**Option 2: Using Docker directly**

```bash
docker run -d --name redis-stack -p 6379:6379 -p 8001:8001 redis/redis-stack:latest
```

This command will:
- Download the `redis/redis-stack:latest` image (if not already present)
- Start a container named `redis-stack` in detached mode (`-d`)
- Expose Redis on port `6379` (default Redis port)
- Expose RedisInsight (web UI) on port `8001`

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

**Useful Docker Commands:**

```bash
# Stop Redis Stack
docker stop redis-stack

# Start Redis Stack (if stopped)
docker start redis-stack

# Restart Redis Stack
docker restart redis-stack

# View Redis logs
docker logs redis-stack

# Remove Redis Stack container (⚠️ This will delete all data)
docker rm -f redis-stack

# Run Redis Stack with persistent data (recommended for production)
docker run -d \
  --name redis-stack \
  -p 6379:6379 \
  -p 8001:8001 \
  -v redis-data:/data \
  redis/redis-stack:latest

# Or simply use docker-compose (includes persistent storage)
docker-compose up -d
```

**Troubleshooting:**

- **Port already in use**: If port 6379 or 8001 is already in use, you can change the ports:
  ```bash
  docker run -d --name redis-stack -p 6380:6379 -p 8002:8001 redis/redis-stack:latest
  ```
  Then update your `.env` file: `REDIS_URL=redis://localhost:6380`

- **Container name exists**: If you get an error about the container name existing:
  ```bash
  docker rm -f redis-stack  # Remove existing container
  # Then run the docker run command again
  ```

- **Check if Redis is accessible**:
  ```bash
  redis-cli -h localhost -p 6379 ping
  # Or using Docker:
  docker exec -it redis-stack redis-cli ping
  ```

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

4. Create a `.env` file in the project root:
```bash
# Copy the example file
cp env.example .env

# Then edit .env and add your GOOGLE_API_KEY
```

Or create it manually with the following content:
```env
# Google Gemini API Key (required)
GOOGLE_API_KEY=your_google_api_key_here

# Redis Connection URL
# When using docker-compose: redis://localhost:6379
REDIS_URL=redis://localhost:6379

# Gemini Model Configuration (optional)
GEMINI_MODEL=gemini-pro
GEMINI_EMBEDDING_MODEL=models/embedding-001

# Semantic Cache Similarity Threshold (optional, 0.0 to 1.0)
CACHE_SIMILARITY_THRESHOLD=0.9
```

**Note:** See `env.example` for detailed explanations of each configuration option, including Redis URL formats for different setups (docker-compose, Docker, Redis Cloud, etc.).

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
- Active Gemini models
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

### Example Session

```
$ python main.py chat

┌─────────────────────────────────────┐
│ Welcome                             │
├─────────────────────────────────────┤
│ RAG Chat Session                    │
│ Type your questions. Type 'exit' or │
│ 'quit' to end.                      │
└─────────────────────────────────────┘

You: What is machine learning?

┌─────────────────────────────────────┐
│ Fresh Generation                    │
├─────────────────────────────────────┤
│ ✓ Generated via RAG                 │
└─────────────────────────────────────┘

Machine learning is a subset of artificial intelligence...

You: Can you explain machine learning?

┌─────────────────────────────────────┐
│ Semantic Cache                      │
├─────────────────────────────────────┤
│ ✓ Cache Hit (Similarity: 95.23%)   │
│ Original query: What is machine     │
│ learning?                           │
└─────────────────────────────────────┘

Machine learning is a subset of artificial intelligence...
```

## Project Structure

```
redis-rag-semantic-cache/
├── main.py              # Entry point
├── cli.py               # CLI commands (Typer)
├── rag_engine.py        # RAG logic and document processing
├── cache_manager.py     # Redis connection and index management
├── requirements.txt     # Python dependencies
├── docker-compose.yml   # Docker Compose configuration for Redis
├── env.example          # Example environment variables file
├── .env                 # Environment variables (create from env.example)
└── README.md           # This file
```

## Configuration

### Environment Variables

- `GOOGLE_API_KEY` (required): Your Google Gemini API key
- `REDIS_URL` (optional): Redis connection URL (default: `redis://localhost:6379`)
- `GEMINI_MODEL` (optional): Gemini model for generation (default: `gemini-pro`)
- `GEMINI_EMBEDDING_MODEL` (optional): Embedding model (default: `models/embedding-001`)
- `CACHE_SIMILARITY_THRESHOLD` (optional): Cache hit threshold 0.0-1.0 (default: `0.9`)

### Adjusting Cache Sensitivity

Lower the `CACHE_SIMILARITY_THRESHOLD` (e.g., 0.8) to cache more aggressively, or raise it (e.g., 0.95) for stricter matching.

## How It Works

1. **Document Indexing**:
   - Documents are loaded and split into chunks (1000 chars, 200 overlap)
   - Each chunk is embedded using Gemini Embeddings
   - Embeddings are stored in Redis Knowledge Index

2. **Query Processing**:
   - User query is embedded
   - Cache Index is searched first (semantic similarity)
   - If cache miss, Knowledge Index is searched for context
   - LLM generates answer from context
   - Query-response pair is cached

3. **Semantic Caching**:
   - Uses cosine similarity to find similar queries
   - Threshold-based matching prevents false positives
   - Significantly reduces API calls and latency

## Troubleshooting

### Redis Connection Issues

Ensure Redis Stack is running:
```bash
docker ps | grep redis-stack
```

Test connection:
```bash
redis-cli ping
```

### API Key Issues

Verify your `.env` file contains a valid `GOOGLE_API_KEY`:
```bash
python -c "from dotenv import load_dotenv; import os; load_dotenv(); print('Key found' if os.getenv('GOOGLE_API_KEY') else 'Key missing')"
```

### Empty Knowledge Base

If `info` shows 0 documents, load some documents first:
```bash
python main.py load-documents /path/to/your/documents
```

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.


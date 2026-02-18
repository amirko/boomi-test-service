# RAG-as-a-Service

A complete **Retrieval-Augmented Generation (RAG)** service with hybrid search capabilities, built with FastAPI, Qdrant, and local embeddings.

## 🎯 Overview

This project demonstrates a production-ready RAG service featuring:

- **Hybrid Search**: Combines dense (semantic) and sparse (keyword) search using Reciprocal Rank Fusion (RRF)
- **Multi-Tenant Isolation**: Logical separation of data by tenant_id
- **Local Embeddings**: Fast sentence-transformers model (all-MiniLM-L6-v2)
- **LLM Integration**: Support for Groq, OpenAI, and Ollama
- **Performance Optimized**: Parallel execution, streaming, and circuit breakers
- **Production Ready**: Docker-based deployment, comprehensive tests, and health checks

## 🏗️ Architecture

```
┌─────────────┐
│   Client    │
└──────┬──────┘
       │
       ▼
┌─────────────────────────────────┐
│       FastAPI Service           │
│  ┌───────────────────────────┐  │
│  │  POST /documents          │  │
│  │  POST /search             │  │
│  │  POST /search-with-summary│  │
│  │  DELETE /documents/:id    │  │
│  │  GET /health              │  │
│  └───────────────────────────┘  │
│                                  │
│  ┌──────────────┐ ┌───────────┐ │
│  │  Embedding   │ │   LLM     │ │
│  │  Service     │ │  Service  │ │
│  │ (local model)│ │ (Groq/OAI)│ │
│  └──────────────┘ └───────────┘ │
└────────┬─────────────────────────┘
         │
         ▼
┌────────────────────┐
│   Qdrant Vector DB │
│  (Hybrid Search)   │
│ • Dense Vectors    │
│ • Sparse Vectors   │
│ • HNSW Index       │
└────────────────────┘
```

## 🚀 Performance Targets

| Component | Target Latency | Optimization |
|-----------|----------------|--------------|
| Embedding | ~30ms | Local sentence-transformers |
| Retrieval | ~150ms | Qdrant (HNSW + Sparse) |
| LLM TTFT | ~300ms | Groq API or Llama-3-8B |
| **Search** | **< 800ms** | **Parallel execution** |
| **Search + Summary** | **< 2.5s** | **Streaming + Circuit Breaker** |

## 🛠️ Technology Stack

- **API Framework**: FastAPI
- **Vector Database**: Qdrant
- **Embeddings**: sentence-transformers (all-MiniLM-L6-v2, 384 dimensions)
- **LLM Providers**: Groq, OpenAI, Ollama
- **Language**: Python 3.11
- **Deployment**: Docker & Docker Compose

## 📦 Quick Start

### Prerequisites

- Docker and Docker Compose
- LLM API key (Groq recommended) or local Ollama installation

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/amirko/boomi-test-service.git
cd boomi-test-service
```

2. **Configure environment**
```bash
cp .env.example .env
# Edit .env and add your LLM_API_KEY
```

3. **Start services**
```bash
docker-compose up --build
```

The services will be available at:
- **API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **Qdrant Dashboard**: http://localhost:6333/dashboard

## 📚 API Usage Examples

### Health Check

```bash
curl http://localhost:8000/health
```

### Ingest Documents

```bash
curl -X POST "http://localhost:8000/documents" \
  -H "Content-Type: application/json" \
  -d '{
    "tenant_id": "user_123",
    "document_id": "doc_1",
    "content": "FastAPI is a modern, fast web framework for building APIs with Python.",
    "metadata": {"source": "docs", "category": "python"}
  }'
```

### Search (< 800ms)

```bash
curl -X POST "http://localhost:8000/search" \
  -H "Content-Type: application/json" \
  -d '{
    "tenant_id": "user_123",
    "query": "What is FastAPI?",
    "top_k": 5
  }'
```

**Response:**
```json
{
  "results": [
    {
      "document_id": "doc_1",
      "content": "FastAPI is a modern, fast web framework...",
      "score": 0.87,
      "metadata": {"source": "docs", "category": "python"}
    }
  ],
  "latency_ms": 156.3
}
```

### Search with Summary (< 2.5s)

```bash
curl -X POST "http://localhost:8000/search-with-summary" \
  -H "Content-Type: application/json" \
  -d '{
    "tenant_id": "user_123",
    "query": "What is FastAPI?",
    "top_k": 3
  }'
```

**Response:**
```json
{
  "results": [...],
  "summary": "FastAPI is a modern web framework for building APIs with Python. It is designed to be fast, easy to use, and provides automatic API documentation.",
  "latency_ms": 1847.2,
  "search_latency_ms": 168.5,
  "llm_latency_ms": 1678.7
}
```

### Delete Tenant Documents

```bash
curl -X DELETE "http://localhost:8000/documents/user_123"
```

## 🎯 Key Features

### 1. Hybrid Search with RRF

Combines two complementary search strategies:

- **Dense Search**: Semantic similarity using embeddings (captures meaning)
- **Sparse Search**: Keyword matching (captures exact terms)

Results are merged using **Reciprocal Rank Fusion (RRF)**:
```
RRF_score = Σ(1 / (k + rank))
```

### 2. Multi-Tenant Isolation

- Each document is tagged with `tenant_id`
- All searches are filtered by `tenant_id`
- Simple, fast, and effective for demo purposes

### 3. Performance Optimizations

- **Parallel Execution**: Dense and sparse searches run concurrently using `asyncio.gather()`
- **Local Embeddings**: No API calls for embeddings (~20-30ms)
- **HNSW Indexing**: Approximate nearest neighbor search in Qdrant
- **Streaming**: LLM responses stream for better perceived performance
- **Context Pruning**: Only top 3-5 chunks sent to LLM
- **Circuit Breaker**: Graceful degradation on LLM timeout

### 4. LLM Integration

Supports multiple providers with consistent interface:

**Groq** (Recommended for speed):
```bash
LLM_PROVIDER=groq
LLM_API_KEY=your_groq_key
LLM_MODEL=llama-3-8b-8192
```

**OpenAI**:
```bash
LLM_PROVIDER=openai
LLM_API_KEY=your_openai_key
LLM_MODEL=gpt-3.5-turbo
```

**Ollama** (Local):
```bash
LLM_PROVIDER=ollama
LLM_MODEL=llama3:8b
OLLAMA_BASE_URL=http://localhost:11434
```

## 🧪 Testing

### Run All Tests

```bash
docker-compose exec api pytest tests/ -v
```

### Run Specific Tests

```bash
# API tests
docker-compose exec api pytest tests/test_api.py -v

# Search and RRF tests
docker-compose exec api pytest tests/test_search.py -v
```

### Test Coverage

- Document ingestion
- Hybrid search
- Multi-tenant isolation
- Health checks
- RRF algorithm
- Embedding generation
- Latency requirements

## 📁 Project Structure

```
boomi-test-service/
├── app/
│   ├── __init__.py
│   ├── main.py                 # FastAPI application
│   ├── config.py               # Configuration management
│   ├── models.py               # Pydantic models
│   ├── services/
│   │   ├── __init__.py
│   │   ├── embedding.py        # Local embedding service
│   │   ├── qdrant_client.py    # Qdrant vector DB client
│   │   ├── llm_service.py      # LLM integration
│   │   └── search.py           # Hybrid search with RRF
│   └── utils/
│       ├── __init__.py
│       ├── circuit_breaker.py  # Circuit breaker pattern
│       └── rrf.py              # Reciprocal Rank Fusion
├── tests/
│   ├── __init__.py
│   ├── test_api.py             # API endpoint tests
│   └── test_search.py          # Search and RRF tests
├── docker-compose.yml          # Service orchestration
├── Dockerfile                  # API container
├── requirements.txt            # Python dependencies
├── .env.example                # Environment template
├── .gitignore
└── README.md
```

## ⚙️ Configuration

All configuration is managed through environment variables (see `.env.example`):

### Qdrant Settings
- `QDRANT_HOST`: Qdrant host (default: localhost)
- `QDRANT_PORT`: Qdrant port (default: 6333)
- `COLLECTION_NAME`: Collection name (default: rag_documents)

### Embedding Settings
- `EMBEDDING_MODEL`: Model name (default: sentence-transformers/all-MiniLM-L6-v2)

### LLM Settings
- `LLM_PROVIDER`: Provider (groq, openai, ollama)
- `LLM_API_KEY`: API key for provider
- `LLM_MODEL`: Model name

### Performance Settings
- `LLM_TIMEOUT`: LLM timeout in seconds (default: 2.0)
- `SEARCH_TIMEOUT`: Search timeout in seconds (default: 0.8)

## 🔧 Troubleshooting

### Qdrant Connection Issues

```bash
# Check if Qdrant is running
curl http://localhost:6333/

# View Qdrant logs
docker-compose logs qdrant
```

### LLM API Rate Limits

If you hit rate limits:
1. Use Ollama for local inference (no rate limits)
2. Increase `LLM_TIMEOUT` in `.env`
3. Implement request queuing

### Memory Issues

The embedding model requires ~400MB RAM. If running on limited resources:
1. Use a smaller model (e.g., `all-MiniLM-L6-v2` is already minimal)
2. Increase Docker memory allocation
3. Consider using API-based embeddings

### Timeout Tuning

Adjust timeouts based on your infrastructure:

```bash
# For slower systems
LLM_TIMEOUT=5.0
SEARCH_TIMEOUT=1.5

# For faster systems
LLM_TIMEOUT=1.5
SEARCH_TIMEOUT=0.5
```

## 🚀 Production Considerations

### Scaling

- **Horizontal**: Run multiple API instances behind a load balancer
- **Vertical**: Increase Qdrant resources for larger datasets
- **Caching**: Add Redis for frequently accessed results

### Security

- Add authentication/authorization
- Use HTTPS in production
- Rotate API keys regularly
- Implement rate limiting

### Monitoring

- Add Prometheus metrics
- Set up health check alerts
- Monitor latency percentiles (P50, P95, P99)
- Track search quality metrics

### Data Management

- Implement batch ingestion for large datasets
- Add document versioning
- Set up automated backups
- Consider data retention policies

## 📝 API Endpoints

### POST /documents
Ingest a document with tenant isolation.

**Request:**
```json
{
  "tenant_id": "string",
  "document_id": "string",
  "content": "string",
  "metadata": {}
}
```

### POST /search
Execute hybrid search.

**Request:**
```json
{
  "tenant_id": "string",
  "query": "string",
  "top_k": 5
}
```

### POST /search-with-summary
Search with LLM-generated summary.

**Request:**
```json
{
  "tenant_id": "string",
  "query": "string",
  "top_k": 3
}
```

### DELETE /documents/{tenant_id}
Delete all documents for a tenant.

### GET /health
Health check endpoint.

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new features
4. Ensure all tests pass
5. Submit a pull request

## 📄 License

This project is provided as a demo/reference implementation. Feel free to use and modify as needed.

## 🙏 Acknowledgments

- FastAPI for the excellent web framework
- Qdrant for the vector database
- sentence-transformers for embeddings
- Groq for fast LLM inference

## 📧 Support

For issues and questions:
- Open an issue on GitHub
- Check existing documentation
- Review the code comments

---

**Built with ❤️ for demonstrating RAG best practices**
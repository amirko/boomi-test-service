# Quick Start Guide

This guide will help you get the RAG-as-a-Service up and running in 5 minutes.

## Prerequisites

- Docker and Docker Compose installed
- Groq API key (or Ollama running locally)

## Step 1: Clone and Configure

```bash
# Clone the repository
git clone https://github.com/amirko/boomi-test-service.git
cd boomi-test-service

# Copy environment template
cp .env.example .env
```

## Step 2: Get an API Key

### Option A: Groq (Recommended - Fast & Free Tier)

1. Sign up at https://console.groq.com/
2. Create an API key
3. Add to `.env`:
```bash
LLM_PROVIDER=groq
LLM_API_KEY=your_groq_api_key_here
LLM_MODEL=llama-3-8b-8192
```

### Option B: Ollama (Local - No API Key Needed)

1. Install Ollama from https://ollama.ai/
2. Pull a model: `ollama pull llama3:8b`
3. Update `.env`:
```bash
LLM_PROVIDER=ollama
LLM_MODEL=llama3:8b
```

## Step 3: Start Services

```bash
docker-compose up --build
```

Wait for services to start (may take a few minutes on first run):
- ✓ Qdrant starts
- ✓ Embedding model downloads
- ✓ API server starts

## Step 4: Test the API

Open another terminal and try these examples:

### 1. Health Check
```bash
curl http://localhost:8000/health | jq
```

### 2. Ingest a Document
```bash
curl -X POST "http://localhost:8000/documents" \
  -H "Content-Type: application/json" \
  -d '{
    "tenant_id": "demo_user",
    "document_id": "doc_1",
    "content": "FastAPI is a modern, fast (high-performance), web framework for building APIs with Python 3.7+ based on standard Python type hints.",
    "metadata": {"source": "docs", "category": "python"}
  }' | jq
```

### 3. Search
```bash
curl -X POST "http://localhost:8000/search" \
  -H "Content-Type: application/json" \
  -d '{
    "tenant_id": "demo_user",
    "query": "What is FastAPI?",
    "top_k": 5
  }' | jq
```

### 4. Search with AI Summary
```bash
curl -X POST "http://localhost:8000/search-with-summary" \
  -H "Content-Type: application/json" \
  -d '{
    "tenant_id": "demo_user",
    "query": "Tell me about FastAPI",
    "top_k": 3
  }' | jq
```

## Step 5: Explore the API

Visit http://localhost:8000/docs for interactive API documentation.

## Troubleshooting

### Service Won't Start

```bash
# Check logs
docker-compose logs -f

# Restart services
docker-compose down
docker-compose up --build
```

### Out of Memory

The embedding model needs ~400MB RAM. If you're on a small machine:
- Increase Docker memory allocation to at least 2GB
- Or use API-based embeddings instead

### LLM Timeouts

If the LLM is slow, increase the timeout in `.env`:
```bash
LLM_TIMEOUT=5.0
```

## Next Steps

- Read the [README.md](README.md) for complete documentation
- Check out the [Architecture](#) section
- Try the [test suite](#): `docker-compose exec api pytest tests/ -v`
- Explore performance optimizations

## Resources

- API Documentation: http://localhost:8000/docs
- Qdrant Dashboard: http://localhost:6333/dashboard
- GitHub Repository: https://github.com/amirko/boomi-test-service

---

**Need help?** Open an issue on GitHub or check the Troubleshooting section in README.md

# ChitraAI Backend - Semantic Movie Recommendation Engine

ChitraAI is a production-grade, AI-powered semantic movie recommendation system. The backend is built using FastAPI, Polars, and Sentence-Transformers, featuring local hybrid candidate retrieval (semantic search + BM25 keyword matching) and dynamic metadata reranking.

---

## 🏗 System Architecture Overview

The backend is built around a highly resilient dual-pathway retrieval system, employing online cloud services as the primary engine and local, in-memory datasets as self-contained fallbacks:

1. **Query Routing & Intent Parsing**:
   * **Bypasses**: Instantly bypasses the LLM for direct movie titles, person search shortcuts (e.g., *"films by Nolan"*), or simple genre words.
   * **Primary Intent extraction**: Deconstructs conversational queries into structured parameters using LLM-based structured JSON outputs.
   * **Fallback Intent extraction**: If LLM endpoints are unreachable, a local regex-based parser extracts basic genre and mood targets.
2. **Dynamic Candidate Retrieval**:
   * **Primary Path**: Queries the live **TMDb Discover API** and matches vectors in real-time against **Qdrant Cloud** (leveraging cosine similarity on dense embeddings).
   * **Fallback Path**: If TMDb/Qdrant is down or returns 0 matches, the engine falls back to a **local in-memory retrieval engine** using Polars LazyFrames (filtered over the local 45,433 canonical movie parquet) and a local numpy-based embeddings matrix.
3. **Local Hybrid Ranking & Scoring**:
   * Compiles matches using semantic relevance, BM25 exact keyword matches (leveraging a memory-resident `rank-bm25` index), and metadata Jaccard scoring.
   * Reranks candidates using a weighted formula (55% Semantic + 25% BM25 + 20% Metadata).
4. **Live Enrichment & Explanations**:
   * Integrates credits, ratings, trailers, and streaming providers via TMDb.
   * Runs the parsed search profile against the selected movies to generate natural language explanations highlighting why each movie matches.

---

## 📁 Directory Structure

```text
backend/
├── app/
│   ├── api/
│   │   ├── routes/          # API route handlers (search, recommend, movies, health, stats)
│   │   └── deps.py          # Dependency injection singletons
│   ├── core/
│   │   ├── config.py        # Settings management (Pydantic BaseSettings)
│   │   ├── logging_config.py# Loguru configuration
│   │   └── model_manager.py # Heavyweight SentenceTransformer and LLM client manager
│   ├── datasets/            # Parquet canonical databases and caching SQLite DBs
│   ├── embeddings/          # Precomputed movie metadata vector embeddings
│   └── services/            # Core business logic (local retrieval, TMDb API, OpenAI parsing)
├── main.py                  # Server entry point & FastAPI application initialization
├── requirements.txt         # Production-locked dependencies list
└── tests/                   # Pytest suite
```

---

## ⚙ Environment Variables (`.env`)

Create a `.env` file in the `backend/` directory:

| Variable | Description | Default / Example |
| :--- | :--- | :--- |
| `GROQ_API_KEY` | Groq API access key (for Intent Extraction) | `gsk_...` |
| `LLM_PROVIDER` | LLM Provider for intent extraction | `groq` |
| `LLM_MODEL` | LLM Model for intent extraction | `llama-3.1-8b-instant` |
| `TMDB_API_KEY` | TMDb API access key (for live posters/metadata enrichment) | `ddf1a4bc...` |
| `EMBEDDING_MODEL` | Sentence-Transformer model name | `all-MiniLM-L6-v2` |
| `DEVICE` | Model compute device | `cpu` or `cuda` |
| `LOG_LEVEL` | Application logging verbosity | `INFO` |

---

## 🚀 Installation & Local Setup

### 1. Prerequisites
* Python 3.10 to 3.13
* Virtual environment (venv or conda)

### 2. Setup steps
```bash
# Navigate to backend directory
cd backend

# Create a virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
.\venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Running the Server
```bash
# Run the FastAPI server in development mode
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```
Visit http://127.0.0.1:8000/docs for the interactive Swagger API documentation.

---

## 🧪 Testing and Validation

### Run Unit & Integration Tests
```bash
python -m pytest
```

### Run Latency & QA Validation Suite
```bash
python scratch/run_full_validation.py
```

---

## 🔌 API Endpoints Table

| Method | Endpoint | Description | Tags |
| :--- | :--- | :--- | :--- |
| `GET` | `/api/v1/health` | Returns server health and cache/DB status | Health |
| `GET` | `/api/v1/stats` | Returns database statistics and total resident vectors | System Stats |
| `GET` | `/api/v1/search` | Performs hybrid semantic/keyword search over movies | Search |
| `GET` | `/api/v1/recommendations/semantic` | Recommends movies matching natural language query | Recommendation |
| `POST`| `/api/v1/recommendations` | Recommends movies via JSON payload | Recommendation |
| `GET` | `/api/v1/recommendations/movie/{movie_id}`| Recommends movies similar to a source movie | Recommendation |
| `GET` | `/api/v1/movies/autocomplete` | Returns title suggestions matching prefix | Movie |
| `GET` | `/api/v1/movies/{movie_id}` | Returns detailed enriched movie metadata | Movie |

---

## 🚢 Production Deployment

For production deployments, execute using Uvicorn standard worker configuration behind a reverse proxy (e.g. Nginx):

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```
Ensure that `GROQ_API_KEY` and `TMDB_API_KEY` are mounted securely.

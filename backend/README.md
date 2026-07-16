# ChitraAI Backend - Semantic Movie Recommendation Engine

ChitraAI is a production-grade, AI-powered semantic movie recommendation system. The backend is built using FastAPI, Polars, and Sentence-Transformers, featuring local hybrid candidate retrieval (semantic search + BM25 keyword matching) and dynamic metadata reranking.

---

## 🏗 System Architecture Overview

The backend uses a local hybrid retrieval architecture, eliminating external database dependencies (e.g. Qdrant) for fast, lightweight in-memory searching:

1. **Query Parsing & Intent Extraction (NLU)**:
   * Translates natural language user requests into structured search constraints (preferred/avoided genres, actors, directors, moods, themes, years, runtimes) using **OpenAI's Structured Output API**.
2. **Local Hard Filtering**:
   * Uses highly optimized **Polars LazyFrames** to filter the 45,433 canonical movies based on hard year, runtime, language, and genre exclusion boundaries.
3. **Local Hybrid Retrieval**:
   * **Semantic Search**: Computes local cosine similarity between the query embedding and precomputed movie embeddings matrix using vectorized NumPy dot products.
   * **BM25 Keyword Search**: Evaluates exact keyword matches using a memory-resident `rank-bm25` Okapi index built over title, taglines, keywords, cast, directors, and Wikipedia plot summaries.
   * **Metadata Jaccard Match**: Calculates Jaccard overlap on categorical preferences.
4. **Vectorized Fusion & Reranking**:
   * Scores are combined via Polars expressions (55% Semantic + 25% BM25 + 20% Metadata) to isolate the top candidate pool.
5. **Weighted Reranking & Enrichment**:
   * The candidate pool is reranked by the `WeightedScorer` combining popularity, rating, cast/director boosts, and semantic relevance, followed by real-time concurrency-throttled TMDb API details enrichment (posters, certifications, streaming providers).

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

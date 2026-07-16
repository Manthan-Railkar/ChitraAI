# 🎬 Movie Recommendation System - Backend Architecture

> **Version:** 2.0
> **Architecture:** Hybrid AI + Local Semantic Retrieval
> **Status:** Backend Redesign
> **Future Extension:** Fine-tuned Ranking LLM

---

# Overview

The backend follows a modular hybrid architecture that combines the strengths of commercial LLMs with local semantic retrieval and a custom fine-tuned ranking model.

Instead of relying on Retrieval-Augmented Generation (RAG) or forcing an LLM to memorize millions of movies, the system separates the recommendation pipeline into independent stages.

Each stage is responsible for one specific task.

- OpenAI → Natural language understanding
- Local Database → Movie retrieval
- Fine-tuned Local LLM → Movie ranking
- Local LLM → Response generation

This makes the system significantly faster, cheaper, easier to maintain, and more scalable than traditional RAG pipelines.

---

# Goals

The redesigned backend should satisfy the following goals:

- Low API token usage
- No RAG
- No external vector database
- Fully local semantic retrieval
- Modular architecture
- Easy future fine-tuning
- Fast recommendation generation
- Production-ready design

---

# High-Level Architecture

```text
                        User Query
                             │
                             ▼
               OpenAI Intent Extraction API
                             │
                             ▼
                  Structured Recommendation JSON
                             │
                             ▼
                 Local Movie Retrieval Engine
                             │
             ┌───────────────┴────────────────┐
             │                                │
      Structured Filters              Semantic Similarity
             │                                │
             └───────────────┬────────────────┘
                             │
                             ▼
                  Candidate Movie Scoring
                             │
                             ▼
                    Top 50 Candidate Movies
                             │
                             ▼
          Fine-tuned Local Movie Ranking LLM
                             │
                             ▼
                  Top Recommended Movies
                             │
                             ▼
             Lightweight Local Response Generator
                             │
                             ▼
                    Final Recommendation
```

---

# Stage 1 — User Query

Example:

> I want a dark psychological thriller like Se7en but without horror.

This query is **not** sent to the local model.

Instead, it is sent directly to OpenAI using a lightweight prompt.

---

# Stage 2 — OpenAI Intent Extraction

The OpenAI API is **only** responsible for understanding user language.

It does NOT recommend movies.

It does NOT search databases.

It only converts natural language into structured JSON.

Example output:

```json
{
  "genres": [
    "Thriller",
    "Crime"
  ],
  "mood": [
    "Dark",
    "Psychological"
  ],
  "themes": [
    "Serial Killer"
  ],
  "similar_movies": [
    "Se7en"
  ],
  "language": "English",
  "year_range": null,
  "runtime": null,
  "actors": [],
  "directors": [],
  "avoid_genres": [
    "Horror"
  ]
}
```

Advantages:

- Extremely low token usage
- Better language understanding
- No movie knowledge required
- Easily replaceable later

---

# Stage 3 — Local Movie Retrieval

After receiving the structured intent, the backend performs all movie retrieval locally.

No API calls are made after this stage.

No RAG pipeline is used.

No vector database is used.

The retrieval engine consists of three phases.

---

## Phase 1 — Structured Filtering

The backend first removes irrelevant movies using metadata.

Possible filters:

- Genres
- Language
- Runtime
- Release year
- Adult content
- Country
- Certification
- Cast
- Directors
- Production companies

Example:

```
Genre = Thriller

Language = English

Exclude Horror

Year > 1990
```

This typically reduces millions of records to a few hundred candidates.

---

## Phase 2 — Semantic Similarity

The remaining movies are ranked using embeddings.

Embeddings are generated locally using a Sentence Transformer.

Recommended models:

- all-MiniLM-L6-v2
- BAAI/bge-small-en-v1.5
- all-mpnet-base-v2

Each movie has one combined embedding generated from:

- Title
- Overview
- Genres
- Keywords
- Tagline
- Director
- Cast
- Wikipedia Plot
- Themes

Example embedding text:

```
Title:
Interstellar

Genres:
Science Fiction
Adventure
Drama

Overview:
A team of explorers travel through a wormhole...

Keywords:
space
black hole
time dilation

Director:
Christopher Nolan

Cast:
Matthew McConaughey
Anne Hathaway

Themes:
Hope
Sacrifice
Love
```

Cosine similarity is then computed between the user query embedding and movie embeddings.

---

## Phase 3 — Weighted Candidate Scoring

The retrieval engine combines multiple scores.

Example:

Final Score =

0.40 × Semantic Similarity

+

0.20 × Genre Match

+

0.15 × Keyword Match

+

0.10 × MovieLens Rating

+

0.10 × TMDB Popularity

+

0.05 × IMDb Rating

This produces the Top 50 candidate movies.

The goal is **high recall**, not selecting the final recommendation.

---

# Stage 4 — Fine-Tuned Ranking LLM

This stage is **not implemented initially**.

It will be added after retrieval is completed.

The model will NOT memorize movies.

Instead, it learns how to rank candidate movies.

Input:

- Original user query
- Structured intent JSON
- Top 50 retrieved movies

Example:

```
User:

Need something emotional like Interstellar.

Candidate Movies:

Arrival

Gravity

The Martian

Ad Astra

Contact

...
```

Output:

```
1. Arrival

2. Contact

3. Gravity

4. Ad Astra

5. The Martian
```

The model simply learns which retrieved movies best satisfy the user's intent.

Recommended base models:

- Qwen 2.5 3B Instruct
- Qwen 3 4B Instruct
- Gemma 3
- Llama 3.2 3B

Training method:

- LoRA
- QLoRA

---

# Stage 5 — Response Generation

Once ranking is complete, only the Top 5 movies are passed to a lightweight local model.

Example response:

> Arrival is the strongest recommendation because it combines emotional storytelling with thoughtful science fiction while emphasizing human relationships, closely matching your request.

This stage generates explanations only.

Movie selection has already been completed.

---

# Database Design

The backend uses a unified movie database built from multiple datasets.

Raw datasets:

```
datasets/

├── imdb/
├── tmdb/
├── movielens/
└── wikipedia/
```

These datasets are merged into one processed database.

Final movie schema:

```
Movie_ID

Title

Original Title

Overview

Wikipedia Plot

Genres

Keywords

Tagline

Cast

Director

Runtime

Language

Country

Release Year

IMDb Rating

MovieLens Rating

Popularity

Vote Count

Production Companies

Collections

Poster Path

Embedding Text
```

---

# Dataset Sources

## TMDB

Provides:

- Movie metadata
- Genres
- Keywords
- Popularity
- Posters
- Runtime
- Budget
- Revenue

---

## IMDb

Provides:

- Cast
- Crew
- Writers
- Directors
- Ratings
- Alternate titles

---

## MovieLens

Provides:

- User ratings
- User tags
- Rating statistics

---

## Wikipedia

Provides:

- Plot summaries
- Themes
- Story descriptions
- Additional metadata

---

# Project Structure

```
backend/

├── api/
│
├── intent/
│      openai_intent.py
│
├── retrieval/
│      filters.py
│      embeddings.py
│      semantic_search.py
│      scorer.py
│
├── database/
│      loader.py
│      merger.py
│
├── ranking/
│      (future)
│
├── generation/
│      (future)
│
└── config/
```

---

# Retrieval Flow

```
User Query
      │
      ▼
OpenAI Intent Parser
      │
      ▼
Structured JSON
      │
      ▼
Metadata Filters
      │
      ▼
Semantic Search
      │
      ▼
Weighted Ranking
      │
      ▼
Top 50 Movies
```

---

# Future Fine-Tuning Pipeline

Future training data format:

```
Instruction

Rank the following movies.

Input

User Query

+

Intent JSON

+

Candidate Movies

Output

Ranked Movies
```

Training objective:

Query

↓

Ranking

NOT

Query

↓

Movie Memorization

---

# Advantages of This Architecture

- Very low OpenAI token usage
- No expensive RAG infrastructure
- Fully local retrieval
- Fast inference
- Modular components
- Easy retraining
- Updatable movie database
- No LLM hallucination during retrieval
- Fine-tuning focused only on ranking
- Easy to scale

---

# Future Roadmap

### Phase 1

- Merge datasets
- Build unified movie database
- Generate embeddings
- Build retrieval engine

---

### Phase 2

- OpenAI intent extraction
- Semantic retrieval
- Weighted scoring
- Return Top 50 candidates

---

### Phase 3

- Generate ranking dataset
- Fine-tune local ranking LLM
- Replace heuristic ranking

---

### Phase 4

- Personalized recommendations
- User history
- Watchlist
- Feedback learning
- Reinforcement-based ranking
- Online model updates

---

# Final Vision

The final system separates the recommendation pipeline into specialized components:

- **OpenAI** understands what the user wants.
- **Local retrieval engine** efficiently searches millions of movies using structured filters and semantic similarity.
- **Fine-tuned ranking LLM** learns how to rank retrieved candidates according to user intent.
- **Lightweight local generator** produces natural explanations for the final recommendations.

This hybrid architecture minimizes API costs, avoids the complexity of RAG, keeps the movie database fully local and updatable, and provides a scalable foundation for future personalization and ranking improvements.
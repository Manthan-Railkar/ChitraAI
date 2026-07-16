export interface BackendMovie {
  id: string;
  title: string;
  original_title?: string;
  overview?: string;
  genres: string[];
  directors: string[];
  cast: string[];
  release_year?: number;
  rating_value?: number;
  popularity?: number;
  vote_count?: number;
  poster_path?: string;
  backdrop_path?: string;
  trailer_url?: string;
  streaming_providers?: string[];
  certification?: string;
  runtime_minutes?: number;
  semantic_score: number;
  boosted_semantic_score?: number;
  reranked_score: number;
  recommendation_reason: string;
}

export interface QueryUnderstandingResult {
  search_intent: string;
  mood?: string;
  genres?: string[];
  themes?: string[];
  actors?: string[];
  directors?: string[];
  reference_movies?: string[];
  user_preferences?: string;
  release_year_constraints?: {
    exact_year?: number;
    start_year?: number;
    end_year?: number;
  };
}

export interface PaginationInfo {
  page: number;
  limit: number;
  total_results: number;
}

export interface ExecutionStatistics {
  elapsed_time_ms: number;
  source: string;
}

export interface RecommendationResponse {
  query: string;
  pagination: PaginationInfo;
  execution_statistics: ExecutionStatistics;
  understanding?: QueryUnderstandingResult;
  results: BackendMovie[];
}

export interface HealthCheckResponse {
  status: string;
  version: string;
  config: {
    qdrant_url: string;
    qdrant_collection: string;
    embedding_model: string;
    device: string;
    log_level: string;
  };
}

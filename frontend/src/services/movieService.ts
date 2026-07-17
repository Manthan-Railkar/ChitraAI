import apiClient from '@/api/client';
import type {
  DatasetPosterResponse,
  HealthCheckResponse,
  RecommendationResponse,
  BackendMovie,
} from '@/types/api';

/**
 * Helper to map backend's StandardizedMovie structure to frontend's BackendMovie structure.
 */
export function mapBackendMovieToFrontend(m: any): BackendMovie {
  return {
    id: String(m.tmdb_id || m.id || ''),
    title: m.title || 'Unknown Title',
    original_title: m.original_title || m.title || '',
    overview: m.overview || '',
    genres: m.genres || [],
    directors: m.directors || [],
    cast: m.cast || [],
    release_year: m.release_year || undefined,
    rating_value: m.rating || m.rating_value || undefined,
    popularity: m.popularity || undefined,
    vote_count: m.vote_count || undefined,
    poster_path: m.poster_path || undefined,
    backdrop_path: m.backdrop_path || undefined,
    trailer_url: m.trailer_url || undefined,
    streaming_providers: m.streaming_providers || [],
    certification: m.certification || undefined,
    runtime_minutes: m.runtime || m.runtime_minutes || undefined,
    semantic_score: m.confidence_score || m.semantic_score || 0.0,
    boosted_semantic_score: m.confidence_score || m.boosted_semantic_score || 0.0,
    reranked_score: m.retrieval_score || m.reranked_score || 0.0,
    recommendation_reason: m.recommendation_reason || '',
  };
}

/**
 * Helper to map backend's RecommendationResponse schema to frontend's RecommendationResponse schema.
 */
function mapBackendResponseToFrontend(response: any): RecommendationResponse {
  const recommendations = response.recommendations || [];
  const metadata = response.metadata || {};
  const executionStats = metadata.execution_statistics || {};
  const understanding = metadata.understanding || {};

  const mappedResults = recommendations.map(mapBackendMovieToFrontend);

  return {
    query: response.query || '',
    pagination: metadata.pagination || {
      page: 1,
      limit: recommendations.length,
      total_results: recommendations.length,
    },
    execution_statistics: {
      elapsed_time_ms: executionStats.elapsed_time_ms || 0,
      source: executionStats.source || 'database',
    },
    understanding: {
      search_intent: understanding.search_intent || 'recommendation',
      mood: understanding.mood,
      genres: understanding.genres || [],
      themes: understanding.themes || [],
      actors: understanding.actors || [],
      directors: understanding.directors || [],
      reference_movies: understanding.reference_movies || [],
      user_preferences: understanding.user_preferences,
      release_year_constraints: understanding.release_year_constraints,
    },
    results: mappedResults,
  };
}

export const movieService = {
  /**
   * Check if backend FastAPI server is healthy and online
   */
  async checkHealth(): Promise<HealthCheckResponse> {
    const response: any = await apiClient.get('/health');
    return {
      status: response.status || 'unhealthy',
      version: response.version || '1.0.0',
      config: {
        qdrant_url: response.qdrant_url || '',
        qdrant_collection: response.qdrant_collection || '',
        embedding_model: response.embedding_model || '',
        device: response.device || '',
        log_level: response.log_level || '',
      },
    };
  },

  /**
   * Get semantic recommendations based on a natural language query
   * Uses POST /api/v1/recommend as single source of truth.
   */
  async getRecommendations(q: string, limit: number = 10): Promise<RecommendationResponse> {
    const response: any = await apiClient.post('/recommend', {
      query: q,
      limit: limit,
    });
    return mapBackendResponseToFrontend(response);
  },

  /**
   * Get recommendations similar to a given movie ID
   */
  async getSimilarMovies(movieId: string, limit: number = 10): Promise<RecommendationResponse> {
    const response: any = await apiClient.get(`/recommendations/movie/${movieId}`, {
      params: { limit },
    });
    return mapBackendResponseToFrontend(response);
  },

  /**
   * Get detailed metadata for a specific movie by TMDb ID or UUID
   */
  async getMovieDetails(movieId: string): Promise<BackendMovie> {
    const response: any = await apiClient.get(`/movies/${movieId}`);
    const movieData = response.movie || response;
    return mapBackendMovieToFrontend(movieData);
  },

  /** Get a randomized poster sample from the application movie dataset. */
  async getDatasetPosters(limit: number = 24): Promise<DatasetPosterResponse> {
    return apiClient.get('/movies/dataset-posters', { params: { limit } });
  },
};

export default movieService;

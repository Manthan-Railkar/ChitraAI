import { useQuery } from '@tanstack/react-query';
import movieService from '@/services/movieService';
import type { HealthCheckResponse, RecommendationResponse, BackendMovie } from '@/types/api';
import type { ApiError } from '@/api/client';

export const useHealthCheck = (options?: { enabled?: boolean; refetchInterval?: number }) => {
  return useQuery<HealthCheckResponse, ApiError>({
    queryKey: ['health'],
    queryFn: () => movieService.checkHealth(),
    retry: false,
    ...options,
  });
};

export const useRecommendations = (
  q: string,
  limit: number = 10,
  options?: { enabled?: boolean }
) => {
  return useQuery<RecommendationResponse, ApiError>({
    queryKey: ['recommendations', q, limit],
    queryFn: () => movieService.getRecommendations(q, limit),
    retry: 1,
    ...options,
  });
};

export const useSimilarMovies = (
  movieId: string,
  limit: number = 10,
  options?: { enabled?: boolean }
) => {
  return useQuery<RecommendationResponse, ApiError>({
    queryKey: ['similar-movies', movieId, limit],
    queryFn: () => movieService.getSimilarMovies(movieId, limit),
    retry: 1,
    ...options,
  });
};

export const useMovieDetails = (
  movieId: string,
  options?: { enabled?: boolean }
) => {
  return useQuery<BackendMovie, ApiError>({
    queryKey: ['movie-details', movieId],
    queryFn: () => movieService.getMovieDetails(movieId),
    retry: 1,
    ...options,
  });
};

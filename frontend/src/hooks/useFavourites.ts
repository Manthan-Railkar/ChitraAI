import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { useAuth } from '@/contexts/AuthContext';
import favouritesService from '@/services/favouritesService';
import type { FavouriteMovie } from '@/services/favouritesService';

/**
 * Fetch all favourites for the current logged-in user
 */
export const useFavourites = () => {
  const { user } = useAuth();

  return useQuery<FavouriteMovie[]>({
    queryKey: ['favourites', user?.id],
    queryFn: () => favouritesService.getFavourites(user!.id),
    enabled: !!user,
    staleTime: 60_000,
  });
};

/**
 * Check if a specific movie is in the user's favourites
 */
export const useIsFavourite = (movieId: string) => {
  const { user } = useAuth();

  return useQuery<boolean>({
    queryKey: ['is-favourite', user?.id, movieId],
    queryFn: () => favouritesService.isFavourite(user!.id, movieId),
    enabled: !!user && !!movieId,
    staleTime: 30_000,
  });
};

/**
 * Toggle a movie as favourite (add if not exists, remove if exists).
 * Uses optimistic updates for snappy UI.
 */
export const useToggleFavourite = () => {
  const { user } = useAuth();
  const queryClient = useQueryClient();

  const addMutation = useMutation({
    mutationFn: (movie: Omit<FavouriteMovie, 'user_id'>) =>
      favouritesService.addFavourite({ ...movie, user_id: user!.id }),
    onMutate: async (movie) => {
      // Cancel in-flight queries
      await queryClient.cancelQueries({ queryKey: ['favourites', user?.id] });
      await queryClient.cancelQueries({ queryKey: ['is-favourite', user?.id, movie.movie_id] });

      // Optimistic: set is-favourite to true
      queryClient.setQueryData(['is-favourite', user?.id, movie.movie_id], true);

      // Optimistic: prepend to favourites list
      const prevFavourites = queryClient.getQueryData<FavouriteMovie[]>(['favourites', user?.id]);
      if (prevFavourites) {
        queryClient.setQueryData<FavouriteMovie[]>(['favourites', user?.id], [
          { ...movie, user_id: user!.id, created_at: new Date().toISOString() },
          ...prevFavourites,
        ]);
      }

      return { prevFavourites };
    },
    onError: (_err, movie, context) => {
      // Rollback
      queryClient.setQueryData(['is-favourite', user?.id, movie.movie_id], false);
      if (context?.prevFavourites) {
        queryClient.setQueryData(['favourites', user?.id], context.prevFavourites);
      }
    },
    onSettled: (_data, _error, movie) => {
      queryClient.invalidateQueries({ queryKey: ['favourites', user?.id] });
      queryClient.invalidateQueries({ queryKey: ['is-favourite', user?.id, movie.movie_id] });
    },
  });

  const removeMutation = useMutation({
    mutationFn: (movieId: string) => favouritesService.removeFavourite(user!.id, movieId),
    onMutate: async (movieId) => {
      await queryClient.cancelQueries({ queryKey: ['favourites', user?.id] });
      await queryClient.cancelQueries({ queryKey: ['is-favourite', user?.id, movieId] });

      queryClient.setQueryData(['is-favourite', user?.id, movieId], false);

      const prevFavourites = queryClient.getQueryData<FavouriteMovie[]>(['favourites', user?.id]);
      if (prevFavourites) {
        queryClient.setQueryData<FavouriteMovie[]>(
          ['favourites', user?.id],
          prevFavourites.filter((f) => f.movie_id !== movieId)
        );
      }

      return { prevFavourites };
    },
    onError: (_err, movieId, context) => {
      queryClient.setQueryData(['is-favourite', user?.id, movieId], true);
      if (context?.prevFavourites) {
        queryClient.setQueryData(['favourites', user?.id], context.prevFavourites);
      }
    },
    onSettled: (_data, _error, movieId) => {
      queryClient.invalidateQueries({ queryKey: ['favourites', user?.id] });
      queryClient.invalidateQueries({ queryKey: ['is-favourite', user?.id, movieId] });
    },
  });

  return { addFavourite: addMutation, removeFavourite: removeMutation };
};

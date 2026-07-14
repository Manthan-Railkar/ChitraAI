import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import { APP_CONFIG } from '@/config';

interface MovieStore {
  favorites: string[];
  recentSearches: string[];
  addFavorite: (movieId: string) => void;
  removeFavorite: (movieId: string) => void;
  addRecentSearch: (query: string) => void;
  clearRecentSearches: () => void;
}

export const useMovieStore = create<MovieStore>()(
  persist(
    (set) => ({
      favorites: [],
      recentSearches: [],
      addFavorite: (movieId) =>
        set((state) => ({
          favorites: state.favorites.includes(movieId)
            ? state.favorites
            : [...state.favorites, movieId],
        })),
      removeFavorite: (movieId) =>
        set((state) => ({
          favorites: state.favorites.filter((id) => id !== movieId),
        })),
      addRecentSearch: (query) =>
        set((state) => {
          const cleanedQuery = query.trim();
          if (!cleanedQuery) return state;
          const filtered = state.recentSearches.filter(
            (q) => q.toLowerCase() !== cleanedQuery.toLowerCase()
          );
          return {
            recentSearches: [cleanedQuery, ...filtered].slice(0, 10), // Keep last 10
          };
        }),
      clearRecentSearches: () => set({ recentSearches: [] }),
    }),
    {
      name: APP_CONFIG.localStorageKeys.favorites,
    }
  )
);

export default useMovieStore;

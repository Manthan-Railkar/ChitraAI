import { supabase } from '@/lib/supabaseClient';

export interface FavouriteMovie {
  id?: string;
  user_id: string;
  movie_id: string;
  title: string;
  poster_path?: string | null;
  genres?: string[];
  rating_value?: number | null;
  release_year?: number | null;
  overview?: string | null;
  runtime_minutes?: number | null;
  created_at?: string;
}

export const favouritesService = {
  /**
   * Get all favourites for a user
   */
  async getFavourites(userId: string): Promise<FavouriteMovie[]> {
    const { data, error } = await supabase
      .from('favourites')
      .select('*')
      .eq('user_id', userId)
      .order('created_at', { ascending: false });

    if (error) {
      console.error('[Favourites] Fetch error:', error.message);
      throw new Error(error.message);
    }
    return (data ?? []) as FavouriteMovie[];
  },

  /**
   * Add a movie to favourites
   */
  async addFavourite(movie: FavouriteMovie): Promise<FavouriteMovie> {
    const { data, error } = await supabase
      .from('favourites')
      .upsert(
        {
          user_id: movie.user_id,
          movie_id: movie.movie_id,
          title: movie.title,
          poster_path: movie.poster_path ?? null,
          genres: movie.genres ?? [],
          rating_value: movie.rating_value ?? null,
          release_year: movie.release_year ?? null,
          overview: movie.overview ?? null,
          runtime_minutes: movie.runtime_minutes ?? null,
        },
        { onConflict: 'user_id,movie_id' }
      )
      .select()
      .single();

    if (error) {
      console.error('[Favourites] Add error:', error.message);
      throw new Error(error.message);
    }
    return data as FavouriteMovie;
  },

  /**
   * Remove a movie from favourites
   */
  async removeFavourite(userId: string, movieId: string): Promise<void> {
    const { error } = await supabase
      .from('favourites')
      .delete()
      .eq('user_id', userId)
      .eq('movie_id', movieId);

    if (error) {
      console.error('[Favourites] Remove error:', error.message);
      throw new Error(error.message);
    }
  },

  /**
   * Check if a specific movie is favourited by a user
   */
  async isFavourite(userId: string, movieId: string): Promise<boolean> {
    const { data, error } = await supabase
      .from('favourites')
      .select('id')
      .eq('user_id', userId)
      .eq('movie_id', movieId)
      .maybeSingle();

    if (error) {
      console.error('[Favourites] Check error:', error.message);
      return false;
    }
    return !!data;
  },
};

export default favouritesService;

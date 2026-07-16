import React from 'react';
import { Heart } from 'lucide-react';
import { useAuth } from '@/contexts/AuthContext';
import { useIsFavourite, useToggleFavourite } from '@/hooks/useFavourites';
import { toast } from 'sonner';

interface FavouriteButtonProps {
  movieId: string;
  title: string;
  posterPath?: string | null;
  genres?: string[];
  ratingValue?: number | null;
  releaseYear?: number | null;
  overview?: string | null;
  runtimeMinutes?: number | null;
  /** 'sm' = small icon-only, 'md' = icon with text */
  size?: 'sm' | 'md';
  className?: string;
}

export const FavouriteButton: React.FC<FavouriteButtonProps> = ({
  movieId,
  title,
  posterPath,
  genres,
  ratingValue,
  releaseYear,
  overview,
  runtimeMinutes,
  size = 'md',
  className = '',
}) => {
  const { user } = useAuth();
  const { data: isFav, isLoading } = useIsFavourite(movieId);
  const { addFavourite, removeFavourite } = useToggleFavourite();

  // Don't render if user is not logged in
  if (!user) return null;

  const handleToggle = (e: React.MouseEvent) => {
    e.stopPropagation();
    e.preventDefault();

    if (isFav) {
      removeFavourite.mutate(movieId, {
        onSuccess: () => toast.success(`Removed "${title}" from favourites`),
        onError: () => toast.error('Failed to remove from favourites'),
      });
    } else {
      addFavourite.mutate(
        {
          movie_id: movieId,
          title,
          poster_path: posterPath,
          genres: genres ?? [],
          rating_value: ratingValue,
          release_year: releaseYear,
          overview: overview,
          runtime_minutes: runtimeMinutes,
        },
        {
          onSuccess: () => toast.success(`Added "${title}" to favourites`),
          onError: () => toast.error('Failed to add to favourites'),
        }
      );
    }
  };

  const isPending = addFavourite.isPending || removeFavourite.isPending || isLoading;

  if (size === 'sm') {
    return (
      <button
        onClick={handleToggle}
        disabled={isPending}
        className={`group/fav w-8 h-8 rounded-full border flex items-center justify-center transition-all duration-300 cursor-pointer ${
          isFav
            ? 'border-rose-500/40 bg-rose-500/15 text-rose-400 hover:bg-rose-500/25'
            : 'border-white/10 bg-white/[0.03] text-white/40 hover:text-rose-400 hover:border-rose-500/30 hover:bg-rose-500/5'
        } ${isPending ? 'opacity-50 pointer-events-none' : ''} ${className}`}
        title={isFav ? 'Remove from favourites' : 'Add to favourites'}
      >
        <Heart
          className={`w-3.5 h-3.5 transition-all duration-300 ${
            isFav ? 'fill-current scale-110' : 'group-hover/fav:scale-110'
          }`}
        />
      </button>
    );
  }

  return (
    <button
      onClick={handleToggle}
      disabled={isPending}
      className={`inline-flex items-center gap-1.5 px-4 py-2 border rounded-full text-[11px] font-bold uppercase tracking-wider transition-all duration-300 cursor-pointer ${
        isFav
          ? 'border-rose-500/30 bg-rose-500/10 text-rose-400 hover:bg-rose-500/20'
          : 'border-white/10 hover:border-rose-500/20 bg-white/[0.03] text-white hover:text-rose-400 hover:bg-rose-500/5'
      } ${isPending ? 'opacity-50 pointer-events-none' : ''} ${className}`}
    >
      <Heart
        className={`w-3.5 h-3.5 transition-all duration-300 ${isFav ? 'fill-current' : ''}`}
      />
      {isFav ? 'Favourited' : 'Favourite'}
    </button>
  );
};

export default FavouriteButton;

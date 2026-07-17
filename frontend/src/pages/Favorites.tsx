import React, { useEffect, useRef, useMemo } from 'react';
import { Link } from 'react-router-dom';
import { MainLayout } from '@/layouts/MainLayout';
import { useAuth } from '@/contexts/AuthContext';
import { useFavourites } from '@/hooks/useFavourites';
import { Heart, Loader2, LogIn, Sparkles } from 'lucide-react';
import { gsap } from 'gsap';
import { PosterSpiral } from '@/components/favourites/PosterSpiral';

// ── Famous movie & series poster URLs (fallback when signed out or empty) ──
const FAMOUS_POSTER_URLS = [
  'https://image.tmdb.org/t/p/w500/xlaY2zyzMfkhk0HSC5VUwzoZPU1.jpg', // Inception
  'https://image.tmdb.org/t/p/w500/uOOtwVbSr4QDjAGIifLDwpb2Pdl.jpg', // Stranger Things
  'https://image.tmdb.org/t/p/w500/qJ2tW6WMUDux911r6m7haRef0WH.jpg', // The Dark Knight
  'https://image.tmdb.org/t/p/w500/ztkUQFLlC19CCMYHW9o1zWhJRNq.jpg', // Breaking Bad
  'https://image.tmdb.org/t/p/w500/yQvGrMoipbRoddT0ZR8tPoR7NfX.jpg', // Interstellar
  'https://image.tmdb.org/t/p/w500/1XS1oqL89opfnbLl8WnZY1O1uJx.jpg', // Game of Thrones
  'https://image.tmdb.org/t/p/w500/vQWk5YBFWF4bZaofAbv0tShwBvQ.jpg', // Pulp Fiction
  'https://image.tmdb.org/t/p/w500/dXNAPwY7VrqMAo51EKhhCJfaGb5.jpg', // The Matrix
  'https://image.tmdb.org/t/p/w500/jSziioSwPVrOy9Yow3XhWIBDjq1.jpg', // Fight Club
  'https://image.tmdb.org/t/p/w500/3bhkrj58Vtu7enYsRolD1fZdja1.jpg', // The Godfather
  'https://image.tmdb.org/t/p/w500/7IiTTgloJzvGI1TAYymCfbfl3vT.jpg', // Parasite
  'https://image.tmdb.org/t/p/w500/reEMJA1uzscCbkpeRJeTT2bjqUp.jpg', // Money Heist
];

export const Favorites: React.FC = () => {
  const { user, isLoading: authLoading } = useAuth();
  const { data: favourites, isLoading: favsLoading } = useFavourites();

  const isLoading = authLoading || favsLoading;

  // ── Refs for GSAP text animation ──
  const textContainerRef = useRef<HTMLDivElement>(null);
  const hasAnimated = useRef(false);

  // ── Animate the right-side text on mount ──
  useEffect(() => {
    if (isLoading || hasAnimated.current) return;
    hasAnimated.current = true;

    const container = textContainerRef.current;
    if (!container) return;

    const words = container.querySelectorAll('.anim-word');
    const elements = container.querySelectorAll('.anim-el');

    gsap.set([...Array.from(words), ...Array.from(elements)], { opacity: 0, y: 30 });

    const tl = gsap.timeline({ delay: 0.3 });
    tl.to(words, {
      opacity: 1,
      y: 0,
      duration: 0.6,
      stagger: 0.12,
      ease: 'power3.out',
    });
    tl.to(
      elements,
      {
        opacity: 1,
        y: 0,
        duration: 0.5,
        stagger: 0.1,
        ease: 'power2.out',
      },
      '-=0.2'
    );
  }, [isLoading]);

  // ── Poster URLs ──
  const posterUrls = useMemo(() => {
    console.log('[Favorites Page] Favourites from DB:', favourites);
    if (!user || !favourites || favourites.length === 0) {
      return FAMOUS_POSTER_URLS;
    }

    // Extract poster URLs from user favourites
    const userPosters = favourites
      .filter((f) => f.poster_path)
      .map((f) => {
        const p = f.poster_path!;
        return p.startsWith('http') ? p : `https://image.tmdb.org/t/p/w500${p}`;
      });

    console.log('[Favorites Page] Parsed posterUrls for Spiral:', userPosters);

    // If user has fewer than 6, pad with famous posters
    if (userPosters.length < 6) {
      const needed = 12 - userPosters.length;
      const padding = FAMOUS_POSTER_URLS.slice(0, needed);
      return [...userPosters, ...padding];
    }

    return userPosters;
  }, [user, favourites]);

  // ── Loading ──
  if (isLoading) {
    return (
      <MainLayout>
        <div className="flex items-center justify-center min-h-[80vh]">
          <Loader2 className="w-8 h-8 text-rose-500 animate-spin" />
        </div>
      </MainLayout>
    );
  }

  // ── Determine CTA ──
  const isSignedIn = !!user;
  const hasFavourites = isSignedIn && favourites && favourites.length > 0;

  return (
    <MainLayout>
      <div className="w-full h-[calc(100vh-112px)] flex flex-col lg:flex-row overflow-hidden bg-black select-none">
        {/* ── Left: Poster Spiral ── */}
        <div className="w-full lg:w-[55%] h-[55vh] lg:h-full relative">
          {/* Gradient overlays for seamless blending */}
          <div className="absolute inset-0 z-10 pointer-events-none">
            {/* Right edge fade on desktop */}
            <div className="hidden lg:block absolute right-0 top-0 bottom-0 w-32 bg-gradient-to-l from-black to-transparent" />
            {/* Bottom edge fade on mobile */}
            <div className="lg:hidden absolute bottom-0 left-0 right-0 h-24 bg-gradient-to-t from-black to-transparent" />
            {/* Top subtle vignette */}
            <div className="absolute top-0 left-0 right-0 h-20 bg-gradient-to-b from-black/40 to-transparent" />
          </div>

          <PosterSpiral posterUrls={posterUrls} />
        </div>

        {/* ── Right: Animated Text Panel ── */}
        <div className="w-full lg:w-[45%] h-[45vh] lg:h-full flex items-center justify-center px-6 sm:px-10 lg:px-16 relative">
          {/* Subtle background glow */}
          <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[400px] h-[400px] rounded-full bg-rose-500/[0.03] blur-[120px] pointer-events-none" />

          <div
            ref={textContainerRef}
            className="relative z-10 flex flex-col items-center lg:items-start text-center lg:text-left gap-6 max-w-[480px]"
          >
            {/* Badge */}
            <div className="anim-el inline-flex items-center gap-2 px-3 py-1.5 rounded-full border border-rose-500/20 bg-rose-500/5 backdrop-blur-md">
              <Heart className="w-3.5 h-3.5 text-rose-400 fill-rose-400/30" />
              <span
                className="text-[10px] font-bold text-rose-400 uppercase tracking-widest"
                style={{ fontFamily: 'Plus Jakarta Sans, sans-serif' }}
              >
                {hasFavourites ? `${favourites!.length} Saved` : 'Collection'}
              </span>
            </div>

            {/* Headline with per-word animation */}
            <h1
              className="text-4xl sm:text-5xl lg:text-6xl font-black uppercase tracking-tight leading-[1.05]"
              style={{ fontFamily: 'Plus Jakarta Sans, sans-serif' }}
            >
              <span className="anim-word inline-block text-white">Find </span>
              <span className="anim-word inline-block text-transparent bg-clip-text bg-gradient-to-r from-rose-500 via-purple-500 to-blue-500">
                all{' '}
              </span>
              <span className="anim-word inline-block text-white">your </span>
              <br className="hidden sm:block" />
              <span className="anim-word inline-block text-transparent bg-clip-text bg-gradient-to-r from-purple-400 to-rose-400">
                favourites{' '}
              </span>
              <span className="anim-word inline-block text-white">at </span>
              <br className="hidden sm:block" />
              <span className="anim-word inline-block text-transparent bg-clip-text bg-gradient-to-r from-blue-400 via-purple-400 to-rose-500">
                one place
              </span>
            </h1>

            {/* Subtitle */}
            <p
              className="anim-el text-sm text-white/45 max-w-[360px] leading-relaxed"
              style={{ fontFamily: 'Inter, sans-serif' }}
            >
              {isSignedIn
                ? 'Your personal cinema collection — every movie and series you loved, spinning in an infinite loop.'
                : 'Sign in to start building your personal collection. Every movie you love, preserved beautifully.'}
            </p>

            {/* CTA Button */}
            <div className="anim-el flex items-center gap-4 mt-2">
              {!isSignedIn ? (
                <Link
                  to="/profile"
                  className="group inline-flex items-center gap-2.5 px-7 py-3.5 bg-white text-black hover:bg-gradient-to-r hover:from-rose-500 hover:to-purple-600 hover:text-white text-xs font-black uppercase tracking-wider rounded-full transition-all duration-500 cursor-pointer shadow-lg hover:shadow-rose-500/20 hover:scale-105 active:scale-95"
                >
                  <LogIn className="w-4 h-4" />
                  Sign In to Continue
                </Link>
              ) : !hasFavourites ? (
                <Link
                  to="/search"
                  className="group inline-flex items-center gap-2.5 px-7 py-3.5 bg-white text-black hover:bg-gradient-to-r hover:from-rose-500 hover:to-purple-600 hover:text-white text-xs font-black uppercase tracking-wider rounded-full transition-all duration-500 cursor-pointer shadow-lg hover:shadow-rose-500/20 hover:scale-105 active:scale-95"
                >
                  <Sparkles className="w-4 h-4" />
                  Start Discovering
                </Link>
              ) : (
                <Link
                  to="/search"
                  className="group inline-flex items-center gap-2.5 px-7 py-3.5 border border-white/10 hover:border-rose-500/30 bg-white/[0.03] hover:bg-rose-500/5 text-white text-xs font-black uppercase tracking-wider rounded-full transition-all duration-500 cursor-pointer hover:scale-105 active:scale-95"
                >
                  <Sparkles className="w-4 h-4 text-rose-400" />
                  Discover More
                </Link>
              )}
            </div>

            {/* Decorative bottom line */}
            <div className="anim-el w-16 h-px bg-gradient-to-r from-rose-500/40 via-purple-500/40 to-transparent mt-2" />
          </div>
        </div>
      </div>
    </MainLayout>
  );
};

export default Favorites;

import React, { useEffect, useMemo, useRef } from 'react';
import { Link } from 'react-router-dom';
import { MainLayout } from '@/layouts/MainLayout';
import { useAuth } from '@/contexts/AuthContext';
import { useDatasetPosters } from '@/hooks/useMovieApi';
import { Database, LogIn, Sparkles } from 'lucide-react';
import { gsap } from 'gsap';
import { PosterSpiral } from '@/components/favourites/PosterSpiral';

// TMDb popular-title fallback used when the local dataset is unavailable.
const TMDB_POPULAR_POSTER_URLS = [
  'https://image.tmdb.org/t/p/w500/xlaY2zyzMfkhk0HSC5VUwzoZPU1.jpg',
  'https://image.tmdb.org/t/p/w500/qJ2tW6WMUDux911r6m7haRef0WH.jpg',
  'https://image.tmdb.org/t/p/w500/yQvGrMoipbRoddT0ZR8tPoR7NfX.jpg',
  'https://image.tmdb.org/t/p/w500/vQWk5YBFWF4bZaofAbv0tShwBvQ.jpg',
  'https://image.tmdb.org/t/p/w500/dXNAPwY7VrqMAo51EKhhCJfaGb5.jpg',
  'https://image.tmdb.org/t/p/w500/jSziioSwPVrOy9Yow3XhWIBDjq1.jpg',
  'https://image.tmdb.org/t/p/w500/3bhkrj58Vtu7enYsRolD1fZdja1.jpg',
  'https://image.tmdb.org/t/p/w500/7IiTTgloJzvGI1TAYymCfbfl3vT.jpg',
  'https://image.tmdb.org/t/p/w500/1XS1oqL89opfnbLl8WnZY1O1uJx.jpg',
  'https://image.tmdb.org/t/p/w500/ztkUQFLlC19CCMYHW9o1zWhJRNq.jpg',
  'https://image.tmdb.org/t/p/w500/reEMJA1uzscCbkpeRJeTT2bjqUp.jpg',
  'https://image.tmdb.org/t/p/w500/uOOtwVbSr4QDjAGIifLDwpb2Pdl.jpg',
];

const toPosterUrl = (posterPath: string) =>
  posterPath.startsWith('http') ? posterPath : `https://image.tmdb.org/t/p/w500${posterPath}`;

export const Dataset: React.FC = () => {
  const { user, isLoading: authLoading } = useAuth();
  const { data: datasetPosters } = useDatasetPosters();
  const textContainerRef = useRef<HTMLDivElement>(null);
  const hasAnimated = useRef(false);

  useEffect(() => {
    if (authLoading || hasAnimated.current) return;
    hasAnimated.current = true;

    const container = textContainerRef.current;
    if (!container) return;

    const words = container.querySelectorAll('.anim-word');
    const elements = container.querySelectorAll('.anim-el');
    gsap.set([...Array.from(words), ...Array.from(elements)], { opacity: 0, y: 30 });

    const timeline = gsap.timeline({ delay: 0.3 });
    timeline.to(words, { opacity: 1, y: 0, duration: 0.6, stagger: 0.12, ease: 'power3.out' });
    timeline.to(
      elements,
      { opacity: 1, y: 0, duration: 0.5, stagger: 0.1, ease: 'power2.out' },
      '-=0.2'
    );
  }, [authLoading]);

  const posterUrls = useMemo(() => {
    const datasetUrls = datasetPosters?.poster_paths.map(toPosterUrl) ?? [];
    return datasetUrls.length >= 6 ? datasetUrls : TMDB_POPULAR_POSTER_URLS;
  }, [datasetPosters]);

  const isSignedIn = Boolean(user);

  return (
    <MainLayout>
      <div className="w-full h-[calc(100vh-112px)] flex flex-col lg:flex-row overflow-hidden bg-black select-none">
        <div className="w-full lg:w-[55%] h-[55vh] lg:h-full relative">
          <div className="absolute inset-0 z-10 pointer-events-none">
            <div className="hidden lg:block absolute right-0 top-0 bottom-0 w-32 bg-gradient-to-l from-black to-transparent" />
            <div className="lg:hidden absolute bottom-0 left-0 right-0 h-24 bg-gradient-to-t from-black to-transparent" />
            <div className="absolute top-0 left-0 right-0 h-20 bg-gradient-to-b from-black/40 to-transparent" />
          </div>
          <PosterSpiral posterUrls={posterUrls} />
        </div>

        <div className="w-full lg:w-[45%] h-[45vh] lg:h-full flex items-center justify-center px-6 sm:px-10 lg:px-16 relative">
          <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[400px] h-[400px] rounded-full bg-rose-500/[0.03] blur-[120px] pointer-events-none" />
          <div
            ref={textContainerRef}
            className="relative z-10 flex flex-col items-center lg:items-start text-center lg:text-left gap-6 max-w-[480px]"
          >
            <div className="anim-el inline-flex items-center gap-2 px-3 py-1.5 rounded-full border border-rose-500/20 bg-rose-500/5 backdrop-blur-md">
              <Database className="w-3.5 h-3.5 text-rose-400" />
              <span
                className="text-[10px] font-bold text-rose-400 uppercase tracking-widest"
                style={{ fontFamily: 'Plus Jakarta Sans, sans-serif' }}
              >
                {isSignedIn ? '620K+ Movies' : 'Dataset'}
              </span>
            </div>

            <h1
              className="text-4xl sm:text-5xl lg:text-6xl font-black uppercase tracking-tight leading-[1.05]"
              style={{ fontFamily: 'Plus Jakarta Sans, sans-serif' }}
            >
              <span className="anim-word inline-block text-white">Explore </span>
              <br className="hidden sm:block" />
              <span className="anim-word inline-block text-transparent bg-clip-text bg-gradient-to-r from-rose-500 via-purple-500 to-blue-500">
                620,000+{' '}
              </span>
              <span className="anim-word inline-block text-white">movies</span>
            </h1>

            <p
              className="anim-el text-sm text-white/45 max-w-[360px] leading-relaxed"
              style={{ fontFamily: 'Inter, sans-serif' }}
            >
              Powered by a knowledge base of over 620,000 movies enriched with genres, themes,
              plots, cast, ratings, and semantic metadata to deliver intelligent AI recommendations.
            </p>

            <div className="anim-el flex items-center gap-4 mt-2">
              {!isSignedIn ? (
                <Link
                  to="/profile"
                  className="group inline-flex items-center gap-2.5 px-7 py-3.5 bg-white text-black hover:bg-gradient-to-r hover:from-rose-500 hover:to-purple-600 hover:text-white text-xs font-black uppercase tracking-wider rounded-full transition-all duration-500 cursor-pointer shadow-lg hover:shadow-rose-500/20 hover:scale-105 active:scale-95"
                >
                  <LogIn className="w-4 h-4" />
                  Sign In to Continue
                </Link>
              ) : (
                <Link
                  to="/search"
                  className="group inline-flex items-center gap-2.5 px-7 py-3.5 border border-white/10 hover:border-rose-500/30 bg-white/[0.03] hover:bg-rose-500/5 text-white text-xs font-black uppercase tracking-wider rounded-full transition-all duration-500 cursor-pointer hover:scale-105 active:scale-95"
                >
                  <Sparkles className="w-4 h-4 text-rose-400" />
                  Explore Dataset
                </Link>
              )}
            </div>
            <div className="anim-el w-16 h-px bg-gradient-to-r from-rose-500/40 via-purple-500/40 to-transparent mt-2" />
          </div>
        </div>
      </div>
    </MainLayout>
  );
};

export default Dataset;

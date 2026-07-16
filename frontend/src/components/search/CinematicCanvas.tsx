import React, { useState, useEffect } from 'react';
import { Sparkles, Check, Info, Star, X, AlertTriangle } from 'lucide-react';
import CinematicBackgroundVideo from '../shared/CinematicBackgroundVideo';
import { FavouriteButton } from '../shared/FavouriteButton';

interface Movie {
  id: string;
  title: string;
  category: string;
  img: string;
  rating: string;
  match: string;
  desc: string;
  year?: string;
  duration?: string;
  trailerUrl?: string;
}

interface CinematicCanvasProps {
  canvasState: 'idle' | 'processing' | 'results';
  activeMovie: Movie | null;
  recommendations: Movie[];
  onSelectMovie: (movie: Movie) => void;
  onShowDetails?: (movie: Movie) => void;
  hasError?: boolean;
  errorMessage?: string | null;
  onRetry?: () => void;
}

export const CinematicCanvas: React.FC<CinematicCanvasProps> = ({
  canvasState,
  activeMovie,
  recommendations,
  onSelectMovie,
  onShowDetails,
  hasError = false,
  errorMessage = '',
  onRetry,
}) => {
  const [processingStep, setProcessingStep] = useState(0);
  const [trailerOpen, setTrailerOpen] = useState(false);

  const processingSteps = [
    { id: 0, label: '🎬 Understanding your request...' },
    { id: 1, label: '🧠 Extracting movie preferences...' },
    { id: 2, label: '🔍 Searching thousands of movies...' },
    { id: 3, label: '⚡ Ranking candidates...' },
    { id: 4, label: '✨ Preparing recommendations...' },
  ];

  // Animate processing steps sequentially
  useEffect(() => {
    if (canvasState !== 'processing') {
      const timer = setTimeout(() => {
        setProcessingStep(0);
      }, 0);
      return () => clearTimeout(timer);
    }

    const interval = setInterval(() => {
      setProcessingStep((prev) => {
        if (prev < processingSteps.length - 1) {
          return prev + 1;
        }
        return prev;
      });
    }, 750);

    return () => clearInterval(interval);
  }, [canvasState, processingSteps.length]);

  return (
    <div className="flex-1 h-full relative overflow-hidden flex flex-col justify-center items-center p-6 sm:p-8 lg:p-10 select-none">
      {/* 1. Cinematic Background Video Component */}
      <CinematicBackgroundVideo overlayOpacity={canvasState === 'idle' ? 0.3 : 0.45} />

      {/* 2. Content Overlay - Dependent on State */}
      <div className="w-full max-w-[720px] relative z-10 flex flex-col items-center justify-center h-full">
        {hasError ? (
          /* ERROR STATE: Centered premium warning and retry */
          <div className="flex flex-col items-center text-center gap-5 animate-fade-in max-w-[420px] p-8 rounded-3xl border border-rose-500/20 bg-black/60 backdrop-blur-xl shadow-2xl">
            <div className="w-12 h-12 rounded-full bg-rose-500/10 border border-rose-500/20 flex items-center justify-center text-rose-400">
              <AlertTriangle className="w-6 h-6" />
            </div>
            <h2
              className="text-xl font-black text-white uppercase tracking-tight"
              style={{ fontFamily: 'Plus Jakarta Sans, sans-serif' }}
            >
              Crawl Interrupted
            </h2>
            <p
              className="text-xs text-white/50 leading-relaxed font-medium"
              style={{ fontFamily: 'Inter, sans-serif' }}
            >
              {errorMessage || 'An error occurred while connecting to the semantic discovery network.'}
            </p>
            <button
              onClick={onRetry}
              className="inline-flex items-center gap-2 px-6 py-2.5 bg-white text-black text-xs font-bold uppercase tracking-wider rounded-full hover:bg-rose-500 hover:text-white transition-all cursor-pointer shadow-lg active:scale-95"
            >
              Retry Connection
            </button>
          </div>
        ) : (
          <>
            {canvasState === 'idle' && (
              /* IDLE STATE: Centered welcoming display */
              <div className="flex flex-col items-center text-center gap-6 animate-fade-in">
                <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full border border-rose-500/20 bg-rose-500/5 backdrop-blur-md text-[10px] font-bold text-rose-400 uppercase tracking-widest">
                  <Sparkles className="w-3.5 h-3.5" />
                  Cinematic Discovery Engine
                </div>
                <h2
                  className="text-4xl sm:text-5xl lg:text-[3.50rem] font-black text-white tracking-tight uppercase leading-[0.95]"
                  style={{ fontFamily: 'Plus Jakarta Sans, sans-serif' }}
                >
                  What story are you <br />
                  <span className="text-transparent bg-clip-text bg-gradient-to-r from-rose-500 via-purple-500 to-blue-500">
                    looking for today?
                  </span>
                </h2>
                <p
                  className="text-sm text-white/50 max-w-[420px] leading-relaxed"
                  style={{ fontFamily: 'Inter, sans-serif' }}
                >
                  Describe a theme, vibe, character trope, or emotional arc on the right panel to search
                  the vector space.
                </p>

                {/* Glowing animated prompt */}
                <div className="flex items-center gap-3 mt-6 border border-white/5 bg-white/[0.01] px-5 py-3 rounded-full backdrop-blur-lg">
                  <div className="w-2.5 h-2.5 rounded-full bg-rose-500 animate-ping" />
                  <span
                    className="text-[10px] font-bold uppercase tracking-widest text-white/40"
                    style={{ fontFamily: 'Plus Jakarta Sans, sans-serif' }}
                  >
                    Awaiting Query Input
                  </span>
                </div>
              </div>
            )}

            {canvasState === 'processing' && (
              /* PROCESSING STATE: Beautiful sequential step check-off */
              <div className="w-full max-w-[400px] p-8 rounded-3xl border border-white/[0.08] bg-black/60 backdrop-blur-xl shadow-[0_12px_40px_rgba(0,0,0,0.8)] flex flex-col gap-6">
                <div className="flex flex-col gap-1.5 border-b border-white/[0.06] pb-4">
                  <h3
                    className="text-sm font-bold text-white uppercase tracking-wider flex items-center gap-2"
                    style={{ fontFamily: 'Plus Jakarta Sans, sans-serif' }}
                  >
                    <div className="w-2.5 h-2.5 rounded-full bg-rose-500 animate-pulse" />
                    Semantic Pipeline
                  </h3>
                  <p className="text-[11px] text-white/40" style={{ fontFamily: 'Inter, sans-serif' }}>
                    Running query through vector neural network...
                  </p>
                </div>

                <div className="flex flex-col gap-4">
                  {processingSteps.map((step) => {
                    const isCompleted = processingStep > step.id;
                    const isActive = processingStep === step.id;
                    return (
                      <div
                        key={step.id}
                        className={`flex items-center justify-between transition-all duration-300 ${isActive
                            ? 'opacity-100 translate-x-1.5'
                            : isCompleted
                              ? 'opacity-80'
                              : 'opacity-30'
                          }`}
                      >
                        <span
                          className={`text-xs font-semibold tracking-wide ${isActive
                              ? 'text-rose-400 font-bold'
                              : isCompleted
                                ? 'text-white/70'
                                : 'text-white/40'
                            }`}
                          style={{ fontFamily: 'Inter, sans-serif' }}
                        >
                          {step.label}
                        </span>
                        <div className="flex items-center justify-center">
                          {isCompleted ? (
                            <div className="w-4 h-4 rounded-full bg-rose-500/10 border border-rose-500/30 flex items-center justify-center text-rose-400 animate-scale-in">
                              <Check className="w-2.5 h-2.5 stroke-[3]" />
                            </div>
                          ) : isActive ? (
                            <div className="w-4 h-4 rounded-full border border-rose-500/50 border-t-transparent animate-spin" />
                          ) : (
                            <div className="w-3.5 h-3.5 rounded-full border border-white/10" />
                          )}
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>
            )}

            {canvasState === 'results' && activeMovie && (
              /* RESULTS STATE: Glassmorphism movie layout */
              <div className="w-full flex flex-col gap-6 animate-scale-up h-full justify-between py-4 select-none">
                {/* Top Empty Space to Center Feature Card */}
                <div className="flex-1" />

                {/* Featured Movie Card */}
                <div className="w-full p-5 sm:p-6 rounded-3xl border border-white/[0.08] bg-black/50 backdrop-blur-xl shadow-[0_24px_64px_rgba(0,0,0,0.8)] flex flex-col sm:flex-row gap-6 items-center">
                  {/* Movie Poster */}
                  <div className="w-[120px] sm:w-[150px] aspect-[2/3] rounded-2xl overflow-hidden shadow-[0_12px_24px_rgba(0,0,0,0.5)] border border-white/10 shrink-0">
                    <img
                      src={activeMovie.img}
                      alt={activeMovie.title}
                      className="w-full h-full object-cover"
                    />
                  </div>

                  {/* Movie Information */}
                  <div className="flex-grow flex flex-col items-center sm:items-start gap-3.5 text-center sm:text-left">
                    <div className="flex flex-col gap-1.5 items-center sm:items-start">
                      <div className="flex flex-wrap items-center justify-center sm:justify-start gap-2.5">
                        <span className="text-[10px] font-extrabold uppercase tracking-widest text-rose-400 border border-rose-500/20 bg-rose-500/5 px-2 py-0.5 rounded">
                          {activeMovie.match}
                        </span>
                        <span className="text-[10px] font-bold text-white/50 uppercase tracking-widest">
                          {activeMovie.category}
                        </span>
                      </div>
                      <h3
                        className="text-xl sm:text-2xl font-black text-white uppercase tracking-tight"
                        style={{ fontFamily: 'Plus Jakarta Sans, sans-serif' }}
                      >
                        {activeMovie.title}
                      </h3>
                    </div>

                    {/* Rating & Tags */}
                    <div
                      className="flex items-center gap-4 text-xs font-semibold text-white/60"
                      style={{ fontFamily: 'Inter, sans-serif' }}
                    >
                      <div className="flex items-center gap-1 text-amber-400">
                        <Star className="w-3.5 h-3.5 fill-current" />
                        <span>{activeMovie.rating}</span>
                      </div>
                      <span>{activeMovie.year ?? '2021'}</span>
                      <span>{activeMovie.duration ?? '2h 10m'}</span>
                    </div>

                    {/* Explanation text */}
                    <p
                      className="text-[11px] leading-relaxed text-white/50 font-medium"
                      style={{ fontFamily: 'Inter, sans-serif' }}
                    >
                      <span className="text-white/80 font-bold block mb-0.5 uppercase tracking-widest text-[9px]">
                        AI Rationale
                      </span>
                      {activeMovie.desc}
                    </p>

                    {/* Actions */}
                    <div className="flex items-center gap-3 mt-1.5">
                      <button
                        onClick={() => onShowDetails && onShowDetails(activeMovie)}
                        className="inline-flex items-center gap-1.5 px-4 py-2 border border-white/10 hover:border-white/20 bg-white/[0.03] text-white text-[11px] font-bold uppercase tracking-wider rounded-full hover:bg-white/[0.06] transition-all cursor-pointer"
                      >
                        <Info className="w-3.5 h-3.5" />
                        Details
                      </button>
                      <FavouriteButton
                        movieId={activeMovie.title}
                        title={activeMovie.title}
                        posterPath={activeMovie.img}
                        ratingValue={parseFloat(activeMovie.rating) || null}
                        releaseYear={activeMovie.year ? parseInt(activeMovie.year) : null}
                        overview={activeMovie.desc}
                      />
                    </div>
                  </div>
                </div>

                {/* Bottom Section: Scrollable Matches list */}
                <div className="flex flex-col gap-2 w-full mt-4 shrink-0">
                  <span
                    className="text-[9px] font-extrabold uppercase tracking-widest text-white/30"
                    style={{ fontFamily: 'Plus Jakarta Sans, sans-serif' }}
                  >
                    Alternative Matches
                  </span>
                  <div className="flex items-center gap-3 overflow-x-auto pb-2 custom-scrollbar justify-start">
                    {recommendations.map((movie, idx) => {
                      const isActive = activeMovie.title === movie.title;
                      return (
                        <button
                          key={idx}
                          onClick={() => onSelectMovie(movie)}
                          className={`flex items-center gap-2.5 p-2 rounded-xl border transition-all cursor-pointer text-left shrink-0 max-w-[200px] ${isActive
                              ? 'border-rose-500/30 bg-rose-500/5 shadow-lg shadow-rose-500/5'
                              : 'border-white/[0.04] bg-white/[0.01] hover:bg-white/[0.04]'
                            }`}
                        >
                          <div className="w-[36px] aspect-[2/3] rounded-lg overflow-hidden shrink-0">
                            <img
                              src={movie.img}
                              alt={movie.title}
                              className="w-full h-full object-cover"
                            />
                          </div>
                          <div className="flex flex-col min-w-0 pr-1">
                            <span className="text-[11px] font-bold text-white truncate block uppercase tracking-tight">
                              {movie.title}
                            </span>
                            <span className="text-[9px] text-white/35 font-bold uppercase tracking-widest">
                              {movie.match}
                            </span>
                          </div>
                        </button>
                      );
                    })}
                  </div>
                </div>
              </div>
            )}
          </>
        )}
      </div>

      {/* 3. Immersive Video Trailer Modal */}
      {trailerOpen && activeMovie && (
        <div className="fixed inset-0 z-100 flex items-center justify-center p-4 bg-black/90 backdrop-blur-xl animate-fade-in">
          <div className="relative w-full max-w-[800px] aspect-video rounded-2xl overflow-hidden border border-white/10 bg-black shadow-2xl">
            <button
              onClick={() => setTrailerOpen(false)}
              className="absolute top-4 right-4 z-50 p-2 rounded-full bg-black/60 border border-white/10 hover:border-white/20 text-white/70 hover:text-white transition-colors cursor-pointer"
            >
              <X className="w-4 h-4" />
            </button>
            <iframe
              src={`${activeMovie.trailerUrl ?? 'https://www.youtube.com/embed/zSWdZVtXT7U'}?autoplay=1`}
              title={`${activeMovie.title} Trailer`}
              allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
              allowFullScreen
              className="w-full h-full border-none"
            />
          </div>
        </div>
      )}
    </div>
  );
};

export default CinematicCanvas;

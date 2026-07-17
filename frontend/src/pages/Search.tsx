import React, { useState, useEffect, useRef } from 'react';
import { createPortal } from 'react-dom';
import { MainLayout } from '@/layouts/MainLayout';
import { CinematicCanvas } from '@/components/search/CinematicCanvas';
import { ConversationPanel } from '@/components/search/ConversationPanel';
import { MobileSearch } from '@/components/search/MobileSearch';
import { MessageSquare, X, Star, AlertTriangle } from 'lucide-react';
import { toast } from 'sonner';
import { useHealthCheck, useRecommendations, useMovieDetails } from '@/hooks/useMovieApi';
import { useAuth } from '@/contexts/AuthContext';
import { GUEST_SEARCH_LIMIT, useGuestSearchLimit } from '@/hooks/useGuestSearchLimit';
import { GuestLimitModal } from '@/components/search/GuestLimitModal';

// Import movie posters for fallback images
import poster1 from '@/assets/posters/poster1.png';
import poster2 from '@/assets/posters/poster2.png';
import poster3 from '@/assets/posters/poster3.jpg';
import poster4 from '@/assets/posters/poster4.jpg';
import poster5 from '@/assets/posters/poster5.jpg';
import poster6 from '@/assets/posters/poster6.png';

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

interface Message {
  sender: 'user' | 'ai';
  text: string;
  isStreaming?: boolean;
}

export const Search: React.FC = () => {
  const [canvasState, setCanvasState] = useState<'idle' | 'processing' | 'results'>('idle');
  const [activeMovie, setActiveMovie] = useState<Movie | null>(null);
  const [recommendations, setRecommendations] = useState<Movie[]>([]);
  const [messages, setMessages] = useState<Message[]>([]);
  const [isProcessing, setIsProcessing] = useState(false);
  const [isChatCollapsed, setIsChatCollapsed] = useState(false);
  const [isMobile, setIsMobile] = useState(false);
  const [showGuestLimitModal, setShowGuestLimitModal] = useState(false);
  const [searchRequestId, setSearchRequestId] = useState(0);
  const searchRequestIdRef = useRef(0);
  const guestRequestCounts = useRef(new Map<number, number>());
  const { user, isLoading: isAuthLoading } = useAuth();
  const { guestSearchCount, isGuestSearchLimitReached, recordSuccessfulGuestSearch } =
    useGuestSearchLimit();
  const isGuestLimitActive = !isAuthLoading && !user && isGuestSearchLimitReached;

  // API Integration States
  const [searchQuery, setSearchQuery] = useState('');
  const [apiError, setApiError] = useState<string | null>(null);

  // Movie Details Modal State
  const [detailMovieId, setDetailMovieId] = useState<string | null>(null);

  // Mobile Detection
  useEffect(() => {
    const checkMobile = () => {
      setIsMobile(window.innerWidth < 1024);
    };
    checkMobile();
    window.addEventListener('resize', checkMobile);
    return () => window.removeEventListener('resize', checkMobile);
  }, []);

  // 1. Health check to monitor backend connectivity
  const { data: healthData, isError: isHealthError } = useHealthCheck({
    refetchInterval: 30000, // Recheck every 30s
  });

  const isBackendOffline = isHealthError || (healthData && healthData.status !== 'healthy');

  // Warn user once if backend is offline
  useEffect(() => {
    if (isBackendOffline) {
      toast.error('ChitraAI service is currently unavailable. Please try again shortly.', {
        id: 'backend-offline-toast',
      });
    }
  }, [isBackendOffline]);

  // 2. Recommendations Query Hook
  const {
    data: recData,
    error: recError,
    isFetching: isRecFetching,
  } = useRecommendations(
    searchQuery,
    3,
    {
      enabled: !!searchQuery && !isAuthLoading && (Boolean(user) || !isGuestLimitActive),
    },
    searchRequestId
  );

  // 3. Movie Details Query Hook
  const {
    data: detailData,
    isFetching: isDetailFetching,
    error: detailError,
  } = useMovieDetails(detailMovieId || '', { enabled: !!detailMovieId });

  // Handle Query Loading state
  useEffect(() => {
    if (isRecFetching) {
      setIsProcessing(true);
      setCanvasState('processing');
      setApiError(null);
    }
  }, [isRecFetching]);

  // Handle Query Error state
  useEffect(() => {
    if (recError) {
      guestRequestCounts.current.delete(searchRequestId);
      const errMsg =
        recError.message || 'An error occurred while communicating with the local FastAPI server.';
      setApiError(errMsg);
      setIsProcessing(false);
      setCanvasState('idle');
      toast.error(errMsg, { id: 'search-error-toast' });
    }
  }, [recError, searchRequestId]);

  // Handle Query Success state and AI response text streaming
  useEffect(() => {
    if (recData) {
      const guestSearchCountAtRequest = guestRequestCounts.current.get(searchRequestId);
      if (guestSearchCountAtRequest !== undefined) {
        guestRequestCounts.current.delete(searchRequestId);
        const hasReachedFinalGuestSearch = guestSearchCountAtRequest >= GUEST_SEARCH_LIMIT - 1;
        recordSuccessfulGuestSearch();
        if (hasReachedFinalGuestSearch) {
          setShowGuestLimitModal(true);
        }
      }

      if (!recData.results || recData.results.length === 0) {
        setRecommendations([]);
        setActiveMovie(null);
        setCanvasState('idle');
        setIsProcessing(false);
        setSearchQuery('');
        toast.info('No semantic matches found. Try describing your vibe in different terms.', {
          id: 'no-matches-toast',
        });
        return;
      }

      // Map backend movies to frontend structure
      const mappedMovies: Movie[] = recData.results.map((m, idx) => {
        // Pick a fallback poster if TMDB poster is missing
        const fallbackPosters = [poster1, poster2, poster3, poster4, poster5, poster6];
        const localFallback = fallbackPosters[idx % fallbackPosters.length];

        return {
          id: m.id,
          title: m.title,
          category: m.genres && m.genres.length > 0 ? m.genres.slice(0, 2).join(' • ') : 'Movie',
          img: m.poster_path
            ? m.poster_path.startsWith('http')
              ? m.poster_path
              : `https://image.tmdb.org/t/p/w500${m.poster_path}`
            : localFallback,
          rating: m.rating_value ? String(m.rating_value) : '7.8',
          match: m.reranked_score ? `${Math.round(m.reranked_score * 100)}% Match` : '92% Match',
          desc:
            m.recommendation_reason ||
            m.overview ||
            'Highly matching title from our semantic index.',
          year: m.release_year ? String(m.release_year) : '2021',
          duration: m.runtime_minutes ? `${m.runtime_minutes}m` : '2h',
          trailerUrl: m.trailer_url ? m.trailer_url.replace('watch?v=', 'embed/') : undefined,
        };
      });

      setRecommendations(mappedMovies);
      setActiveMovie(mappedMovies[0]);
      setCanvasState('results');

      // Build personalized AI response dynamically from parsed query understanding
      const themes = recData.understanding?.themes || [];
      const genres = recData.understanding?.genres || [];
      let responseIntro = 'Processed semantic match for your request. ';

      if (themes.length > 0 || genres.length > 0) {
        const themePart = themes.length > 0 ? `themes like "${themes.slice(0, 3).join(', ')}"` : '';
        const genrePart = genres.length > 0 ? `genres like "${genres.slice(0, 2).join(', ')}"` : '';
        const parts = [themePart, genrePart].filter(Boolean).join(' and ');
        responseIntro = `Initiated semantic crawl for ${parts}. I found these premium options in our vector database:`;
      } else {
        responseIntro = `I found these movie recommendations matching the emotional vibe of your request:`;
      }

      // Stream response character by character
      const streamingMsg: Message = { sender: 'ai', text: '', isStreaming: true };
      setMessages((prev) => [...prev, streamingMsg]);

      let charIndex = 0;
      const interval = setInterval(() => {
        charIndex += 3;
        if (charIndex >= responseIntro.length) {
          clearInterval(interval);
          setMessages((prev) => {
            const updated = [...prev];
            const last = updated[updated.length - 1];
            if (last && last.sender === 'ai') {
              last.text = responseIntro;
              last.isStreaming = false;
            }
            return updated;
          });
          setIsProcessing(false);
          setSearchQuery(''); // Reset query state
        } else {
          setMessages((prev) => {
            const updated = [...prev];
            const last = updated[updated.length - 1];
            if (last && last.sender === 'ai') {
              last.text = responseIntro.slice(0, charIndex);
            }
            return updated;
          });
        }
      }, 20);
    }
  }, [recData, recordSuccessfulGuestSearch, searchRequestId]);

  const handleSendMessage = (text: string) => {
    if (isProcessing) return;

    if (isAuthLoading) return;

    if (isGuestLimitActive) {
      setShowGuestLimitModal(true);
      return;
    }

    if (isBackendOffline) {
      toast.error('Cannot submit query. The local FastAPI server is offline.', {
        id: 'backend-offline-submit',
      });
      return;
    }

    setApiError(null);
    setMessages((prev) => [...prev, { sender: 'user', text }]);
    const nextRequestId = searchRequestIdRef.current + 1;
    searchRequestIdRef.current = nextRequestId;
    if (!user) {
      guestRequestCounts.current.set(nextRequestId, guestSearchCount);
    }
    setSearchRequestId(nextRequestId);
    setSearchQuery(text);
  };

  const handleClearChat = () => {
    setMessages([]);
    setCanvasState('idle');
    setActiveMovie(null);
    setRecommendations([]);
    setSearchQuery('');
    setApiError(null);
  };

  const handleRetry = () => {
    if (isAuthLoading) return;

    if (isGuestLimitActive) {
      setShowGuestLimitModal(true);
      return;
    }

    const userMessages = messages.filter((m) => m.sender === 'user');
    if (userMessages.length > 0) {
      const lastText = userMessages[userMessages.length - 1].text;
      setApiError(null);
      const nextRequestId = searchRequestIdRef.current + 1;
      searchRequestIdRef.current = nextRequestId;
      if (!user) {
        guestRequestCounts.current.set(nextRequestId, guestSearchCount);
      }
      setSearchRequestId(nextRequestId);
      setSearchQuery(lastText);
    }
  };

  return (
    <MainLayout>
      {isMobile ? (
        createPortal(
          <MobileSearch
            messages={messages}
            onSendMessage={handleSendMessage}
            isProcessing={isProcessing}
            onClearChat={handleClearChat}
            canvasState={canvasState}
            activeMovie={activeMovie}
            recommendations={recommendations}
            onSelectMovie={setActiveMovie}
            onShowDetails={(movie) => setDetailMovieId(movie.id)}
            hasError={!!apiError}
            errorMessage={apiError}
            onRetry={handleRetry}
            guestSearchCount={user ? undefined : guestSearchCount}
            isGuestSearchLimitReached={isGuestLimitActive}
          />,
          document.body
        )
      ) : (
        <div className="relative w-full h-[calc(100vh-112px)] flex flex-col lg:flex-row overflow-hidden bg-black select-none">
          {/* Left Side: Cinematic Canvas (70% width or 100% width when collapsed) */}
          <div
            className={`h-full relative overflow-hidden flex flex-col transition-all duration-500 ease-in-out ${
              isChatCollapsed ? 'w-full lg:w-full' : 'w-full lg:w-[70%]'
            }`}
          >
            <CinematicCanvas
              canvasState={canvasState}
              activeMovie={activeMovie}
              recommendations={recommendations}
              onSelectMovie={setActiveMovie}
              onShowDetails={(movie) => setDetailMovieId(movie.id)}
              hasError={!!apiError}
              errorMessage={apiError}
              onRetry={handleRetry}
            />

            {/* Floating Workspace Restore Button (visible when collapsed) */}
            {isChatCollapsed && (
              <button
                onClick={() => setIsChatCollapsed(false)}
                className="absolute top-6 right-6 z-40 p-3 rounded-full border border-rose-500/20 bg-rose-500/10 hover:bg-rose-500/20 text-rose-400 hover:text-white transition-all duration-300 cursor-pointer shadow-lg shadow-rose-500/10 flex items-center gap-2 hover:scale-105"
              >
                <MessageSquare className="w-4 h-4" />
                <span
                  className="text-[10px] font-bold uppercase tracking-widest"
                  style={{ fontFamily: 'Plus Jakarta Sans, sans-serif' }}
                >
                  Open Workspace
                </span>
              </button>
            )}
          </div>

          {/* Right Side: AI Conversation Panel (30% width or 0% width when collapsed) */}
          <div
            className={`h-[350px] lg:h-full shrink-0 transition-all duration-500 ease-in-out overflow-hidden ${
              isChatCollapsed
                ? 'w-0 lg:w-0 opacity-0 pointer-events-none'
                : 'w-full lg:w-[30%] opacity-100'
            }`}
          >
            <ConversationPanel
              messages={messages}
              onSendMessage={handleSendMessage}
              isProcessing={isProcessing}
              onClearChat={handleClearChat}
              onToggleCollapse={() => setIsChatCollapsed(true)}
              guestSearchCount={user ? undefined : guestSearchCount}
              isGuestSearchLimitReached={isGuestLimitActive}
            />
          </div>
        </div>
      )}

      {/* Movie Details Modal */}
      {detailMovieId && (
        <div className="fixed inset-0 z-[100] flex items-center justify-center p-4 bg-black/80 backdrop-blur-md animate-fade-in">
          <div className="relative w-full max-w-[650px] rounded-3xl border border-white/[0.08] bg-zinc-950/95 backdrop-blur-xl shadow-2xl overflow-hidden flex flex-col max-h-[85vh]">
            {/* Backdrop Area */}
            <div className="relative h-[200px] sm:h-[240px] w-full shrink-0 overflow-hidden">
              {isDetailFetching ? (
                <div className="w-full h-full bg-zinc-900 animate-pulse flex items-center justify-center text-white/20">
                  Loading Backdrop...
                </div>
              ) : detailData?.backdrop_path ? (
                <>
                  <img
                    src={`https://image.tmdb.org/t/p/w780${detailData.backdrop_path}`}
                    alt="Backdrop"
                    className="w-full h-full object-cover opacity-45 scale-105 blur-[2px]"
                  />
                  <div className="absolute inset-0 bg-gradient-to-t from-zinc-950 via-zinc-950/60 to-transparent" />
                </>
              ) : (
                <div className="w-full h-full bg-gradient-to-t from-zinc-950 to-zinc-900/50" />
              )}

              {/* Close Button */}
              <button
                onClick={() => setDetailMovieId(null)}
                className="absolute top-4 right-4 z-50 p-2 rounded-full bg-black/60 border border-white/10 hover:border-white/20 text-white/70 hover:text-white transition-colors cursor-pointer"
              >
                <X className="w-4 h-4" />
              </button>
            </div>

            {/* Content Area */}
            <div className="p-6 sm:p-8 overflow-y-auto flex-grow custom-scrollbar flex flex-col gap-6 -mt-20 relative z-10">
              {isDetailFetching ? (
                <div className="flex flex-col gap-4 py-12 items-center justify-center">
                  <div className="w-8 h-8 rounded-full border border-rose-500/50 border-t-transparent animate-spin" />
                  <span className="text-xs text-white/50 font-medium">
                    Fetching details from TMDb...
                  </span>
                </div>
              ) : detailError ? (
                <div className="flex flex-col gap-3 py-12 items-center justify-center text-center">
                  <AlertTriangle className="w-8 h-8 text-rose-500" />
                  <span className="text-xs text-white/60 font-medium">
                    Failed to load movie details.
                  </span>
                  <button
                    onClick={() => setDetailMovieId(null)}
                    className="mt-2 px-4 py-1.5 bg-white/10 text-white rounded-full text-xs font-bold"
                  >
                    Close Modal
                  </button>
                </div>
              ) : detailData ? (
                <div className="flex flex-col gap-6 text-white">
                  {/* Title and Poster Card */}
                  <div className="flex flex-col sm:flex-row gap-5 items-start">
                    <div className="w-[100px] aspect-[2/3] rounded-xl overflow-hidden shadow-lg border border-white/10 shrink-0 bg-zinc-900">
                      <img
                        src={
                          detailData.poster_path
                            ? `https://image.tmdb.org/t/p/w300${detailData.poster_path}`
                            : poster1
                        }
                        alt={detailData.title}
                        className="w-full h-full object-cover"
                      />
                    </div>
                    <div className="flex-grow flex flex-col gap-2 pt-2">
                      <h3 className="text-xl sm:text-2xl font-black uppercase tracking-tight leading-none">
                        {detailData.title}
                      </h3>
                      {detailData.original_title &&
                        detailData.original_title !== detailData.title && (
                          <p className="text-xs text-white/40 font-semibold uppercase tracking-wider -mt-1">
                            Original: {detailData.original_title}
                          </p>
                        )}

                      {/* Genres, Duration, Year */}
                      <div className="flex flex-wrap items-center gap-3 text-xs text-white/50 font-medium pt-1">
                        {detailData.genres && detailData.genres.length > 0 && (
                          <span className="text-rose-400 font-bold uppercase tracking-wider">
                            {detailData.genres.join(' • ')}
                          </span>
                        )}
                        <span>{detailData.release_year}</span>
                        {detailData.runtime_minutes && <span>{detailData.runtime_minutes}m</span>}
                      </div>

                      {/* Ratings and Popularity */}
                      <div className="flex items-center gap-4 text-xs font-bold text-white/70 mt-1">
                        {detailData.rating_value && (
                          <div className="flex items-center gap-1 text-amber-400">
                            <Star className="w-3.5 h-3.5 fill-current" />
                            <span>{detailData.rating_value.toFixed(1)} / 10</span>
                          </div>
                        )}
                        {detailData.popularity && (
                          <span className="text-white/40 font-semibold uppercase tracking-widest text-[9px]">
                            Popularity: {Math.round(detailData.popularity)}
                          </span>
                        )}
                      </div>
                    </div>
                  </div>

                  {/* Overview */}
                  <div className="flex flex-col gap-1.5">
                    <span className="text-[10px] font-extrabold uppercase tracking-widest text-rose-500/80">
                      Overview
                    </span>
                    <p className="text-xs text-white/60 leading-relaxed font-medium">
                      {detailData.overview || 'No synopsis available.'}
                    </p>
                  </div>

                  {/* Cast and Directors */}
                  <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                    {detailData.directors && detailData.directors.length > 0 && (
                      <div className="flex flex-col gap-1">
                        <span className="text-[10px] font-extrabold uppercase tracking-widest text-rose-500/80">
                          Director
                        </span>
                        <span className="text-xs text-white/80 font-bold">
                          {detailData.directors.join(', ')}
                        </span>
                      </div>
                    )}
                    {detailData.cast && detailData.cast.length > 0 && (
                      <div className="flex flex-col gap-1">
                        <span className="text-[10px] font-extrabold uppercase tracking-widest text-rose-500/80">
                          Top Cast
                        </span>
                        <span className="text-xs text-white/80 font-semibold leading-relaxed">
                          {detailData.cast.slice(0, 5).join(', ')}
                        </span>
                      </div>
                    )}
                  </div>

                  {/* Streaming Providers */}
                  {detailData.streaming_providers && detailData.streaming_providers.length > 0 && (
                    <div className="flex flex-col gap-2">
                      <span className="text-[10px] font-extrabold uppercase tracking-widest text-rose-500/80">
                        Where to Stream (US)
                      </span>
                      <div className="flex flex-wrap gap-2">
                        {detailData.streaming_providers.map((provider: string, idx: number) => (
                          <span
                            key={idx}
                            className="text-[9px] font-bold text-white/80 border border-white/10 bg-white/[0.03] px-2.5 py-1 rounded-full uppercase tracking-wider"
                          >
                            {provider}
                          </span>
                        ))}
                      </div>
                    </div>
                  )}

                  {/* Certification */}
                  {detailData.certification && (
                    <div className="flex flex-col gap-1">
                      <span className="text-[10px] font-extrabold uppercase tracking-widest text-rose-500/80">
                        Content Rating
                      </span>
                      <span className="text-xs text-white/80 font-bold uppercase">
                        {detailData.certification}
                      </span>
                    </div>
                  )}
                </div>
              ) : null}
            </div>
          </div>
        </div>
      )}

      {showGuestLimitModal && !user && (
        <GuestLimitModal onClose={() => setShowGuestLimitModal(false)} />
      )}
    </MainLayout>
  );
};

export default Search;

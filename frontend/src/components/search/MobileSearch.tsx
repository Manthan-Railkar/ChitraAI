import React, { useState, useRef, useEffect } from 'react';
import { Send, Info, Star, X, Check, RotateCcw, AlertTriangle } from 'lucide-react';
import { gsap } from 'gsap';
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

interface Message {
  sender: 'user' | 'ai';
  text: string;
  isStreaming?: boolean;
}

interface MobileSearchProps {
  messages: Message[];
  onSendMessage: (text: string) => void;
  isProcessing: boolean;
  onClearChat: () => void;
  canvasState: 'idle' | 'processing' | 'results';
  activeMovie: Movie | null;
  recommendations: Movie[];
  onSelectMovie: (movie: Movie) => void;
  onShowDetails?: (movie: Movie) => void;
  hasError?: boolean;
  errorMessage?: string | null;
  onRetry?: () => void;
}

export const MobileSearch: React.FC<MobileSearchProps> = ({
  messages,
  onSendMessage,
  isProcessing,
  onClearChat,
  canvasState,
  activeMovie,
  recommendations,
  onSelectMovie,
  onShowDetails,
  hasError = false,
  errorMessage = '',
  onRetry,
}) => {
  const [input, setInput] = useState('');
  const [showChat, setShowChat] = useState(messages.length > 0);
  const [processingStep, setProcessingStep] = useState(0);
  const [trailerMovie, setTrailerMovie] = useState<Movie | null>(null);

  const heroRef = useRef<HTMLDivElement>(null);
  const chatFeedRef = useRef<HTMLDivElement>(null);
  const scrollContainerRef = useRef<HTMLDivElement>(null);
  const chatEndRef = useRef<HTMLDivElement>(null);

  const processingSteps = [
    { id: 0, label: '🎬 Understanding your request...' },
    { id: 1, label: '🧠 Extracting movie preferences...' },
    { id: 2, label: '🔍 Searching thousands of movies...' },
    { id: 3, label: '⚡ Ranking candidates...' },
    { id: 4, label: '✨ Preparing recommendations...' },
  ];

  // GSAP: Floating Animation for Hero Content
  useEffect(() => {
    let ctx = gsap.context(() => {
      if (heroRef.current && !showChat) {
        gsap.to(heroRef.current, {
          y: -10,
          repeat: -1,
          yoyo: true,
          ease: 'power1.inOut',
          duration: 3,
        });
      }
    });
    return () => ctx.revert();
  }, [showChat]);

  // GSAP: Transition from Hero to Chat Experience on first message
  useEffect(() => {
    if (messages.length > 0 && !showChat) {
      setShowChat(true);

      // Animate Hero out
      if (heroRef.current) {
        gsap.to(heroRef.current, {
          opacity: 0,
          y: -30,
          duration: 0.5,
          ease: 'power2.inOut',
          onComplete: () => {
            if (heroRef.current) heroRef.current.style.display = 'none';
          },
        });
      }

      // Animate Chat Feed in
      if (chatFeedRef.current) {
        gsap.fromTo(
          chatFeedRef.current,
          { opacity: 0, y: 30 },
          { opacity: 1, y: 0, duration: 0.7, ease: 'power3.out', delay: 0.1 }
        );
      }
    }
  }, [messages.length, showChat]);

  // Reset state if messages are cleared
  useEffect(() => {
    if (messages.length === 0 && showChat) {
      setShowChat(false);
      if (heroRef.current) {
        heroRef.current.style.display = 'flex';
        gsap.to(heroRef.current, {
          opacity: 1,
          y: 0,
          duration: 0.5,
          ease: 'power2.out',
        });
      }
    }
  }, [messages.length, showChat]);

  // Auto-scroll to bottom of chat
  useEffect(() => {
    if (showChat) {
      setTimeout(() => {
        chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
      }, 50);
    }
  }, [messages, isProcessing, canvasState, showChat]);

  // Animate processing steps
  useEffect(() => {
    if (canvasState !== 'processing') {
      setProcessingStep(0);
      return;
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
  }, [canvasState]);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || isProcessing) return;
    onSendMessage(input);
    setInput('');
  };

  return (
    <div className="fixed inset-0 w-full h-full overflow-hidden bg-black select-none z-10">
      {/* 1. Cinematic Background Video with customized opacities for mobile layout */}
      <CinematicBackgroundVideo
        videoOpacity={0.9}
        overlayOpacity={canvasState === 'idle' ? 0.3 : 0.45}
        className="fixed inset-0 w-full h-full z-0"
      />

      {/* 2. Clear Chat Session button */}
      {showChat && (
        <button
          onClick={onClearChat}
          className="fixed top-20 right-4 z-45 p-2.5 rounded-full border border-white/10 bg-black/60 backdrop-blur-md text-white/60 hover:text-white transition-all cursor-pointer shadow-lg active:scale-95"
          title="Reset Discovery Session"
        >
          <RotateCcw className="w-3.5 h-3.5" />
        </button>
      )}

      {/* Welcome State (Welcome Hero) */}
      <div
        ref={heroRef}
        style={{ display: showChat ? 'none' : 'flex' }}
        className="absolute inset-0 z-20 flex flex-col items-center justify-center text-center px-6"
      >
        {/* ChitraAI Logo */}
        <div className="flex items-center gap-2.5 mb-6 opacity-90">
          <div className="w-9 h-9 rounded-xl bg-gradient-to-tr from-rose-500 via-purple-600 to-blue-500 flex items-center justify-center text-white font-black text-base shadow-lg shadow-purple-500/20">
            C
          </div>
          <span
            className="text-lg font-extrabold tracking-wider uppercase text-white"
            style={{ fontFamily: 'Plus Jakarta Sans, sans-serif' }}
          >
            Chitra<span className="text-rose-500">AI</span>
          </span>
        </div>

        {/* Premium Heading */}
        <h1
          className="text-3xl sm:text-4xl leading-[1.1] font-black uppercase text-white tracking-tight mb-4 max-w-[340px]"
          style={{ fontFamily: 'Plus Jakarta Sans, sans-serif' }}
        >
          Discover movies <br />
          through <span className="text-transparent bg-clip-text bg-gradient-to-r from-rose-400 via-purple-400 to-blue-400">meaning</span>, <br />
          not keywords.
        </h1>

        {/* Concise Subtitle */}
        <p
          className="text-[12px] sm:text-xs text-white/45 max-w-[280px] leading-relaxed font-medium"
          style={{ fontFamily: 'Inter, sans-serif' }}
        >
          Describe a theme, emotional state, or storyline in natural language to search the cinematic vector space.
        </p>
      </div>

      {/* Active Conversation Feed */}
      <div
        ref={chatFeedRef}
        style={{ opacity: showChat ? 1 : 0 }}
        className="absolute inset-x-0 top-0 bottom-0 z-20 flex flex-col pointer-events-auto"
      >
        <div
          ref={scrollContainerRef}
          className="flex-1 overflow-y-auto px-4 pt-24 pb-32 no-scrollbar"
        >
          <div className="max-w-[600px] mx-auto flex flex-col gap-5 justify-end min-h-full">
            {messages.map((msg, index) => {
              const isUser = msg.sender === 'user';
              return (
                <div
                  key={index}
                  className={`flex flex-col w-full ${isUser ? 'items-end' : 'items-start'} animate-fade-in`}
                >
                  <div
                    className={`max-w-[85%] px-4 py-3 rounded-2xl text-xs leading-relaxed shadow-lg ${
                      isUser
                        ? 'bg-rose-500/10 border border-rose-500/20 text-white rounded-tr-none'
                        : 'bg-white/[0.03] border border-white/[0.06] text-white/90 rounded-tl-none backdrop-blur-md'
                    }`}
                    style={{ fontFamily: 'Inter, sans-serif' }}
                  >
                    {msg.text}
                    {msg.isStreaming && (
                      <span className="inline-block w-1.5 h-3 bg-rose-500 animate-pulse ml-1 align-middle" />
                    )}
                  </div>

                  {/* Horizontal Movie Recommendations Carousel */}
                  {!isUser && !msg.isStreaming && index === messages.length - 1 && recommendations.length > 0 && (
                    <div className="w-full mt-4 select-none">
                      <div className="flex items-center gap-4 overflow-x-auto py-2 px-0.5 no-scrollbar scroll-smooth snap-x snap-mandatory">
                        {recommendations.map((movie, idx) => (
                          <div
                            key={idx}
                            onClick={() => onSelectMovie(movie)}
                            className={`w-[260px] shrink-0 snap-center rounded-2xl border p-4 flex flex-col gap-3 shadow-2xl transition-all duration-300 cursor-pointer ${
                              activeMovie?.title === movie.title
                                ? 'border-rose-500/40 bg-rose-500/5 shadow-[0_0_20px_rgba(244,63,94,0.05)]'
                                : 'border-white/[0.08] bg-black/60 hover:bg-black/80'
                            }`}
                          >
                            {/* Poster & Badges */}
                            <div className="relative aspect-[16/10] rounded-xl overflow-hidden border border-white/5 shrink-0">
                              <img
                                src={movie.img}
                                alt={movie.title}
                                className="w-full h-full object-cover"
                              />
                              <div className="absolute inset-0 bg-gradient-to-t from-black/80 via-black/20 to-transparent" />
                              <span className="absolute top-2 left-2 text-[9px] font-black uppercase tracking-widest text-rose-400 border border-rose-500/20 bg-rose-500/10 px-2 py-0.5 rounded backdrop-blur-md">
                                {movie.match}
                              </span>
                              <div className="absolute bottom-2 left-2 right-2 flex items-center justify-between">
                                <span className="text-[10px] font-bold text-white/60 truncate max-w-[140px]">
                                  {movie.category}
                                </span>
                                <div className="flex items-center gap-1 text-amber-400 text-[10px] font-bold">
                                  <Star className="w-3 h-3 fill-current" />
                                  <span>{movie.rating}</span>
                                </div>
                              </div>
                            </div>

                            {/* Details */}
                            <div className="flex flex-col gap-1">
                              <h4
                                className="text-sm font-black text-white uppercase tracking-wide truncate"
                                style={{ fontFamily: 'Plus Jakarta Sans, sans-serif' }}
                              >
                                {movie.title}
                              </h4>
                              <div className="flex gap-2 text-[10px] text-white/40 font-semibold uppercase tracking-wider">
                                <span>{movie.year ?? '2021'}</span>
                                <span>•</span>
                                <span>{movie.duration ?? '2h 10m'}</span>
                              </div>
                              <p
                                className="text-[11px] text-white/50 leading-relaxed font-medium line-clamp-3 mt-1"
                                style={{ fontFamily: 'Inter, sans-serif' }}
                              >
                                {movie.desc}
                              </p>
                            </div>

                            {/* Actions */}
                            <div className="flex items-center gap-2 mt-auto pt-1">
                              <button
                                onClick={() => onShowDetails && onShowDetails(movie)}
                                className="flex-grow inline-flex items-center justify-center gap-1.5 py-2 border border-white/10 bg-white/[0.03] text-white text-[10px] font-extrabold uppercase tracking-widest rounded-full hover:bg-white/[0.06] transition-all cursor-pointer active:scale-95"
                              >
                                <Info className="w-3.5 h-3.5" />
                                Details
                              </button>
                              <FavouriteButton
                                movieId={movie.title}
                                title={movie.title}
                                posterPath={movie.img}
                                ratingValue={parseFloat(movie.rating) || null}
                                releaseYear={movie.year ? parseInt(movie.year) : null}
                                overview={movie.desc}
                                size="sm"
                              />
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              );
            })}

            {/* Semantic Pipeline Progress Steps in Chat */}
            {canvasState === 'processing' && (
              <div className="self-start max-w-[85%] rounded-2xl border border-white/[0.08] bg-black/60 backdrop-blur-md p-4 flex flex-col gap-4 shadow-lg w-full animate-pulse">
                <div className="flex items-center gap-2 border-b border-white/5 pb-2">
                  <div className="w-2 h-2 rounded-full bg-rose-500 animate-ping" />
                  <span
                    className="text-[10px] font-bold text-white/70 uppercase tracking-widest"
                    style={{ fontFamily: 'Plus Jakarta Sans, sans-serif' }}
                  >
                    Semantic Pipeline Active
                  </span>
                </div>
                <div className="flex flex-col gap-2.5">
                  {processingSteps.map((step) => {
                    const isCompleted = processingStep > step.id;
                    const isActive = processingStep === step.id;
                    return (
                      <div
                        key={step.id}
                        className={`flex items-center justify-between text-[11px] transition-all duration-300 ${
                          isActive ? 'opacity-100 pl-1' : isCompleted ? 'opacity-70' : 'opacity-25'
                        }`}
                      >
                        <span
                          className={`font-semibold tracking-wide ${
                            isActive ? 'text-rose-400 font-bold' : isCompleted ? 'text-white/60' : 'text-white/40'
                          }`}
                          style={{ fontFamily: 'Inter, sans-serif' }}
                        >
                          {step.label}
                        </span>
                        <div>
                          {isCompleted ? (
                            <Check className="w-3 h-3 text-rose-400 stroke-[3]" />
                          ) : isActive ? (
                            <div className="w-3 h-3 rounded-full border border-rose-500/50 border-t-transparent animate-spin" />
                          ) : (
                            <div className="w-2.5 h-2.5 rounded-full border border-white/10" />
                          )}
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>
            )}

            {/* Error Message Bubble */}
            {hasError && (
              <div className="self-center p-5 rounded-2xl border border-rose-500/20 bg-rose-500/5 backdrop-blur-md flex flex-col gap-3 items-center text-center max-w-[85%] animate-fade-in my-2 w-full">
                <div className="w-10 h-10 rounded-full bg-rose-500/10 border border-rose-500/20 flex items-center justify-center text-rose-400">
                  <AlertTriangle className="w-5 h-5" />
                </div>
                <div className="flex flex-col gap-1">
                  <h4 className="text-xs font-bold text-white uppercase tracking-wider">Search Interrupted</h4>
                  <p className="text-[10px] text-white/50 leading-relaxed font-medium">
                    {errorMessage || 'Failed to communicate with local FastAPI server.'}
                  </p>
                </div>
                <button
                  onClick={onRetry}
                  className="inline-flex items-center gap-1.5 px-4 py-1.5 bg-white text-black hover:bg-rose-500 hover:text-white text-[10px] font-bold uppercase tracking-wider rounded-full transition-all cursor-pointer shadow-md active:scale-95"
                >
                  Retry Search
                </button>
              </div>
            )}
            <div ref={chatEndRef} />
          </div>
        </div>
      </div>

      {/* 3. Floating Input Bar - docked at the bottom of the screen */}
      <div
        className="fixed bottom-0 inset-x-0 px-4 bg-gradient-to-t from-black via-black/60 to-transparent pt-12 z-30 pointer-events-none"
        style={{ paddingBottom: 'calc(1.25rem + env(safe-area-inset-bottom, 0px))' }}
      >
        <form
          onSubmit={handleSubmit}
          className="max-w-[600px] mx-auto w-full pointer-events-auto relative group"
        >
          {/* Blue-purple ambient glow backdrop */}
          <div className="absolute -inset-0.5 bg-gradient-to-r from-blue-600 via-purple-600 to-pink-500 rounded-full blur opacity-15 group-focus-within:opacity-35 transition duration-500" />

          {/* Pill bar structure with glassmorphism */}
          <div className="relative flex items-center bg-black/50 border border-white/10 rounded-full p-2.5 backdrop-blur-2xl shadow-2xl transition-all duration-300 group-focus-within:border-white/20">
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              disabled={isProcessing}
              placeholder={isProcessing ? 'AI is crawling vectors...' : 'Describe your perfect film vibe...'}
              className="flex-grow bg-transparent border-none text-xs text-white placeholder-white/30 outline-none px-4 py-2.5 disabled:opacity-50"
              style={{ fontFamily: 'Inter, sans-serif' }}
            />
            <button
              type="submit"
              disabled={!input.trim() || isProcessing}
              className="inline-flex items-center justify-center w-9 h-9 rounded-full bg-white text-black hover:bg-rose-500 hover:text-white transition-all disabled:opacity-20 disabled:hover:bg-white disabled:hover:text-black cursor-pointer shadow-md select-none shrink-0"
            >
              <Send className="w-3.5 h-3.5" />
            </button>
          </div>
        </form>
      </div>

      {/* 4. Fullscreen Video Trailer Modal */}
      {trailerMovie && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/95 backdrop-blur-xl animate-fade-in">
          <div className="relative w-full max-w-[800px] aspect-video rounded-2xl overflow-hidden border border-white/10 bg-black shadow-2xl">
            <button
              onClick={() => setTrailerMovie(null)}
              className="absolute top-4 right-4 z-50 p-2 rounded-full bg-black/60 border border-white/10 hover:border-white/20 text-white/70 hover:text-white transition-colors cursor-pointer"
            >
              <X className="w-4 h-4" />
            </button>
            <iframe
              src={`${trailerMovie.trailerUrl ?? 'https://www.youtube.com/embed/zSWdZVtXT7U'}?autoplay=1`}
              title={`${trailerMovie.title} Trailer`}
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

export default MobileSearch;

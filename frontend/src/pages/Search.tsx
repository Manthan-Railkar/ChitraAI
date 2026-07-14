import React, { useState } from 'react';
import { MainLayout } from '@/layouts/MainLayout';
import { CinematicCanvas } from '@/components/search/CinematicCanvas';
import { ConversationPanel } from '@/components/search/ConversationPanel';
import { MessageSquare } from 'lucide-react';

// Import movie posters
import poster1 from '@/assets/posters/poster1.png';
import poster2 from '@/assets/posters/poster2.png';
import poster3 from '@/assets/posters/poster3.jpg';
import poster4 from '@/assets/posters/poster4.jpg';
import poster5 from '@/assets/posters/poster5.jpg';
import poster6 from '@/assets/posters/poster6.png';

interface Movie {
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

  // Mock Movie Data Sets mapped to query keywords
  const sciFiMovies: Movie[] = [
    {
      title: 'Interstellar',
      category: 'Sci-fi • Movie',
      img: poster5,
      rating: '8.7',
      match: '98% Match',
      desc: 'Matches your query for time dilation, gravity anomaly, and realistic space travel physics. A journey beyond our galaxy to save humanity.',
      year: '2014',
      duration: '2h 49m',
      trailerUrl: 'https://www.youtube.com/embed/zSWdZVtXT7U',
    },
    {
      title: 'Stranger Things',
      category: 'Adventure • Series',
      img: poster6,
      rating: '8.7',
      match: '92% Match',
      desc: 'A nostalgic retro sci-fi mystery set in the 1980s dealing with government laboratories, portals, and parallel dimension exploration.',
      year: '2016',
      duration: '4 Seasons',
      trailerUrl: 'https://www.youtube.com/embed/b9EkMc79ZSU',
    },
    {
      title: 'Spider Man - Brand New Day',
      category: 'Action • Movie',
      img: poster3,
      rating: '7.8',
      match: '85% Match',
      desc: 'Multiverse anomalies, high-tech suits, and fast-paced superhero physics collide in a spectacular visual adventure.',
      year: '2021',
      duration: '2h 28m',
      trailerUrl: 'https://www.youtube.com/embed/t06RUxPbp_c',
    },
  ];

  const nostalgicMovies: Movie[] = [
    {
      title: 'Stranger Things',
      category: 'Adventure • Series',
      img: poster6,
      rating: '8.7',
      match: '95% Match',
      desc: 'Evokes 80s nostalgia, synth soundtracks, retro arcade aesthetics, and close-knit childhood friendships tackling small-town mysteries.',
      year: '2016',
      duration: '4 Seasons',
      trailerUrl: 'https://www.youtube.com/embed/b9EkMc79ZSU',
    },
    {
      title: 'Spider Man - Brand New Day',
      category: 'Action • Movie',
      img: poster3,
      rating: '7.8',
      match: '88% Match',
      desc: 'A nostalgic return to youth and high-school hero tropes, paying tribute to iconic comic styling and comic book history.',
      year: '2021',
      duration: '2h 28m',
      trailerUrl: 'https://www.youtube.com/embed/t06RUxPbp_c',
    },
    {
      title: 'Interstellar',
      category: 'Sci-fi • Movie',
      img: poster5,
      rating: '8.7',
      match: '82% Match',
      desc: 'A nostalgic look back at humanity as space explorers and pioneers, capturing the wonder of old-school celestial voyages.',
      year: '2014',
      duration: '2h 49m',
      trailerUrl: 'https://www.youtube.com/embed/zSWdZVtXT7U',
    },
  ];

  const thrillerMovies: Movie[] = [
    {
      title: 'Squid Game',
      category: 'Suspense • Series',
      img: poster2,
      rating: '8.0',
      match: '97% Match',
      desc: 'An intense, dark psychological suspense survival game forcing debt-ridden participants to play lethal childhood games for cash.',
      year: '2021',
      duration: '1 Season',
      trailerUrl: 'https://www.youtube.com/embed/oqxAJKy0R4A',
    },
    {
      title: 'Breaking Bad',
      category: 'Crime • Series',
      img: poster1,
      rating: '9.5',
      match: '93% Match',
      desc: "Traces a high school chemistry teacher's slow descent into the dark underworld of methamphetamine production and cartel politics.",
      year: '2008',
      duration: '5 Seasons',
      trailerUrl: 'https://www.youtube.com/embed/HhesaQXLuRY',
    },
    {
      title: 'Stranger Things',
      category: 'Adventure • Series',
      img: poster6,
      rating: '8.7',
      match: '84% Match',
      desc: 'Features dark psychological elements, government coverups, and supernatural telekinesis elements in a small Indiana town.',
      year: '2016',
      duration: '4 Seasons',
      trailerUrl: 'https://www.youtube.com/embed/b9EkMc79ZSU',
    },
  ];

  const defaultMovies: Movie[] = [
    {
      title: 'Breaking Bad',
      category: 'Crime • Series',
      img: poster1,
      rating: '9.5',
      match: '96% Match',
      desc: 'Highly recommended crime masterpiece focusing on character transformation, extreme tension, and family survival elements.',
      year: '2008',
      duration: '5 Seasons',
      trailerUrl: 'https://www.youtube.com/embed/HhesaQXLuRY',
    },
    {
      title: 'Avengers: Endgame',
      category: 'SuperHeroes • Movie',
      img: poster4,
      rating: '8.4',
      match: '91% Match',
      desc: 'The epic finale of the infinity saga, featuring grand scales, time travel, and heroic resolution on a cosmic level.',
      year: '2019',
      duration: '3h 01m',
      trailerUrl: 'https://www.youtube.com/embed/TcMBFSGVi1c',
    },
    {
      title: 'Interstellar',
      category: 'Sci-fi • Movie',
      img: poster5,
      rating: '8.7',
      match: '88% Match',
      desc: 'A gorgeous, scientifically grounded sci-fi exploration epic capturing deep space, massive gravity wells, and parental love.',
      year: '2014',
      duration: '2h 49m',
      trailerUrl: 'https://www.youtube.com/embed/zSWdZVtXT7U',
    },
  ];

  const handleSendMessage = (text: string) => {
    if (isProcessing) return;

    setIsProcessing(true);
    setCanvasState('processing');

    const newMessages: Message[] = [...messages, { sender: 'user', text }];
    setMessages(newMessages);

    // Pick movie set based on the query text
    const queryLower = text.toLowerCase();
    let selectedMovies: Movie[] = [];
    let responseIntro = '';

    if (
      queryLower.includes('sci-fi') ||
      queryLower.includes('space') ||
      queryLower.includes('interstellar')
    ) {
      selectedMovies = sciFiMovies;
      responseIntro =
        'Initiated semantic crawl for mind-bending sci-fi matches. I found three premium options that explore quantum physics, deep-space isolation, and cosmic time distortions.';
    } else if (
      queryLower.includes('nostalgic') ||
      queryLower.includes('nostalgia') ||
      queryLower.includes('retro') ||
      queryLower.includes('stranger')
    ) {
      selectedMovies = nostalgicMovies;
      responseIntro =
        'Located retro-vibe matches in the vector database. These options deliver strong synth aesthetics, small-town mysteries, and nostalgic references.';
    } else if (
      queryLower.includes('thriller') ||
      queryLower.includes('psychological') ||
      queryLower.includes('dark') ||
      queryLower.includes('squid') ||
      queryLower.includes('crime')
    ) {
      selectedMovies = thrillerMovies;
      responseIntro =
        'Discovered high-suspense psychological thrillers that examine human behavior under pressure, morality tests, and character degradation.';
    } else {
      selectedMovies = defaultMovies;
      responseIntro = `Processed semantic match for "${text}". I have mapped the closest visual vectors inside our catalog. Here are the recommendations:`;
    }

    setRecommendations(selectedMovies);

    // After 3750ms (the time it takes for 5 processing steps * 750ms), stream the response
    setTimeout(() => {
      // Trigger movie results state
      setActiveMovie(selectedMovies[0]);
      setCanvasState('results');

      // Start streaming AI response text
      const streamingMsg: Message = { sender: 'ai', text: '', isStreaming: true };
      setMessages([...newMessages, streamingMsg]);

      let charIndex = 0;
      const interval = setInterval(() => {
        charIndex += 3; // stream 3 characters per tick for fluid motion
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
      }, 25);
    }, 3750);
  };

  const handleClearChat = () => {
    setMessages([]);
    setCanvasState('idle');
    setActiveMovie(null);
    setRecommendations([]);
  };

  return (
    <MainLayout>
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
          />
        </div>
      </div>
    </MainLayout>
  );
};

export default Search;

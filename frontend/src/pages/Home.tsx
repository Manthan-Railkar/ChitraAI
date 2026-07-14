import React, { useRef } from 'react';
import { MainLayout } from '@/layouts/MainLayout';
import { Sparkles, ArrowRight } from 'lucide-react';
import { Link } from 'react-router-dom';
import VariableProximity from '@/components/reactbits/VariableProximity';
import { RotatingText } from '@/components/reactbits/RotatingText';
import { CardSwap, Card } from '@/components/reactbits/CardSwap';
import { ScrollTimeline } from '@/components/ScrollTimeline';
import ScrollFloat from '@/components/reactbits/ScrollFloat';
import { WhyChitraAI } from '@/components/WhyChitraAI';
import { InsidePipeline } from '@/components/InsidePipeline';

// Import movie posters
import poster1 from '@/assets/posters/poster1.png';
import poster2 from '@/assets/posters/poster2.png';
import poster3 from '@/assets/posters/poster3.jpg';
import poster4 from '@/assets/posters/poster4.jpg';
import poster5 from '@/assets/posters/poster5.jpg';
import poster6 from '@/assets/posters/poster6.png';

export const Home: React.FC = () => {
  const containerRef = useRef<HTMLDivElement>(null);

  const movies = [
    { title: 'Breaking Bad', category: 'Crime • TV Series', img: poster1 },
    { title: 'Squid Game', category: 'Suspense • TV Series', img: poster2 },
    { title: 'Spider Man - Brand New Day', category: 'Action • Movie', img: poster3 },
    { title: 'Avengers: Endgame', category: 'SuperHeroes • Movie', img: poster4 },
    { title: 'Interstellar', category: 'Sci-fi • Movie', img: poster5 },
    { title: 'Stranger Things', category: 'Adventure • TV Series', img: poster6 },
  ];

  return (
    <MainLayout>
      <div
        ref={containerRef}
        className="min-h-[85vh] max-w-[1280px] mx-auto px-6 sm:px-8 lg:px-10 relative flex flex-col justify-center"
      >
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 lg:gap-6 items-center w-full">
          {/* Left Hero Content Section */}
          <div className="lg:col-span-7 flex flex-col items-center lg:items-start text-center lg:text-left z-10">
            {/* Pill Badge */}
            <div className="inline-flex items-center gap-2.5 px-4 py-1.5 bg-white/[0.04] border border-white/[0.08] rounded-full text-white/50 text-[10px] font-semibold tracking-[0.2em] uppercase mb-8">
              <div className="w-5 h-5 rounded-full bg-white/[0.08] flex items-center justify-center text-white/70">
                <Sparkles className="w-3 h-3 stroke-[2]" />
              </div>
              <span>Introducing ChitraAI 1.0</span>
              <span className="bg-primary/20 text-primary rounded px-1.5 py-0.5 text-[9px] font-black uppercase tracking-normal">
                New
              </span>
            </div>

            {/* Headline */}
            <h1 className="text-3xl sm:text-4xl md:text-[3.25rem] lg:text-[3.5rem] font-black tracking-tight leading-[1.1] text-white uppercase select-none mb-2">
              <span className="flex flex-wrap items-baseline justify-center lg:justify-start gap-x-3">
                <span>Search by</span>
                <span className="inline-block">
                  <RotatingText
                    texts={['Moods', 'Themes', 'Memories', 'Natural Language']}
                    mainClassName="text-primary overflow-hidden inline-flex py-0.5 uppercase"
                    staggerFrom="last"
                    initial={{ y: '100%' }}
                    animate={{ y: 0 }}
                    exit={{ y: '-120%' }}
                    staggerDuration={0.025}
                    splitLevelClassName="overflow-hidden pb-1"
                    transition={{ type: 'spring', damping: 30, stiffness: 400 }}
                    rotationInterval={3000}
                    splitBy="characters"
                    auto
                    loop
                  />
                </span>
              </span>
            </h1>

            {/* Variable Proximity subtitle */}
            <div className="mt-1 mb-8">
              <VariableProximity
                label="not just titles"
                fromFontVariationSettings="'wght' 100, 'wdth' 100"
                toFontVariationSettings="'wght' 1000, 'wdth' 150"
                containerRef={containerRef}
                radius={200}
                falloff="linear"
                className="text-primary text-3xl sm:text-4xl md:text-[3.25rem] lg:text-[3.5rem] uppercase cursor-default font-black leading-[1.1]"
              />
            </div>

            {/* Description */}
            <p className="text-white/45 text-sm sm:text-[15px] md:text-base max-w-[480px] font-normal leading-[1.7] tracking-wide mb-10">
              ChitraAI unifies vector embeddings and semantic search to understand the emotional
              context of storylines, dialogue, and visuals — go from vague memories to your next
              favorite film in seconds.
            </p>

            {/* CTA Button */}
            <Link
              to="/search"
              className="group inline-flex items-center gap-3 px-7 py-3.5 rounded-full border border-white/10 bg-white/[0.04] hover:bg-white/[0.08] hover:border-white/20 text-white text-xs font-bold uppercase tracking-[0.2em] transition-all duration-300 backdrop-blur-sm"
            >
              Start Searching
              <ArrowRight className="w-4 h-4 opacity-60 group-hover:opacity-100 group-hover:translate-x-1 transition-all duration-200" />
            </Link>
          </div>

          {/* Right Cards Stack Section */}
          <div className="lg:col-span-5 flex justify-center items-center relative h-[360px] sm:h-[440px] lg:h-[520px] overflow-visible pointer-events-auto mt-16 lg:mt-0 w-full z-20 -translate-x-[50px] sm:-translate-x-[60px] lg:-translate-x-8 lg:translate-y-[50px]">
            <CardSwap
              width={280}
              height={420}
              cardDistance={40}
              verticalDistance={40}
              delay={3000}
              pauseOnHover={false}
              skewAmount={6}
            >
              {movies.map((movie, index) => (
                <Card key={index}>
                  <div className="relative w-full h-full rounded-2xl overflow-hidden border border-white/[0.08] bg-[#09090b] group select-none">
                    <img
                      src={movie.img}
                      alt={movie.title}
                      className="w-full h-full object-cover pointer-events-none"
                    />
                    {/* Gradient overlay */}
                    <div className="absolute inset-x-0 bottom-0 h-[45%] bg-gradient-to-t from-black via-black/60 to-transparent" />
                    {/* Text content */}
                    <div className="absolute inset-x-0 bottom-0 px-5 pb-5 flex flex-col justify-end">
                      <h3 className="text-white text-[15px] font-bold tracking-wide leading-snug">
                        {movie.title}
                      </h3>
                      <p className="text-white/50 text-[10px] font-semibold uppercase tracking-[0.15em] mt-1.5">
                        {movie.category}
                      </p>
                    </div>
                  </div>
                </Card>
              ))}
            </CardSwap>
          </div>
        </div>
      </div>

      {/* ── ScrollFloat Heading Section ─────────────────────── */}
      <div className="relative w-full overflow-hidden py-24 lg:py-32">
        {/* Subtle divider top */}
        <div className="absolute top-0 inset-x-0 h-px bg-white/[0.04]" />

        {/* Background radial */}
        <div
          className="absolute inset-0 pointer-events-none"
          style={{
            background:
              'radial-gradient(ellipse at 50% 0%, rgba(225,29,72,0.05) 0%, transparent 60%)',
          }}
        />

        <div className="relative max-w-[1280px] mx-auto px-6 sm:px-10 lg:px-14 flex flex-col items-center text-center">
          {/* Eyebrow */}
          <div
            className="inline-flex items-center gap-3 mb-10"
            style={{ fontFamily: 'Inter, sans-serif' }}
          >
            <div className="h-px w-8 bg-[#e11d48]/60" />
            <span className="text-[10px] font-semibold tracking-[0.3em] uppercase text-white/30">
              Intelligence at every step
            </span>
            <div className="h-px w-8 bg-[#e11d48]/60" />
          </div>

          {/* ScrollFloat heading — line 1 */}
          <ScrollFloat
            animationDuration={1}
            ease="back.inOut(2)"
            scrollStart="center bottom+=50%"
            scrollEnd="bottom bottom-=40%"
            stagger={0.025}
            textClassName="text-white"
          >
            The path to your
          </ScrollFloat>

          {/* ScrollFloat heading — line 2 (red accent) */}
          <ScrollFloat
            animationDuration={1}
            ease="back.inOut(2)"
            scrollStart="center bottom+=40%"
            scrollEnd="bottom bottom-=50%"
            stagger={0.025}
            textClassName="text-[#e11d48]"
          >
            perfect watch.
          </ScrollFloat>

          {/* Subtext */}
          <p
            className="mt-8 max-w-[440px] text-[15px] leading-[1.75]"
            style={{
              fontFamily: 'Inter, sans-serif',
              color: 'rgba(255,255,255,0.35)',
              fontWeight: 400,
            }}
          >
            Six intelligent steps — from your first taste to a shelf of titles curated exactly for
            the person watching.
          </p>
        </div>

        {/* Subtle divider bottom */}
        <div className="absolute bottom-0 inset-x-0 h-px bg-white/[0.04]" />
      </div>

      {/* Scroll Timeline Section */}
      <ScrollTimeline />

      {/* Why ChitraAI Features Grid Section */}
      <WhyChitraAI />

      {/* Inside ChitraAI Pipeline Section */}
      <InsidePipeline />
    </MainLayout>
  );
};

export default Home;

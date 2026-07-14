/* eslint-disable */
import React, { useEffect, useRef, useState } from 'react';
import { gsap } from 'gsap';
import { ScrollTrigger } from 'gsap/ScrollTrigger';
import { motion, AnimatePresence } from 'framer-motion';
import {
  MessageSquare,
  Code,
  Globe,
  Users,
  Database,
  FileText,
  Cpu,
  Sparkles,
  RefreshCw,
  Terminal,
  Play,
  Check,
  Star,
} from 'lucide-react';
import poster1 from '@/assets/posters/poster1.png';
import poster2 from '@/assets/posters/poster2.png';
import poster3 from '@/assets/posters/poster3.jpg';
import './InsidePipeline.css';

gsap.registerPlugin(ScrollTrigger);

interface PillItem {
  id: string;
  label: string;
  icon: React.ReactNode;
}

const leftPills: PillItem[] = [
  {
    id: 'in-0',
    label: 'User Query',
    icon: <MessageSquare className="w-4.5 h-4.5 text-rose-400" />,
  },
  { id: 'in-1', label: 'Taste Map', icon: <Code className="w-4.5 h-4.5 text-indigo-400" /> },
  { id: 'in-2', label: 'Watch History', icon: <Globe className="w-4.5 h-4.5 text-purple-400" /> },
  { id: 'in-3', label: 'Mood & Context', icon: <Users className="w-4.5 h-4.5 text-blue-400" /> },
];

const rightPills: PillItem[] = [
  {
    id: 'out-0',
    label: 'TMDb Enrichment',
    icon: <Database className="w-4.5 h-4.5 text-emerald-400" />,
  },
  {
    id: 'out-1',
    label: 'AI Explanations',
    icon: <FileText className="w-4.5 h-4.5 text-cyan-400" />,
  },
  { id: 'out-2', label: 'Watchlist API', icon: <Cpu className="w-4.5 h-4.5 text-blue-400" /> },
  {
    id: 'out-3',
    label: 'Personalized Results',
    icon: <Sparkles className="w-4.5 h-4.5 text-amber-400" />,
  },
];

interface RecommendMovie {
  title: string;
  poster: string;
  rating: string;
  match: string;
  desc: string;
}

const recommendMovies: RecommendMovie[] = [
  {
    title: 'Interstellar',
    poster: poster3,
    rating: '8.7',
    match: '98% Match',
    desc: 'Matches your query for time dilation, gravity anomaly, and realistic space travel mechanics.',
  },
  {
    title: 'Arrival',
    poster: poster2,
    rating: '8.0',
    match: '94% Match',
    desc: 'Features deep semantic translation, non-linear time structures, and first contact themes.',
  },
  {
    title: 'Coherence',
    poster: poster1,
    rating: '7.2',
    match: '91% Match',
    desc: 'An interactive, mind-bending movie centering on quantum decoherence and character tension.',
  },
];

export const InsidePipeline: React.FC = () => {
  const containerRef = useRef<HTMLDivElement>(null);
  const headingRef = useRef<HTMLHeadingElement>(null);
  const subtitleRef = useRef<HTMLParagraphElement>(null);
  const pipelineRef = useRef<HTMLDivElement>(null);
  const svgRef = useRef<SVGSVGElement>(null);

  const [coords, setCoords] = useState<{ [key: string]: any }>({});
  const [isMobile, setIsMobile] = useState(false);
  const [activeStage, setActiveStage] = useState<number>(-1); // -1: idle, 0: typing, 1: router, 2: gemini, 3: transformer, 4: qdrant, 5: ranker, 6: enrichment, 7: results

  // Typing state
  const [typedText, setTypedText] = useState('');

  // Sub-animation states
  const [transformerVectors, setTransformerVectors] = useState(false);
  const [qdrantSearch, setQdrantSearch] = useState(false);
  const [rankerSorting, setRankerSorting] = useState(false);

  const masterTimeline = useRef<gsap.core.Timeline | null>(null);

  // Typewriter Simulator
  useEffect(() => {
    if (activeStage === 0) {
      const fullText = 'sci-fi movie about time loop and space travel...';
      let i = 0;
      setTypedText('');
      const interval = setInterval(() => {
        if (i < fullText.length) {
          setTypedText((prev) => prev + fullText.charAt(i));
          i++;
        } else {
          clearInterval(interval);
        }
      }, 30);
      return () => clearInterval(interval);
    }
  }, [activeStage]);

  // Recalculate component coordinates dynamically
  const updateCoords = () => {
    const svgEl = svgRef.current;
    if (!svgEl) return;
    const svgRect = svgEl.getBoundingClientRect();
    const mobile = window.innerWidth < 768;
    setIsMobile(mobile);

    const newCoords: typeof coords = {};
    const ids = [
      'in-0',
      'in-1',
      'in-2',
      'in-3',
      'node-1',
      'node-2',
      'node-3',
      'out-0',
      'out-1',
      'out-2',
      'out-3',
    ];

    ids.forEach((id) => {
      const el = document.getElementById(id);
      if (el) {
        const rect = el.getBoundingClientRect();
        if (mobile) {
          newCoords[id] = {
            topX: rect.left - svgRect.left + rect.width / 2,
            topY: rect.top - svgRect.top,
            bottomX: rect.left - svgRect.left + rect.width / 2,
            bottomY: rect.bottom - svgRect.top,
            centerX: rect.left - svgRect.left + rect.width / 2,
            centerY: rect.top - svgRect.top + rect.height / 2,
          };
        } else {
          newCoords[id] = {
            leftX: rect.left - svgRect.left,
            leftY: rect.top - svgRect.top + rect.height / 2,
            rightX: rect.right - svgRect.left,
            rightY: rect.top - svgRect.top + rect.height / 2,
            centerX: rect.left - svgRect.left + rect.width / 2,
            centerY: rect.top - svgRect.top + rect.height / 2,
          };
        }
      }
    });

    setCoords(newCoords);
  };

  useEffect(() => {
    updateCoords();
    const timer = setTimeout(updateCoords, 150);
    window.addEventListener('resize', updateCoords);
    return () => {
      clearTimeout(timer);
      window.removeEventListener('resize', updateCoords);
    };
  }, []);

  // Compute curve paths (Cubic Beziers)
  const getPathString = (
    start: { x: number; y: number },
    end: { x: number; y: number },
    vertical: boolean
  ) => {
    if (vertical) {
      const midY = (start.y + end.y) / 2;
      return `M ${start.x},${start.y} C ${start.x},${midY} ${end.x},${midY} ${end.x},${end.y}`;
    } else {
      const midX = (start.x + end.x) / 2;
      return `M ${start.x},${start.y} C ${midX},${start.y} ${midX},${end.y} ${end.x},${end.y}`;
    }
  };

  // Build Synchronized GSAP Master Timeline
  useEffect(() => {
    if (Object.keys(coords).length === 0) return;

    const ctx = gsap.context(() => {
      const tl = gsap.timeline({
        scrollTrigger: {
          trigger: containerRef.current,
          start: 'top 65%',
          toggleActions: 'play none none none',
        },
      });

      masterTimeline.current = tl;

      // ── Header Text Entrance ──
      const words = headingRef.current?.querySelectorAll('.word');
      if (words) {
        tl.fromTo(
          words,
          { y: 40, opacity: 0, filter: 'blur(8px)' },
          {
            y: 0,
            opacity: 1,
            filter: 'blur(0px)',
            duration: 0.8,
            stagger: 0.08,
            ease: 'power3.out',
          }
        );
      }
      if (subtitleRef.current) {
        tl.fromTo(
          subtitleRef.current,
          { y: 20, opacity: 0 },
          { y: 0, opacity: 1, duration: 0.7, ease: 'power3.out' },
          '-=0.45'
        );
      }

      // ── Step 0: User Query Typing ──
      tl.call(() => setActiveStage(0)).to({}, { duration: 1.8 }); // Wait for typewriter

      // ── Step 1: Semantic Router Activates ──
      tl.call(() => setActiveStage(1)).to({}, { duration: 1.2 }); // Wait for Router node process

      // ── Step 2: Gemini AI (inside node-2) ──
      tl.call(() => setActiveStage(2)).to({}, { duration: 1.0 }); // Wait for Gemini tag extraction

      // ── Step 3: Sentence Transformer ──
      tl.call(() => {
        setActiveStage(3);
        setTransformerVectors(true);
      }).to({}, { duration: 1.0 }); // Wait for vectors compression

      // ── Step 4: Qdrant Database Vector Search ──
      tl.call(() => {
        setActiveStage(4);
        setQdrantSearch(true);
      }).to({}, { duration: 1.2 }); // Wait for nearest neighbor search

      // ── Step 5: Hybrid Ranker ──
      tl.call(() => {
        setActiveStage(5);
        setRankerSorting(true);
      }).to({}, { duration: 1.4 }); // Wait for movie card sorting

      // ── Step 6: Enrichment (TMDb, Explanations, API) ──
      tl.call(() => setActiveStage(6)).to({}, { duration: 1.2 }); // Wait for output branches to light up

      // ── Step 7: Results Reveal (3 Premium Cards) ──
      tl.call(() => setActiveStage(7));
    }, containerRef);

    return () => ctx.revert();
  }, [coords]);

  // Replay Functionality
  const handleReplay = () => {
    setTypedText('');
    setTransformerVectors(false);
    setQdrantSearch(false);
    setRankerSorting(false);
    setActiveStage(0);

    if (masterTimeline.current) {
      masterTimeline.current.restart();
    }
  };

  return (
    <div
      ref={containerRef}
      className="relative w-full overflow-hidden bg-transparent py-14 sm:py-16 lg:py-20"
    >
      {/* Background ambient lighting */}
      <div className="absolute inset-0 pointer-events-none">
        <div className="absolute top-1/4 right-1/4 w-[45vw] h-[45vw] rounded-full bg-[#e11d48]/4 blur-[130px] floating-glow-1" />
        <div className="absolute bottom-1/4 left-1/4 w-[35vw] h-[35vw] rounded-full bg-[#e11d48]/3 blur-[110px] floating-glow-2" />
        <div
          className="absolute inset-0"
          style={{
            background: 'radial-gradient(circle at 50% 50%, rgba(225,29,72,0.04), transparent 75%)',
          }}
        />
        <div className="absolute top-0 inset-x-0 h-px bg-white/[0.04]" />
      </div>

      <div className="relative z-10 max-w-[1400px] mx-auto px-6 sm:px-10 lg:px-14 flex flex-col items-center">
        {/* Eyebrow */}
        <div className="flex items-center gap-3 mb-6" style={{ fontFamily: 'Inter, sans-serif' }}>
          <div className="h-px w-6 bg-rose-500/60" />
          <span className="text-[10px] font-semibold tracking-[0.25em] uppercase text-white/30">
            Semantic Pipeline
          </span>
          <div className="h-px w-6 bg-rose-500/60" />
        </div>

        {/* Heading */}
        <h2
          ref={headingRef}
          className="text-center font-extrabold text-white leading-[1.1] mb-5 tracking-tight"
          style={{
            fontFamily: 'Plus Jakarta Sans, sans-serif',
            fontSize: 'clamp(2.2rem, 6vw, 4rem)',
          }}
        >
          <span className="inline-block mr-3 word">Inside</span>
          <span className="inline-block word">
            <span className="bg-clip-text text-transparent bg-gradient-to-r from-rose-500 via-[#e11d48] to-purple-500 font-extrabold animate-gradient-text drop-shadow-[0_0_20px_rgba(225,29,72,0.35)]">
              ChitraAI
            </span>
          </span>
        </h2>

        {/* Subtitle */}
        <p
          ref={subtitleRef}
          className="text-center text-white/40 text-base sm:text-lg max-w-[580px] mb-6 leading-[1.6]"
          style={{ fontFamily: 'Inter, sans-serif' }}
        >
          Observe the real-time AI inference pipeline processing a natural language query,
          extracting semantic properties, and serving contextual picks.
        </p>

        {/* Replay Button Centered */}
        <button
          onClick={handleReplay}
          className="bg-white/5 hover:bg-white/10 border border-white/10 hover:border-white/20 rounded-full px-5 py-2 font-mono text-[9px] font-bold uppercase tracking-wider text-white/60 hover:text-white transition-all cursor-pointer flex items-center gap-2 mb-10"
          style={{ backdropFilter: 'blur(8px)' }}
        >
          <Play className="w-3.5 h-3.5 fill-current text-blue-400" />
          Replay Pipeline
        </button>

        {/* ── Pipeline Tree Visualization Area ── */}
        <div
          ref={pipelineRef}
          className="relative w-full min-h-[380px] flex flex-col md:flex-row items-center justify-between mt-6"
        >
          {/* Dynamic SVG Connectors overlay */}
          <svg ref={svgRef} className="pipeline-svg-container" fill="none">
            {coords['node-1'] && coords['node-2'] && coords['node-3'] && (
              <>
                {/* Center Core Connections */}
                {isMobile ? (
                  <>
                    {/* Node 1 -> Node 2 Base */}
                    <path
                      d={getPathString(
                        coords['node-1'].bottomY
                          ? { x: coords['node-1'].bottomX, y: coords['node-1'].bottomY }
                          : coords['node-1'],
                        coords['node-2'].topY
                          ? { x: coords['node-2'].topX, y: coords['node-2'].topY }
                          : coords['node-2'],
                        true
                      )}
                      stroke="rgba(255, 255, 255, 0.12)"
                      strokeWidth="1"
                    />
                    {/* Node 1 -> Node 2 Active */}
                    {(activeStage === 1 || activeStage === 2) && (
                      <path
                        d={getPathString(
                          coords['node-1'].bottomY
                            ? { x: coords['node-1'].bottomX, y: coords['node-1'].bottomY }
                            : coords['node-1'],
                          coords['node-2'].topY
                            ? { x: coords['node-2'].topX, y: coords['node-2'].topY }
                            : coords['node-2'],
                          true
                        )}
                        stroke="#8b5cf6"
                        strokeWidth="1.5"
                        className="connector-pulse active"
                      />
                    )}

                    {/* Node 2 -> Node 3 Base */}
                    <path
                      d={getPathString(
                        coords['node-2'].bottomY
                          ? { x: coords['node-2'].bottomX, y: coords['node-2'].bottomY }
                          : coords['node-2'],
                        coords['node-3'].topY
                          ? { x: coords['node-3'].topX, y: coords['node-3'].topY }
                          : coords['node-3'],
                        true
                      )}
                      stroke="rgba(255, 255, 255, 0.12)"
                      strokeWidth="1"
                    />
                    {/* Node 2 -> Node 3 Active */}
                    {activeStage === 5 && (
                      <path
                        d={getPathString(
                          coords['node-2'].bottomY
                            ? { x: coords['node-2'].bottomX, y: coords['node-2'].bottomY }
                            : coords['node-2'],
                          coords['node-3'].topY
                            ? { x: coords['node-3'].topX, y: coords['node-3'].topY }
                            : coords['node-3'],
                          true
                        )}
                        stroke="#06b6d4"
                        strokeWidth="1.5"
                        className="connector-pulse active"
                      />
                    )}
                  </>
                ) : (
                  <>
                    {/* Node 1 -> Node 2 Base */}
                    <path
                      d={getPathString(
                        coords['node-1'].rightX
                          ? { x: coords['node-1'].rightX, y: coords['node-1'].rightY }
                          : coords['node-1'],
                        coords['node-2'].leftX
                          ? { x: coords['node-2'].leftX, y: coords['node-2'].leftY }
                          : coords['node-2'],
                        false
                      )}
                      stroke="rgba(255, 255, 255, 0.12)"
                      strokeWidth="1"
                    />
                    {/* Node 1 -> Node 2 Active */}
                    {(activeStage === 1 || activeStage === 2) && (
                      <path
                        d={getPathString(
                          coords['node-1'].rightX
                            ? { x: coords['node-1'].rightX, y: coords['node-1'].rightY }
                            : coords['node-1'],
                          coords['node-2'].leftX
                            ? { x: coords['node-2'].leftX, y: coords['node-2'].leftY }
                            : coords['node-2'],
                          false
                        )}
                        stroke="#8b5cf6"
                        strokeWidth="1.5"
                        className="connector-pulse active"
                      />
                    )}

                    {/* Node 2 -> Node 3 Base */}
                    <path
                      d={getPathString(
                        coords['node-2'].rightX
                          ? { x: coords['node-2'].rightX, y: coords['node-2'].rightY }
                          : coords['node-2'],
                        coords['node-3'].leftX
                          ? { x: coords['node-3'].leftX, y: coords['node-3'].leftY }
                          : coords['node-3'],
                        false
                      )}
                      stroke="rgba(255, 255, 255, 0.12)"
                      strokeWidth="1"
                    />
                    {/* Node 2 -> Node 3 Active */}
                    {activeStage === 5 && (
                      <path
                        d={getPathString(
                          coords['node-2'].rightX
                            ? { x: coords['node-2'].rightX, y: coords['node-2'].rightY }
                            : coords['node-2'],
                          coords['node-3'].leftX
                            ? { x: coords['node-3'].leftX, y: coords['node-3'].leftY }
                            : coords['node-3'],
                          false
                        )}
                        stroke="#06b6d4"
                        strokeWidth="1.5"
                        className="connector-pulse active"
                      />
                    )}
                  </>
                )}

                {/* Left side input curves merging in */}
                {leftPills.map((pill, idx) => {
                  if (!coords[pill.id]) return null;
                  const startPt = isMobile
                    ? { x: coords[pill.id].bottomX, y: coords[pill.id].bottomY }
                    : { x: coords[pill.id].rightX, y: coords[pill.id].rightY };
                  const endPt = isMobile
                    ? { x: coords['node-1'].topX, y: coords['node-1'].topY }
                    : { x: coords['node-1'].leftX, y: coords['node-1'].leftY };

                  const isActive =
                    (idx === 0 && activeStage === 0) || (idx > 0 && activeStage === 1);

                  return (
                    <g key={`l-conn-${pill.id}`}>
                      <path
                        d={getPathString(startPt, endPt, isMobile)}
                        stroke="rgba(255, 255, 255, 0.12)"
                        strokeWidth="1"
                      />
                      {isActive && (
                        <path
                          d={getPathString(startPt, endPt, isMobile)}
                          stroke="#e11d48"
                          strokeWidth="1.5"
                          className="connector-pulse active"
                        />
                      )}
                    </g>
                  );
                })}

                {/* Right side output curves branching out */}
                {rightPills.map((pill) => {
                  if (!coords[pill.id]) return null;
                  const startPt = isMobile
                    ? { x: coords['node-3'].bottomX, y: coords['node-3'].bottomY }
                    : { x: coords['node-3'].rightX, y: coords['node-3'].rightY };
                  const endPt = isMobile
                    ? { x: coords[pill.id].topX, y: coords[pill.id].topY }
                    : { x: coords[pill.id].leftX, y: coords[pill.id].leftY };

                  const isActive = activeStage === 6;

                  return (
                    <g key={`r-conn-${pill.id}`}>
                      <path
                        d={getPathString(startPt, endPt, isMobile)}
                        stroke="rgba(255, 255, 255, 0.12)"
                        strokeWidth="1"
                      />
                      {isActive && (
                        <path
                          d={getPathString(startPt, endPt, isMobile)}
                          stroke="#10b981"
                          strokeWidth="1.5"
                          className="connector-pulse active"
                        />
                      )}
                    </g>
                  );
                })}
              </>
            )}
          </svg>

          {/* Central Blue Aura Glow */}
          <div className="central-aura-glow" />

          {/* Animated Energy Packet (Traveling dot) */}
          {/* ══════════════════════════════════════════════════════════ */}
          {/* DESKTOP VIEW                                              */}
          {/* ══════════════════════════════════════════════════════════ */}
          <div className="hidden md:flex w-full items-center justify-between relative z-10">
            {/* Left Inputs Pillar */}
            <div className="flex flex-col gap-6 justify-between h-[260px] w-[180px]">
              {leftPills.map((pill, idx) => {
                const isActive = activeStage === 0 && idx === 0;
                return (
                  <div
                    key={pill.id}
                    id={pill.id}
                    className="pipeline-pill rounded-full py-2 px-5 flex items-center justify-between gap-3 text-white border select-none transition-all duration-300"
                    style={{
                      borderColor: isActive ? '#e11d48' : 'rgba(59, 130, 246, 0.15)',
                      boxShadow: isActive ? '0 0 15px rgba(225,29,72,0.15)' : 'none',
                    }}
                  >
                    <span className="text-[10px] font-bold text-white/50">{pill.icon}</span>
                    <span
                      className="text-[10px] font-semibold tracking-wider text-blue-200"
                      style={{ fontFamily: 'Inter, sans-serif' }}
                    >
                      {pill.label}
                    </span>
                  </div>
                );
              })}
            </div>

            {/* Central Processing Group */}
            <div className="flex items-center gap-12">
              {/* Node 1: Router */}
              <div className="flex flex-col items-center">
                <div
                  id="node-1"
                  className="central-node-outer"
                  style={{
                    borderColor: activeStage === 1 ? '#3b82f6' : 'rgba(59, 130, 246, 0.25)',
                    boxShadow: activeStage === 1 ? '0 0 35px rgba(59,130,246,0.3)' : 'none',
                  }}
                >
                  <div className="central-node-inner">
                    <Terminal
                      className={`w-6.5 h-6.5 text-blue-400 ${activeStage === 1 ? 'animate-pulse' : ''}`}
                    />
                  </div>
                </div>
                <span
                  className="text-[11px] font-extrabold tracking-widest text-blue-100 uppercase mt-4 block"
                  style={{ fontFamily: 'Plus Jakarta Sans, sans-serif' }}
                >
                  Semantic Router
                </span>
              </div>

              {/* Node 2: LLMs Core */}
              <div className="flex flex-col items-center">
                <div
                  id="node-2"
                  className="llms-node-outer flex items-center gap-3"
                  style={{
                    borderColor:
                      activeStage >= 2 && activeStage <= 4 ? '#8b5cf6' : 'rgba(59, 130, 246, 0.25)',
                    boxShadow:
                      activeStage >= 2 && activeStage <= 4
                        ? '0 0 45px rgba(139,92,246,0.35)'
                        : 'none',
                  }}
                >
                  <div
                    className="sub-icon-box"
                    style={{
                      borderColor: activeStage === 2 ? '#8b5cf6' : 'rgba(59, 130, 246, 0.3)',
                      background: activeStage === 2 ? 'rgba(139,92,246,0.1)' : 'rgba(8,14,44,0.5)',
                    }}
                  >
                    <Sparkles
                      className={`w-5.5 h-5.5 text-blue-400 ${activeStage === 2 ? 'animate-bounce' : ''}`}
                    />
                  </div>
                  <div
                    className="sub-icon-box"
                    style={{
                      borderColor: activeStage === 3 ? '#a855f7' : 'rgba(59, 130, 246, 0.3)',
                      background: activeStage === 3 ? 'rgba(168,85,247,0.1)' : 'rgba(8,14,44,0.5)',
                    }}
                  >
                    <RefreshCw
                      className={`w-5.5 h-5.5 text-indigo-400 ${activeStage === 3 ? 'animate-spin' : ''}`}
                    />
                  </div>
                  <div
                    className="sub-icon-box"
                    style={{
                      borderColor: activeStage === 4 ? '#3b82f6' : 'rgba(59, 130, 246, 0.3)',
                      background: activeStage === 4 ? 'rgba(59,130,246,0.1)' : 'rgba(8,14,44,0.5)',
                    }}
                  >
                    <Database
                      className={`w-5.5 h-5.5 text-purple-400 ${activeStage === 4 ? 'animate-pulse' : ''}`}
                    />
                  </div>
                </div>
                <span
                  className="text-[11px] font-extrabold tracking-widest text-blue-100 uppercase mt-4 block"
                  style={{ fontFamily: 'Plus Jakarta Sans, sans-serif' }}
                >
                  Core LLMs
                </span>
              </div>

              {/* Node 3: Ranker */}
              <div className="flex flex-col items-center">
                <div
                  id="node-3"
                  className="central-node-outer"
                  style={{
                    borderColor: activeStage === 5 ? '#06b6d4' : 'rgba(59, 130, 246, 0.25)',
                    boxShadow: activeStage === 5 ? '0 0 35px rgba(6,182,212,0.3)' : 'none',
                  }}
                >
                  <div className="central-node-inner">
                    <Cpu
                      className={`w-6.5 h-6.5 text-blue-400 ${activeStage === 5 ? 'animate-spin' : ''}`}
                      style={{ animationDuration: '3s' }}
                    />
                  </div>
                </div>
                <span
                  className="text-[11px] font-extrabold tracking-widest text-blue-100 uppercase mt-4 block"
                  style={{ fontFamily: 'Plus Jakarta Sans, sans-serif' }}
                >
                  Hybrid Ranker
                </span>
              </div>
            </div>

            {/* Right Outputs Pillar */}
            <div className="flex flex-col gap-6 justify-between h-[260px] w-[180px]">
              {rightPills.map((pill) => {
                const isActive = activeStage >= 6;
                return (
                  <div
                    key={pill.id}
                    id={pill.id}
                    className="pipeline-pill rounded-full py-2 px-5 flex items-center justify-between gap-3 text-white border select-none transition-all duration-300"
                    style={{
                      borderColor: isActive ? '#10b981' : 'rgba(59, 130, 246, 0.15)',
                      boxShadow: isActive ? '0 0 15px rgba(16,185,129,0.12)' : 'none',
                    }}
                  >
                    <span className="text-[10px] font-bold text-white/50">{pill.icon}</span>
                    <span
                      className="text-[10px] font-semibold tracking-wider text-blue-200"
                      style={{ fontFamily: 'Inter, sans-serif' }}
                    >
                      {pill.label}
                    </span>
                  </div>
                );
              })}
            </div>
          </div>

          {/* ══════════════════════════════════════════════════════════ */}
          {/* MOBILE VIEW                                               */}
          {/* ══════════════════════════════════════════════════════════ */}
          <div className="md:hidden flex flex-col items-center gap-8 w-full relative z-10">
            {/* Top Inputs Grid */}
            <div className="grid grid-cols-2 gap-4 w-full px-4">
              {leftPills.map((pill, idx) => {
                const isActive = activeStage === 0 && idx === 0;
                return (
                  <div
                    key={pill.id}
                    id={pill.id}
                    className="pipeline-pill rounded-full py-2 px-4 flex items-center justify-center gap-2 text-white border select-none transition-all duration-300"
                    style={{
                      borderColor: isActive ? '#e11d48' : 'rgba(59, 130, 246, 0.15)',
                    }}
                  >
                    <span>{pill.icon}</span>
                    <span
                      className="text-[9px] font-semibold tracking-wider text-blue-200"
                      style={{ fontFamily: 'Inter, sans-serif' }}
                    >
                      {pill.label}
                    </span>
                  </div>
                );
              })}
            </div>

            {/* Central Vertical Stack */}
            <div className="flex flex-col items-center gap-6 mt-2 mb-2">
              {/* Node 1 */}
              <div className="flex flex-col items-center gap-2">
                <div
                  id="node-1"
                  className="central-node-outer"
                  style={{
                    borderColor: activeStage === 1 ? '#3b82f6' : 'rgba(59, 130, 246, 0.25)',
                    boxShadow: activeStage === 1 ? '0 0 35px rgba(59,130,246,0.3)' : 'none',
                  }}
                >
                  <div className="central-node-inner">
                    <Terminal className="w-6.5 h-6.5 text-blue-400" />
                  </div>
                </div>
                <span
                  className="text-[10px] font-extrabold tracking-widest text-blue-100 uppercase"
                  style={{ fontFamily: 'Plus Jakarta Sans, sans-serif' }}
                >
                  Router
                </span>
              </div>

              {/* Node 2 */}
              <div className="flex flex-col items-center gap-2">
                <div
                  id="node-2"
                  className="llms-node-outer flex flex-col gap-3 py-3 px-2"
                  style={{
                    borderColor:
                      activeStage >= 2 && activeStage <= 4 ? '#8b5cf6' : 'rgba(59, 130, 246, 0.25)',
                    boxShadow:
                      activeStage >= 2 && activeStage <= 4
                        ? '0 0 45px rgba(139,92,246,0.35)'
                        : 'none',
                  }}
                >
                  <div className="sub-icon-box">
                    <Sparkles className="w-5.5 h-5.5 text-blue-400" />
                  </div>
                  <div className="sub-icon-box">
                    <RefreshCw className="w-5.5 h-5.5 text-indigo-400" />
                  </div>
                  <div className="sub-icon-box">
                    <Database className="w-5.5 h-5.5 text-purple-400" />
                  </div>
                </div>
                <span
                  className="text-[10px] font-extrabold tracking-widest text-blue-100 uppercase"
                  style={{ fontFamily: 'Plus Jakarta Sans, sans-serif' }}
                >
                  LLMs
                </span>
              </div>

              {/* Node 3 */}
              <div className="flex flex-col items-center gap-2">
                <div
                  id="node-3"
                  className="central-node-outer"
                  style={{
                    borderColor: activeStage === 5 ? '#06b6d4' : 'rgba(59, 130, 246, 0.25)',
                    boxShadow: activeStage === 5 ? '0 0 35px rgba(6,182,212,0.3)' : 'none',
                  }}
                >
                  <div className="central-node-inner">
                    <Cpu
                      className="w-6.5 h-6.5 text-blue-400 animate-spin"
                      style={{ animationDuration: '3s' }}
                    />
                  </div>
                </div>
                <span
                  className="text-[10px] font-extrabold tracking-widest text-blue-100 uppercase"
                  style={{ fontFamily: 'Plus Jakarta Sans, sans-serif' }}
                >
                  Ranker
                </span>
              </div>
            </div>

            {/* Bottom Outputs Grid */}
            <div className="grid grid-cols-2 gap-4 w-full px-4">
              {rightPills.map((pill) => {
                const isActive = activeStage >= 6;
                return (
                  <div
                    key={pill.id}
                    id={pill.id}
                    className="pipeline-pill rounded-full py-2 px-4 flex items-center justify-center gap-2 text-white border select-none transition-all duration-300"
                    style={{
                      borderColor: isActive ? '#10b981' : 'rgba(59, 130, 246, 0.15)',
                    }}
                  >
                    <span>{pill.icon}</span>
                    <span
                      className="text-[9px] font-semibold tracking-wider text-blue-200"
                      style={{ fontFamily: 'Inter, sans-serif' }}
                    >
                      {pill.label}
                    </span>
                  </div>
                );
              })}
            </div>
          </div>
        </div>

        {/* ── Inline Node Sub-Animations Section (Centered details block below) ── */}
        {activeStage >= 0 && activeStage <= 6 && (
          <div className="w-full max-w-[800px] mt-8 min-h-[120px] flex flex-col items-center justify-center relative">
            {/* Stage 0: Typewriter */}
            {activeStage === 0 && (
              <motion.div
                initial={{ opacity: 0, y: 15 }}
                animate={{ opacity: 1, y: 0 }}
                className="text-left w-full max-w-[450px] p-4 bg-black/60 rounded-xl border border-white/[0.06] select-none"
              >
                <div className="flex items-center gap-2 mb-2">
                  <div className="w-2 h-2 rounded-full bg-[#e11d48]" />
                  <span
                    className="text-[9px] font-bold text-white/30 uppercase tracking-widest"
                    style={{ fontFamily: 'Inter, sans-serif' }}
                  >
                    User Request Entry
                  </span>
                </div>
                <p className="text-sm font-mono text-white/95 leading-normal">
                  {typedText}
                  <span className="typewriter-cursor" />
                </p>
              </motion.div>
            )}

            {/* Stage 2: Gemini Tags */}
            {activeStage === 2 && (
              <div className="flex flex-col items-center text-center">
                <span
                  className="text-[10px] font-bold tracking-widest text-[#8b5cf6] uppercase mb-4"
                  style={{ fontFamily: 'Inter, sans-serif' }}
                >
                  Gemini Extraction
                </span>
                <div className="flex flex-wrap gap-2.5 justify-center">
                  {[
                    'Intent: Search',
                    'Mood: Epic',
                    'Theme: Time loop',
                    'Genre: Sci-Fi',
                    'Constraints: >7.5 rating',
                  ].map((tag, i) => (
                    <motion.span
                      key={tag}
                      initial={{ opacity: 0, scale: 0.7, y: 10 }}
                      animate={{ opacity: 1, scale: 1, y: 0 }}
                      transition={{ delay: i * 0.12, duration: 0.35 }}
                      className="px-3 py-1 rounded-full bg-indigo-500/10 border border-indigo-500/20 text-[#8b5cf6] text-[10px] font-bold uppercase tracking-wider"
                      style={{ fontFamily: 'Inter, sans-serif' }}
                    >
                      {tag}
                    </motion.span>
                  ))}
                </div>
              </div>
            )}

            {/* Stage 3: Sentence Transformer Vectors */}
            {activeStage === 3 && (
              <div className="relative w-full h-[100px] flex flex-col items-center justify-center">
                <span
                  className="text-[10px] font-bold tracking-widest text-[#a855f7] uppercase mb-3"
                  style={{ fontFamily: 'Inter, sans-serif' }}
                >
                  Dense Embedding Generation
                </span>
                <div className="relative w-full max-w-[200px] h-[50px] flex items-center justify-center">
                  {Array.from({ length: 12 }).map((_, i) => (
                    <div
                      key={i}
                      className={`vector-dot ${transformerVectors ? 'active' : ''}`}
                      style={{
                        top: `${20 + Math.sin(i) * 18}px`,
                        left: `${80 + Math.cos(i) * 35}px`,
                        transitionDelay: `${i * 0.04}s`,
                      }}
                    />
                  ))}
                </div>
              </div>
            )}

            {/* Stage 4: Qdrant Database Nearest Neighbors */}
            {activeStage === 4 && (
              <div className="flex flex-col items-center text-center">
                <span
                  className="text-[10px] font-bold tracking-widest text-[#3b82f6] uppercase mb-3"
                  style={{ fontFamily: 'Inter, sans-serif' }}
                >
                  Qdrant Vector Search
                </span>
                <div className="relative w-full max-w-[240px] h-[60px] flex items-center justify-center border border-white/[0.04] bg-black/30 rounded-xl px-4">
                  {Array.from({ length: 16 }).map((_, i) => (
                    <div
                      key={i}
                      className="absolute w-1.5 h-1.5 rounded-full bg-blue-500/30"
                      style={{
                        top: `${10 + Math.random() * 40}px`,
                        left: `${10 + Math.random() * 220}px`,
                      }}
                    />
                  ))}
                  {qdrantSearch && (
                    <>
                      <div className="absolute w-3.5 h-3.5 rounded-full bg-[#3b82f6]/20 animate-ping" />
                      <div className="absolute w-2.5 h-2.5 rounded-full bg-[#3b82f6] shadow-[0_0_10px_#3b82f6]" />
                      <span className="absolute bottom-1 right-2 text-[8px] font-mono text-white/30 uppercase tracking-widest">
                        Matched k-NN
                      </span>
                    </>
                  )}
                </div>
              </div>
            )}

            {/* Stage 5: Hybrid Ranker Sorting */}
            {activeStage === 5 && (
              <div className="flex flex-col items-center w-full max-w-[280px]">
                <span
                  className="text-[10px] font-bold tracking-widest text-[#06b6d4] uppercase mb-3"
                  style={{ fontFamily: 'Inter, sans-serif' }}
                >
                  Hybrid Scoring & Reranking
                </span>
                <div className="flex flex-col gap-1.5 w-full">
                  {rankerSorting ? (
                    <>
                      <motion.div
                        animate={{ y: [0, 12, 0] }}
                        transition={{ duration: 2.0, repeat: Infinity, ease: 'easeInOut' }}
                        className="px-3 py-1.5 rounded-lg bg-[#06b6d4]/10 border border-[#06b6d4]/20 text-[9px] font-bold text-[#06b6d4] flex justify-between items-center"
                      >
                        <span style={{ fontFamily: 'Plus Jakarta Sans, sans-serif' }}>
                          Interstellar
                        </span>
                        <span className="font-mono text-[#06b6d4]">Score: 98%</span>
                      </motion.div>
                      <motion.div
                        animate={{ y: [0, -12, 0] }}
                        transition={{ duration: 2.0, repeat: Infinity, ease: 'easeInOut' }}
                        className="px-3 py-1.5 rounded-lg bg-white/[0.02] border border-white/[0.04] text-[9px] font-bold text-white/40 flex justify-between items-center"
                      >
                        <span style={{ fontFamily: 'Plus Jakarta Sans, sans-serif' }}>Arrival</span>
                        <span className="font-mono text-white/20">Score: 94%</span>
                      </motion.div>
                    </>
                  ) : (
                    <div className="text-[9px] text-white/20 uppercase tracking-widest">
                      Sorting Candidates...
                    </div>
                  )}
                </div>
              </div>
            )}

            {/* Stage 6: Enrichment Sync */}
            {activeStage === 6 && (
              <div className="flex flex-col items-center text-center">
                <span
                  className="text-[10px] font-bold tracking-widest text-[#10b981] uppercase mb-3"
                  style={{ fontFamily: 'Inter, sans-serif' }}
                >
                  TMDb API Metadata Sync
                </span>
                <div className="flex items-center gap-3">
                  <div className="w-8 h-8 rounded-full bg-[#10b981]/15 border border-[#10b981]/30 flex items-center justify-center text-[#10b981]">
                    <Check className="w-4 h-4 stroke-[3]" />
                  </div>
                  <div className="text-left">
                    <p
                      className="text-xs font-bold text-white"
                      style={{ fontFamily: 'Plus Jakarta Sans, sans-serif' }}
                    >
                      Enrichment Complete
                    </p>
                    <p
                      className="text-[8px] font-semibold text-white/30 uppercase tracking-wider"
                      style={{ fontFamily: 'Inter, sans-serif' }}
                    >
                      Cast, Trailers, Streaming synced
                    </p>
                  </div>
                </div>
              </div>
            )}
          </div>
        )}

        {/* ── Stage 7: Personalized Recommendation Results Reveal (3 Premium Cards) ── */}
        <AnimatePresence>
          {activeStage === 7 && (
            <motion.div
              initial={{ opacity: 0, y: 30 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6, ease: 'easeOut' }}
              className="w-full mt-6 pt-6 border-t border-white/[0.04]"
            >
              <div className="text-center mb-8">
                <span
                  className="text-[9px] font-bold tracking-[0.25em] text-amber-500 uppercase block mb-1.5"
                  style={{ fontFamily: 'Inter, sans-serif' }}
                >
                  Recommendations Ready
                </span>
                <h3
                  className="text-white text-xl font-extrabold tracking-tight"
                  style={{ fontFamily: 'Plus Jakarta Sans, sans-serif' }}
                >
                  Engineered for you
                </h3>
              </div>

              {/* Responsive 3-card grid */}
              <div className="grid grid-cols-1 md:grid-cols-3 gap-6 w-full max-w-[1100px] mx-auto px-4">
                {recommendMovies.map((movie, idx) => (
                  <motion.div
                    key={movie.title}
                    initial={{ opacity: 0, y: 24 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: idx * 0.15, duration: 0.5 }}
                    className="relative rounded-2xl overflow-hidden border border-white/[0.06] bg-black/45 p-4 flex flex-col justify-between"
                    style={{
                      boxShadow:
                        '0 16px 40px rgba(0,0,0,0.5), inset 0 1px 1px rgba(255,255,255,0.02)',
                      backdropFilter: 'blur(10px)',
                    }}
                  >
                    {/* Poster Image with sheen */}
                    <div className="relative aspect-[2/3] w-full rounded-xl overflow-hidden border border-white/[0.04] mb-4">
                      <img src={movie.poster} alt="" className="w-full h-full object-cover" />
                      <div className="absolute inset-0 bg-gradient-to-t from-black/60 to-transparent pointer-events-none" />
                      <div
                        className="absolute top-3 right-3 bg-amber-500 text-white text-[9px] font-black tracking-wider px-2 py-0.5 rounded-full select-none"
                        style={{ fontFamily: 'Inter, sans-serif' }}
                      >
                        {movie.match}
                      </div>
                    </div>

                    {/* Metadata */}
                    <div className="text-left">
                      <div className="flex items-center justify-between mb-2">
                        <h4
                          className="text-white text-base font-extrabold tracking-tight"
                          style={{ fontFamily: 'Plus Jakarta Sans, sans-serif' }}
                        >
                          {movie.title}
                        </h4>
                        <div className="flex items-center gap-1 text-amber-400">
                          <Star className="w-3.5 h-3.5 fill-current" />
                          <span className="text-[10px] font-bold text-white/80">
                            {movie.rating}
                          </span>
                        </div>
                      </div>

                      {/* Explanation */}
                      <p
                        className="text-[11px] leading-[1.6] text-white/45"
                        style={{ fontFamily: 'Inter, sans-serif' }}
                      >
                        {movie.desc}
                      </p>
                    </div>
                  </motion.div>
                ))}
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </div>
  );
};

export default InsidePipeline;

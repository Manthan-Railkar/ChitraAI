import React, { useLayoutEffect, useRef, useState } from 'react';
import { gsap } from 'gsap';
import { ScrollTrigger } from 'gsap/ScrollTrigger';
import { motion, AnimatePresence } from 'framer-motion';
import poster1 from '@/assets/posters/poster1.png';
import poster2 from '@/assets/posters/poster2.png';
import poster3 from '@/assets/posters/poster3.jpg';
import poster4 from '@/assets/posters/poster4.jpg';
import poster5 from '@/assets/posters/poster5.jpg';
import poster6 from '@/assets/posters/poster6.png';

import { Star, Check } from 'lucide-react';

gsap.registerPlugin(ScrollTrigger);

/* ─── Step data ──────────────────────────────────────────────────── */
interface Step {
  num: string;
  label: string;
  heading: string;
  sub: string;
  body: string;
  posters: string[];
  tag: string;
}

const steps: Step[] = [
  {
    num: '01',
    label: 'Discover your taste',
    heading: 'Tell us what\nyou love',
    sub: 'Your personal taste atlas',
    body: 'Browse a curated starter collection of films and shows. Every choice you make teaches ChitraAI something new about the stories that move you.',
    posters: [poster1, poster2, poster3],
    tag: 'Taste mapping',
  },
  {
    num: '02',
    label: 'AI preference analysis',
    heading: 'Your tastes,\ndeep learned',
    sub: 'Semantic profile generation',
    body: 'Our models extract invisible threads connecting your choices — genre, tone, pacing, emotional arcs, visual style. The result evolves with every interaction.',
    posters: [poster4, poster5, poster6],
    tag: 'Neural analysis',
  },
  {
    num: '03',
    label: 'Semantic search',
    heading: 'Words become\nunderstanding',
    sub: 'Vector embedding engine',
    body: 'Type anything — a half-remembered scene, a colour palette, a feeling. Vector embeddings translate ambiguity into precise cinematic meaning.',
    posters: [poster2, poster4, poster1],
    tag: 'Semantic engine',
  },
  {
    num: '04',
    label: 'Intelligent ranking',
    heading: 'The right title\nrises to the top',
    sub: 'Personalised scoring',
    body: 'Thousands of candidates are scored against your unique profile in milliseconds — ranked not by popularity, but by personal fit.',
    posters: [poster5, poster3, poster6],
    tag: 'Smart ranking',
  },
  {
    num: '05',
    label: 'Personalised picks',
    heading: 'A shelf built\njust for you',
    sub: 'Dynamic recommendation layer',
    body: 'An ever-evolving shelf of titles perfectly matched to your taste right now. Hidden gems you would never find browsing surface effortlessly.',
    posters: [poster6, poster1, poster4],
    tag: 'Curated for you',
  },
  {
    num: '06',
    label: 'Watch, rate & improve',
    heading: 'Every watch\nmakes it smarter',
    sub: 'Continuous learning loop',
    body: 'Every rating, every save, every rewatch feeds the model. ChitraAI continuously refines your profile — sharper and more personal over time.',
    posters: [poster3, poster5, poster2],
    tag: 'Feedback loop',
  },
];

/* ─── Desktop: Floating 3-card stack ───────────────────────────── */
interface PosterStackProps {
  posters: string[];
  stepIndex: number;
}

const PosterStack: React.FC<PosterStackProps> = ({ posters, stepIndex }) => {
  const transforms = [
    { rotate: '-6deg', x: '-60px', y: '10px', z: 0, scale: 0.88 },
    { rotate: '0deg', x: '0px', y: '-14px', z: 1, scale: 1.0 },
    { rotate: '7deg', x: '58px', y: '8px', z: 0, scale: 0.88 },
  ];

  return (
    <div className="relative w-full h-full flex items-center justify-center">
      <div
        className="absolute inset-0 pointer-events-none"
        style={{
          background:
            'radial-gradient(ellipse at 50% 80%, rgba(225,29,72,0.18) 0%, transparent 65%)',
        }}
      />
      <div className="relative" style={{ width: 220, height: 330 }}>
        {posters.map((poster, i) => {
          const t = transforms[i];
          return (
            <motion.div
              key={`${stepIndex}-${i}`}
              className="absolute inset-0 rounded-2xl overflow-hidden border border-white/[0.08]"
              initial={{ opacity: 0, y: 40, scale: 0.9 }}
              animate={{ opacity: 1, y: 0, scale: t.scale, rotate: t.rotate, x: t.x }}
              transition={{ delay: i * 0.07, duration: 0.6, ease: [0.4, 0, 0.2, 1] }}
              style={{
                zIndex: t.z,
                transformOrigin: 'center bottom',
                boxShadow:
                  i === 1
                    ? '0 32px 64px rgba(0,0,0,0.7), 0 0 0 1px rgba(225,29,72,0.12)'
                    : '0 16px 40px rgba(0,0,0,0.5)',
              }}
            >
              <img src={poster} alt="" className="w-full h-full object-cover" />
              {i === 1 && (
                <div className="absolute inset-0 bg-gradient-to-br from-white/[0.04] via-transparent to-transparent" />
              )}
              <div className="absolute inset-x-0 bottom-0 h-1/3 bg-gradient-to-t from-black/80 to-transparent" />
            </motion.div>
          );
        })}
      </div>
      <motion.div
        initial={{ opacity: 0, y: 12 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.4, duration: 0.5 }}
        className="absolute bottom-6 left-1/2 -translate-x-1/2 px-4 py-1.5 rounded-full border border-white/[0.08] backdrop-blur-md text-[10px] font-semibold tracking-[0.18em] text-white/40 uppercase"
        style={{ background: 'rgba(255,255,255,0.03)' }}
      >
        Step {steps[stepIndex]?.num} · {steps[stepIndex]?.tag}
      </motion.div>
    </div>
  );
};

/* ─── Mobile: Full-width cinematic card ────────────────────────── */
interface MobileCardProps {
  posters: string[];
  stepIndex: number;
  tag: string;
}

const MobileCard: React.FC<MobileCardProps> = ({ posters, stepIndex, tag }) => {
  /* Show the primary poster large + a small peek of the second */
  return (
    <div className="relative w-[180px] mx-auto" style={{ height: 260 }}>
      {/* Depth card behind (slightly rotated) */}
      <div
        className="absolute inset-x-4 inset-y-3 rounded-2xl overflow-hidden border border-white/[0.05]"
        style={{
          transform: 'rotate(2.5deg) scale(0.97)',
          zIndex: 0,
          background: '#111',
          boxShadow: '0 12px 40px rgba(0,0,0,0.6)',
        }}
      >
        <img
          src={posters[2] ?? posters[0]}
          alt=""
          className="w-full h-full object-cover opacity-50"
        />
      </div>

      {/* Secondary card */}
      <div
        className="absolute inset-x-2 inset-y-1.5 rounded-2xl overflow-hidden border border-white/[0.06]"
        style={{
          transform: 'rotate(-1.5deg) scale(0.985)',
          zIndex: 1,
          boxShadow: '0 16px 48px rgba(0,0,0,0.65)',
        }}
      >
        <img
          src={posters[1] ?? posters[0]}
          alt=""
          className="w-full h-full object-cover opacity-70"
        />
      </div>

      {/* Primary card (front) */}
      <motion.div
        key={`mb-${stepIndex}`}
        className="absolute inset-0 rounded-2xl overflow-hidden border border-white/[0.09]"
        initial={{ opacity: 0, scale: 0.94, y: 20 }}
        animate={{ opacity: 1, scale: 1, y: 0 }}
        exit={{ opacity: 0, scale: 0.96, y: -12 }}
        transition={{ duration: 0.45, ease: [0.4, 0, 0.2, 1] }}
        style={{
          zIndex: 2,
          boxShadow: '0 0 0 1px rgba(225,29,72,0.10), 0 24px 64px rgba(0,0,0,0.75)',
        }}
      >
        <img src={posters[0]} alt="" className="w-full h-full object-cover" />
        {/* Glass sheen */}
        <div className="absolute inset-0 bg-gradient-to-br from-white/[0.04] to-transparent pointer-events-none" />
        {/* Bottom fade */}
        <div className="absolute inset-x-0 bottom-0 h-2/5 bg-gradient-to-t from-black/90 via-black/40 to-transparent" />
        {/* Tag */}
        <div className="absolute bottom-4 left-4">
          <span
            className="px-3 py-1 rounded-full text-[10px] font-bold uppercase tracking-[0.18em]"
            style={{
              background: 'rgba(225,29,72,0.15)',
              color: '#e11d48',
              border: '1px solid rgba(225,29,72,0.2)',
              backdropFilter: 'blur(8px)',
            }}
          >
            {tag}
          </span>
        </div>
      </motion.div>

      {/* Ambient glow */}
      <div
        className="absolute -bottom-4 inset-x-8 h-16 pointer-events-none"
        style={{
          background: 'radial-gradient(ellipse at 50% 100%, rgba(225,29,72,0.20), transparent 70%)',
          filter: 'blur(12px)',
        }}
      />
    </div>
  );
};

/* ─── Feedback Interactive Card (Redesigned 6th Step) ────────────── */
const FeedbackInteractiveCard: React.FC = () => {
  return (
    <div className="relative w-full h-full flex flex-col justify-between p-5 select-none bg-[#0a0a0b] text-left">
      {/* Background poster */}
      <div className="absolute inset-0">
        <img src={poster5} alt="" className="w-full h-full object-cover opacity-25" />
        <div className="absolute inset-0 bg-gradient-to-t from-[#040406] via-[#040406]/70 to-[#040406]/40" />
      </div>

      {/* Top Header */}
      <div className="relative z-10 flex items-center justify-between">
        <span
          className="text-[9px] font-bold tracking-widest text-white/30 uppercase"
          style={{ fontFamily: 'Inter, sans-serif' }}
        >
          Live feedback
        </span>
        <div className="flex items-center gap-1.5">
          <div className="w-1.5 h-1.5 rounded-full bg-[#e11d48] animate-pulse" />
          <span
            className="text-[9px] font-bold text-[#e11d48] tracking-wider uppercase"
            style={{ fontFamily: 'Inter, sans-serif' }}
          >
            Interactive
          </span>
        </div>
      </div>

      {/* Middle ratings panel */}
      <div className="relative z-10 flex flex-col items-center justify-center gap-3 my-auto text-center">
        <div>
          <p className="text-[9px] font-medium text-white/40 uppercase tracking-[0.2em] mb-1">
            Rate recommendation
          </p>
          <h4
            className="text-white text-base font-extrabold tracking-tight"
            style={{ fontFamily: 'Plus Jakarta Sans, sans-serif' }}
          >
            Interstellar
          </h4>
        </div>

        {/* Pulsing Stars */}
        <div className="flex gap-1">
          {[1, 2, 3, 4, 5].map((star) => (
            <motion.div
              key={star}
              animate={{
                scale: star <= 4 ? [1, 1.15, 1] : 1,
                color: star <= 4 ? '#e11d48' : 'rgba(255,255,255,0.12)',
              }}
              transition={{
                repeat: star <= 4 ? Infinity : 0,
                repeatDelay: 2.2,
                duration: 0.4,
                delay: star * 0.12,
              }}
            >
              <Star className="w-4 h-4 fill-current" />
            </motion.div>
          ))}
        </div>
      </div>

      {/* Bottom sliding toast feedback */}
      <motion.div
        initial={{ opacity: 0, y: 15 }}
        animate={{ opacity: [0, 1, 1, 0], y: [15, 0, 0, 15] }}
        transition={{
          repeat: Infinity,
          duration: 3.2,
          ease: 'easeInOut',
          times: [0, 0.12, 0.88, 1],
        }}
        className="relative z-10 w-full rounded-xl border border-white/[0.06] bg-black/85 backdrop-blur-md px-3 py-2 flex items-center gap-2.5"
        style={{ boxShadow: '0 8px 32px rgba(0,0,0,0.6)' }}
      >
        <div className="w-5 h-5 rounded-full bg-[#e11d48]/15 flex items-center justify-center text-[#e11d48] shrink-0 border border-[#e11d48]/30">
          <Check className="w-2.5 h-2.5 stroke-[3]" />
        </div>
        <div className="text-left leading-tight">
          <p
            className="text-[10px] font-bold text-white"
            style={{ fontFamily: 'Plus Jakarta Sans, sans-serif' }}
          >
            Profile updated
          </p>
          <p
            className="text-[8px] font-semibold text-white/40 uppercase tracking-widest"
            style={{ fontFamily: 'Inter, sans-serif' }}
          >
            +18% Sci-fi affinity
          </p>
        </div>
      </motion.div>
    </div>
  );
};

/* ─── Desktop: Timeline label item ────────────────────────────── */
interface LabelItemProps {
  step: Step;
  isActive: boolean;
  onClick: () => void;
}

const LabelItem: React.FC<LabelItemProps> = ({ step, isActive, onClick }) => (
  <button
    onClick={onClick}
    className="group flex items-start gap-5 text-left w-full focus:outline-none py-1"
  >
    <span
      className="shrink-0 text-[10px] tracking-[0.2em] font-medium mt-1 transition-all duration-500 w-7 text-right"
      style={{
        fontFamily: 'Inter, monospace',
        color: isActive ? '#e11d48' : 'rgba(255,255,255,0.16)',
      }}
    >
      {step.num}
    </span>
    <motion.span
      animate={{
        opacity: isActive ? 1 : 0.2,
        x: isActive ? 3 : 0,
        scale: isActive ? 1.025 : 1,
        filter: isActive ? 'blur(0px)' : 'blur(0.5px)',
      }}
      transition={{ duration: 0.45, ease: [0.4, 0, 0.2, 1] }}
      className="text-[1.05rem] sm:text-[1.15rem] lg:text-[1.2rem] font-bold text-white leading-snug capitalize"
      style={{
        fontFamily: 'Plus Jakarta Sans, sans-serif',
        textShadow: isActive
          ? '0 0 28px rgba(225,29,72,0.45), 0 0 60px rgba(225,29,72,0.12)'
          : 'none',
      }}
    >
      {step.label}
    </motion.span>
  </button>
);

/* ─── Shared background effects ─────────────────────────────────── */
const CinematicBg: React.FC<{ activeStep: number }> = ({ activeStep }) => (
  <div className="absolute inset-0 pointer-events-none">
    <div
      style={{
        position: 'absolute',
        top: 0,
        left: '50%',
        transform: 'translateX(-50%)',
        width: '900px',
        height: '600px',
        background:
          'radial-gradient(ellipse at 50% 0%, rgba(255,255,255,0.04) 0%, transparent 70%)',
      }}
    />
    <motion.div
      animate={{ x: `${(activeStep / (steps.length - 1)) * 30 - 15}%` }}
      transition={{ duration: 0.8, ease: 'easeInOut' }}
      style={{
        position: 'absolute',
        bottom: 0,
        right: '10%',
        width: '600px',
        height: '500px',
        background: 'radial-gradient(ellipse at center, rgba(225,29,72,0.07) 0%, transparent 70%)',
      }}
    />
    <div
      style={{
        position: 'absolute',
        inset: 0,
        opacity: 0.025,
        backgroundImage: `url("data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='noise'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23noise)' opacity='1'/%3E%3C/svg%3E")`,
        backgroundRepeat: 'repeat',
        backgroundSize: '200px 200px',
      }}
    />
    <div
      style={{
        position: 'absolute',
        top: 0,
        left: 0,
        right: 0,
        height: '1px',
        background: 'rgba(255,255,255,0.04)',
      }}
    />
  </div>
);

/* ─── Main ScrollTimeline ────────────────────────────────────────── */
export const ScrollTimeline: React.FC = () => {
  const sectionRef = useRef<HTMLDivElement>(null);
  const stickyRef = useRef<HTMLDivElement>(null);
  const [activeStep, setActiveStep] = useState(0);
  const [prevStep, setPrevStep] = useState(0);

  useLayoutEffect(() => {
    const ctx = gsap.context(() => {
      ScrollTrigger.create({
        trigger: sectionRef.current,
        start: 'top top',
        end: 'bottom bottom',
        pin: stickyRef.current,
        scrub: true,
        snap: {
          snapTo: 1 / (steps.length - 1),
          duration: { min: 0.4, max: 0.9 },
          ease: 'power2.inOut',
        },
        onUpdate: (self) => {
          const raw = self.progress * (steps.length - 1);
          const idx = Math.min(Math.round(raw), steps.length - 1);
          setActiveStep((prev) => {
            if (prev !== idx) setPrevStep(prev);
            return idx;
          });
        },
      });
    }, sectionRef);
    return () => ctx.revert();
  }, []);

  const step = steps[activeStep];
  const dir = activeStep >= prevStep ? 1 : -1;
  const progress = (activeStep / (steps.length - 1)) * 100;

  const scrollToStep = (i: number) => {
    if (!sectionRef.current) return;
    const ratio = i / (steps.length - 1);
    const top = sectionRef.current.offsetTop;
    const height = sectionRef.current.offsetHeight - window.innerHeight;
    window.scrollTo({ top: top + ratio * height, behavior: 'smooth' });
  };

  return (
    <div ref={sectionRef} style={{ height: `${steps.length * 100}vh` }}>
      <div ref={stickyRef} className="w-full h-screen overflow-hidden relative bg-transparent">
        <CinematicBg activeStep={activeStep} />

        {/* ══════════════════════════════════════════════════════════ */}
        {/* MOBILE layout  (< md = 768px)                             */}
        {/* Order: step dots → number+sub → heading → body → card → progress */}
        {/* ══════════════════════════════════════════════════════════ */}
        <div
          className="md:hidden relative z-10 h-full flex flex-col justify-between px-5"
          style={{ paddingTop: '88px', paddingBottom: '28px' }}
        >
          {/* ── Step dot navigation ─────────────────────────────── */}
          <div className="flex items-center justify-center gap-2 mb-4">
            {steps.map((s, i) => (
              <button
                key={s.num}
                onClick={() => scrollToStep(i)}
                className="focus:outline-none transition-all duration-400"
                style={{
                  height: 4,
                  borderRadius: 2,
                  flex: i === activeStep ? 3 : 1,
                  background: i === activeStep ? '#e11d48' : 'rgba(255,255,255,0.12)',
                  transition: 'flex 0.4s ease, background 0.3s ease',
                  boxShadow: i === activeStep ? '0 0 8px rgba(225,29,72,0.5)' : 'none',
                }}
              />
            ))}
          </div>

          {/* ── Animated step content block ─────────────────────── */}
          <AnimatePresence mode="wait">
            <motion.div
              key={`mob-${activeStep}`}
              initial={{ opacity: 0, y: 28 * dir, scale: 0.97 }}
              animate={{ opacity: 1, y: 0, scale: 1 }}
              exit={{ opacity: 0, y: -20 * dir, scale: 0.98 }}
              transition={{ duration: 0.42, ease: [0.4, 0, 0.2, 1] }}
              className="flex flex-col items-center justify-center text-center flex-1 min-h-0"
            >
              {/* Step number + sub-label */}
              <div className="flex items-center justify-center gap-3 mb-2.5 w-full">
                <span
                  className="text-[11px] font-black tracking-[0.25em] uppercase"
                  style={{ color: '#e11d48', fontFamily: 'Inter, monospace' }}
                >
                  {step.num}
                </span>
                <div className="h-px w-6" style={{ background: 'rgba(225,29,72,0.4)' }} />
                <span
                  className="text-[10px] font-semibold tracking-[0.2em] uppercase"
                  style={{ color: 'rgba(255,255,255,0.3)', fontFamily: 'Inter, sans-serif' }}
                >
                  {step.sub}
                </span>
              </div>

              {/* Heading */}
              <h2
                className="font-extrabold text-white text-center mb-2.5 leading-[1.08]"
                style={{
                  fontFamily: 'Plus Jakarta Sans, sans-serif',
                  fontSize: 'clamp(2rem, 9vw, 2.5rem)',
                  letterSpacing: '-0.03em',
                  whiteSpace: 'pre-line',
                }}
              >
                {step.heading}
              </h2>

              {/* Body */}
              <p
                className="mb-4 text-center mx-auto max-w-[420px] leading-[1.65]"
                style={{
                  fontFamily: 'Inter, sans-serif',
                  fontSize: 'clamp(14px, 4vw, 16px)',
                  color: 'rgba(255,255,255,0.42)',
                  fontWeight: 400,
                }}
              >
                {step.body}
              </p>

              {/* Movie card — centered horizontally and vertically directly below body */}
              <div
                className="w-full flex items-center justify-center"
                style={{ minHeight: 220, maxHeight: 280 }}
              >
                {activeStep === 5 ? (
                  <div
                    className="rounded-2xl overflow-hidden w-[180px] mx-auto relative"
                    style={{
                      height: 260,
                      background: 'rgba(255,255,255,0.015)',
                      border: '1px solid rgba(255,255,255,0.05)',
                      backdropFilter: 'blur(2px)',
                      boxShadow: '0 0 0 1px rgba(225,29,72,0.08), 0 24px 60px rgba(0,0,0,0.5)',
                    }}
                  >
                    <FeedbackInteractiveCard />
                  </div>
                ) : (
                  <MobileCard posters={step.posters} stepIndex={activeStep} tag={step.tag} />
                )}
              </div>
            </motion.div>
          </AnimatePresence>

          {/* ── Progress bar ────────────────────────────────────── */}
          <div className="mt-5 flex items-center gap-3">
            <div
              className="flex-1 rounded-full overflow-hidden"
              style={{ height: 2, background: 'rgba(255,255,255,0.07)' }}
            >
              <motion.div
                className="h-full rounded-full"
                style={{ background: '#e11d48' }}
                animate={{ width: `${progress}%` }}
                transition={{ duration: 0.5, ease: 'easeInOut' }}
              />
            </div>
            <span
              className="shrink-0 text-[10px] font-medium tracking-widest"
              style={{ fontFamily: 'Inter, monospace', color: 'rgba(255,255,255,0.25)' }}
            >
              <span style={{ color: '#e11d48' }}>{String(activeStep + 1).padStart(2, '0')}</span>
              <span className="mx-1" style={{ color: 'rgba(255,255,255,0.15)' }}>
                /
              </span>
              {String(steps.length).padStart(2, '0')}
            </span>
          </div>
        </div>

        {/* ══════════════════════════════════════════════════════════ */}
        {/* DESKTOP layout  (≥ md = 768px)                            */}
        {/* ══════════════════════════════════════════════════════════ */}
        <div className="hidden md:flex relative z-10 h-full flex-row max-w-[1280px] mx-auto px-8 lg:px-14">
          {/* Left: numbered list */}
          <div className="w-[44%] flex flex-col justify-center py-0">
            <div
              className="flex items-center gap-3 mb-8 lg:mb-10"
              style={{ fontFamily: 'Inter, sans-serif' }}
            >
              <div className="h-px w-5 bg-white/20" />
              <span className="text-[10px] font-medium tracking-[0.3em] text-white/25 uppercase">
                How it works
              </span>
            </div>

            <div className="flex flex-col gap-3 lg:gap-5">
              {steps.map((s, i) => (
                <div key={s.num}>
                  <LabelItem step={s} isActive={activeStep === i} onClick={() => scrollToStep(i)} />
                  {i < steps.length - 1 && (
                    <div
                      className="mt-3 lg:mt-5 h-px"
                      style={{ background: 'rgba(255,255,255,0.04)' }}
                    />
                  )}
                </div>
              ))}
            </div>

            <div className="mt-8 lg:mt-14 flex items-center gap-4">
              <div
                className="flex-1 h-px rounded-full overflow-hidden"
                style={{ background: 'rgba(255,255,255,0.06)' }}
              >
                <motion.div
                  className="h-full rounded-full"
                  style={{ background: '#e11d48' }}
                  animate={{ width: `${progress}%` }}
                  transition={{ duration: 0.55, ease: 'easeInOut' }}
                />
              </div>
              <span
                className="text-[10px] font-medium tracking-widest shrink-0"
                style={{ fontFamily: 'Inter, monospace', color: 'rgba(255,255,255,0.2)' }}
              >
                <AnimatePresence mode="wait">
                  <motion.span
                    key={activeStep}
                    initial={{ opacity: 0, y: 6 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, y: -6 }}
                    transition={{ duration: 0.2 }}
                    className="inline-block"
                    style={{ color: '#e11d48' }}
                  >
                    {String(activeStep + 1).padStart(2, '0')}
                  </motion.span>
                </AnimatePresence>
                <span className="mx-1.5" style={{ color: 'rgba(255,255,255,0.15)' }}>
                  /
                </span>
                {String(steps.length).padStart(2, '0')}
              </span>
            </div>
          </div>

          {/* Vertical divider */}
          <div
            className="w-px mx-12 lg:mx-16 self-stretch"
            style={{ background: 'rgba(255,255,255,0.04)' }}
          />

          {/* Right: content + visual */}
          <div className="flex-1 flex flex-col justify-center gap-6 lg:gap-10">
            <AnimatePresence mode="wait">
              <motion.div
                key={`txt-${activeStep}`}
                initial={{ opacity: 0, y: 20 * dir, filter: 'blur(6px)' }}
                animate={{ opacity: 1, y: 0, filter: 'blur(0px)' }}
                exit={{ opacity: 0, y: -14 * dir, filter: 'blur(4px)' }}
                transition={{ duration: 0.45, ease: [0.4, 0, 0.2, 1] }}
              >
                <div
                  className="inline-flex items-center gap-2 mb-4"
                  style={{ fontFamily: 'Inter, sans-serif' }}
                >
                  <motion.div
                    layoutId="accent-pill-desk"
                    className="h-px w-8 rounded-full"
                    style={{ background: '#e11d48' }}
                  />
                  <span
                    className="text-[10px] font-semibold tracking-[0.22em] uppercase"
                    style={{ color: '#e11d48', fontFamily: 'Inter, sans-serif' }}
                  >
                    {step.sub}
                  </span>
                </div>
                <h2
                  className="font-extrabold text-white leading-[1.08] mb-4"
                  style={{
                    fontFamily: 'Plus Jakarta Sans, sans-serif',
                    fontSize: 'clamp(1.8rem, 3.5vw, 3.25rem)',
                    letterSpacing: '-0.03em',
                    whiteSpace: 'pre-line',
                  }}
                >
                  {step.heading}
                </h2>
                <p
                  className="text-[14px] lg:text-[15px] leading-[1.75] max-w-[420px]"
                  style={{
                    fontFamily: 'Inter, sans-serif',
                    color: 'rgba(255,255,255,0.4)',
                    fontWeight: 400,
                  }}
                >
                  {step.body}
                </p>
              </motion.div>
            </AnimatePresence>

            <AnimatePresence mode="wait">
              <motion.div
                key={`vis-${activeStep}`}
                initial={{ opacity: 0, y: 16 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -10 }}
                transition={{ duration: 0.35 }}
                className="relative w-full shrink-0"
                style={{ height: 'clamp(180px, 26vh, 280px)' }}
              >
                <div
                  className="absolute inset-0 rounded-2xl overflow-hidden"
                  style={{
                    background: 'rgba(255,255,255,0.015)',
                    border: '1px solid rgba(255,255,255,0.05)',
                    backdropFilter: 'blur(2px)',
                    boxShadow: '0 0 0 1px rgba(225,29,72,0.08), 0 24px 60px rgba(0,0,0,0.5)',
                  }}
                >
                  {activeStep === 5 ? (
                    <FeedbackInteractiveCard />
                  ) : (
                    <PosterStack posters={step.posters} stepIndex={activeStep} />
                  )}
                </div>
              </motion.div>
            </AnimatePresence>
          </div>
        </div>

        {/* Desktop scroll hint */}
        <div
          className="hidden md:flex absolute bottom-7 left-1/2 -translate-x-1/2 flex-col items-center gap-2 opacity-30"
          style={{ fontFamily: 'Inter, sans-serif' }}
        >
          <span className="text-[9px] tracking-[0.3em] text-white/50 uppercase">Scroll</span>
          <motion.div
            animate={{ y: [0, 5, 0] }}
            transition={{ repeat: Infinity, duration: 1.8, ease: 'easeInOut' }}
            className="w-px h-8 rounded-full"
            style={{ background: 'linear-gradient(to bottom, rgba(225,29,72,0.6), transparent)' }}
          />
        </div>
      </div>
    </div>
  );
};

export default ScrollTimeline;

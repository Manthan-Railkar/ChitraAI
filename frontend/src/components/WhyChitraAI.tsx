import React, { useEffect, useRef } from 'react';
import { gsap } from 'gsap';
import { ScrollTrigger } from 'gsap/ScrollTrigger';
import { Search, Sparkles, Brain, Zap } from 'lucide-react';
import './WhyChitraAI.css';

gsap.registerPlugin(ScrollTrigger);

interface FeatureItem {
  icon: React.ReactNode;
  title: string;
  desc: string;
  accent: string;
}
const features: FeatureItem[] = [
  {
    icon: <Search className="w-6 h-6 text-blue-400" />,
    title: 'Natural Conversations',
    desc: 'Describe a mood, memory, favourite movie, scene, or even a vague feeling. ChitraAI understands your intent and helps you discover the right films without relying on rigid filters.',
    accent: 'rgba(59, 130, 246, 0.15)',
  },
  {
    icon: <Sparkles className="w-6 h-6 text-indigo-400" />,
    title: 'Personalised Recommendations',
    desc: 'Every recommendation is tailored to your unique taste by combining your preferences, favourites, and search context, creating suggestions that become smarter with every interaction.',
    accent: 'rgba(99, 102, 241, 0.15)',
  },
  {
    icon: <Brain className="w-6 h-6 text-purple-400" />,
    title: 'Rich Movie Discovery',
    desc: 'Explore a vast collection of films enriched with detailed information, ratings, genres, languages, and cinematic insights to uncover both hidden gems and timeless classics.',
    accent: 'rgba(168, 85, 247, 0.15)',
  },
  {
    icon: <Zap className="w-6 h-6 text-pink-400" />,
    title: 'Fast & Transparent Results',
    desc: 'Receive personalised recommendations in seconds, complete with simple explanations that clearly show why each movie matches your preferences and search.',
    accent: 'rgba(236, 72, 153, 0.15)',
  },
];

export const WhyChitraAI: React.FC = () => {
  const containerRef = useRef<HTMLDivElement>(null);
  const headingRef = useRef<HTMLHeadingElement>(null);
  const subtitleRef = useRef<HTMLParagraphElement>(null);
  const cardsContainerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const ctx = gsap.context(() => {
      // Split heading into words and animate staggered
      const words = headingRef.current?.querySelectorAll('.word');
      if (words) {
        gsap.fromTo(
          words,
          {
            y: 40,
            opacity: 0,
            filter: 'blur(8px)',
          },
          {
            y: 0,
            opacity: 1,
            filter: 'blur(0px)',
            duration: 0.85,
            stagger: 0.08,
            ease: 'power3.out',
            scrollTrigger: {
              trigger: headingRef.current,
              start: 'top 85%',
            },
          }
        );
      }

      // Subtitle fade
      if (subtitleRef.current) {
        gsap.fromTo(
          subtitleRef.current,
          {
            y: 20,
            opacity: 0,
          },
          {
            y: 0,
            opacity: 1,
            duration: 0.8,
            delay: 0.35,
            ease: 'power3.out',
            scrollTrigger: {
              trigger: headingRef.current,
              start: 'top 85%',
            },
          }
        );
      }

      // Cards staggered entry
      const cards = cardsContainerRef.current?.querySelectorAll('.feature-card-wrapper');
      if (cards) {
        gsap.fromTo(
          cards,
          {
            y: 60,
            opacity: 0,
            rotate: 1.5,
          },
          {
            y: 0,
            opacity: 1,
            rotate: 0,
            duration: 1.0,
            stagger: 0.12,
            ease: 'power4.out',
            scrollTrigger: {
              trigger: cardsContainerRef.current,
              start: 'top 80%',
            },
          }
        );
      }

      // Subtle parallax effect on scroll trigger elements
      gsap.to(headingRef.current, {
        y: -30,
        ease: 'none',
        scrollTrigger: {
          trigger: containerRef.current,
          start: 'top bottom',
          end: 'bottom top',
          scrub: true,
        },
      });

      gsap.to(subtitleRef.current, {
        y: -15,
        ease: 'none',
        scrollTrigger: {
          trigger: containerRef.current,
          start: 'top bottom',
          end: 'bottom top',
          scrub: true,
        },
      });

      gsap.to(cardsContainerRef.current, {
        y: 15,
        ease: 'none',
        scrollTrigger: {
          trigger: containerRef.current,
          start: 'top bottom',
          end: 'bottom top',
          scrub: true,
        },
      });
    }, containerRef);

    return () => ctx.revert();
  }, []);

  return (
    <div
      ref={containerRef}
      className="relative w-full overflow-hidden bg-transparent py-24 sm:py-32 lg:py-40"
    >
      {/* Background ambient lighting */}
      <div className="absolute inset-0 pointer-events-none">
        <div className="absolute top-1/4 left-1/4 w-[45vw] h-[45vw] rounded-full bg-[#e11d48]/4 blur-[130px] floating-glow-1" />
        <div className="absolute bottom-1/4 right-1/4 w-[35vw] h-[35vw] rounded-full bg-[#e11d48]/3 blur-[110px] floating-glow-2" />
        {/* Central red glow behind text */}
        <div
          className="absolute inset-0"
          style={{
            background: 'radial-gradient(circle at 50% 40%, rgba(225,29,72,0.05), transparent 70%)',
          }}
        />
        {/* Horizontal separator line */}
        <div className="absolute top-0 inset-x-0 h-px bg-white/[0.04]" />
      </div>

      <div className="relative z-10 max-w-[1280px] mx-auto px-6 sm:px-10 lg:px-14 flex flex-col items-center">
        {/* Eyebrow */}
        <div className="flex items-center gap-3 mb-6" style={{ fontFamily: 'Inter, sans-serif' }}>
          <div className="h-px w-6 bg-indigo-500/60" />
          <span className="text-[10px] font-semibold tracking-[0.25em] uppercase text-white/30">
            Next-gen Discovery
          </span>
          <div className="h-px w-6 bg-indigo-500/60" />
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
          <span className="inline-block mr-3 word">Why</span>
          <span className="inline-block mr-3 word">Choose</span>
          <span className="inline-block word">
            <span className="bg-clip-text text-transparent bg-gradient-to-r from-rose-500 via-[#e11d48] to-purple-500 font-extrabold animate-gradient-text drop-shadow-[0_0_20px_rgba(225,29,72,0.35)]">
              ChitraAI?
            </span>
          </span>
        </h2>

        {/* Subtitle */}
        <p
          ref={subtitleRef}
          className="text-center text-white/40 text-base sm:text-lg max-w-[580px] mb-16 leading-[1.6]"
          style={{ fontFamily: 'Inter, sans-serif' }}
        >
          Designed for cinema lovers who demand a smarter, more aesthetic, and context-aware
          recommendation platform.
        </p>

        {/* Feature Grid */}
        <div
          ref={cardsContainerRef}
          className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 w-full"
        >
          {features.map((item, index) => (
            <div key={index} className="feature-card-wrapper">
              {/* Spinning border outline glow */}
              <div className="feature-card-glow" />

              {/* Card Inner Content */}
              <div className="feature-card-inner">
                {/* Icon Container */}
                <div
                  className="icon-container w-12 h-12 rounded-xl flex items-center justify-center mb-6 border border-white/[0.06]"
                  style={{
                    background: `linear-gradient(135deg, ${item.accent}, rgba(255,255,255,0.01))`,
                    boxShadow: 'inset 0 1px 1px rgba(255,255,255,0.05)',
                  }}
                >
                  {item.icon}
                </div>

                {/* Title */}
                <h3
                  className="text-white text-lg font-bold tracking-tight mb-3"
                  style={{ fontFamily: 'Plus Jakarta Sans, sans-serif' }}
                >
                  {item.title}
                </h3>

                {/* Description */}
                <p
                  className="text-white/40 text-sm leading-[1.7]"
                  style={{ fontFamily: 'Inter, sans-serif' }}
                >
                  {item.desc}
                </p>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

export default WhyChitraAI;

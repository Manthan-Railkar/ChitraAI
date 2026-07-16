import React, { useState, useEffect } from 'react';
import { Outlet, Link, useLocation } from 'react-router-dom';
import { LightRays } from '@/components/reactbits/LightRays';
import { Footer } from '@/components/Footer';
import { Moon, ArrowUpRight, Menu, X } from 'lucide-react';
import { AnimatePresence, motion } from 'framer-motion';
import Lenis from 'lenis';
import { gsap } from 'gsap';
import { ScrollTrigger } from 'gsap/ScrollTrigger';
import { useAuth } from '@/contexts/AuthContext';

gsap.registerPlugin(ScrollTrigger);

export const AppShell: React.FC = () => {
  const location = useLocation();
  const [isScrolled, setIsScrolled] = useState(false);
  const [isMenuOpen, setIsMenuOpen] = useState(false);
  const { user, profile } = useAuth();

  useEffect(() => {
    // Lenis smooth scroll + GSAP ScrollTrigger integration
    const lenis = new Lenis({ lerp: 0.08, smoothWheel: true });
    lenis.on('scroll', ScrollTrigger.update);
    gsap.ticker.add((time) => lenis.raf(time * 1000));
    gsap.ticker.lagSmoothing(0);

    const handleScroll = () => {
      if (window.scrollY > 15) setIsScrolled(true);
      else setIsScrolled(false);
    };
    window.addEventListener('scroll', handleScroll);
    return () => {
      lenis.destroy();
      window.removeEventListener('scroll', handleScroll);
    };
  }, []);

  // Close mobile menu on route change
  useEffect(() => {
    const timer = setTimeout(() => {
      setIsMenuOpen(false);
    }, 0);
    return () => clearTimeout(timer);
  }, [location.pathname]);

  const navItems = [
    { path: '/', label: 'Home' },
    { path: '/search', label: 'Product' },
    { path: '/favorites', label: 'Favorites' },
  ];

  return (
    <div className="min-h-screen bg-black text-foreground flex flex-col font-sans select-none relative overflow-x-hidden">
      {/* WebGL Light Rays background covering the entire screen with exact user props */}
      {location.pathname !== '/search' && location.pathname !== '/favorites' && (
        <div className="fixed inset-0 pointer-events-none overflow-hidden select-none z-0">
          <LightRays
            raysOrigin="top-center"
            raysColor="#ffffff"
            raysSpeed={1}
            lightSpread={0.5}
            rayLength={3}
            followMouse={true}
            mouseInfluence={0.1}
            noiseAmount={0}
            distortion={0}
            className="custom-rays"
            pulsating={false}
            fadeDistance={1}
            saturation={1}
          />
        </div>
      )}

      {/* Floating Header */}
      <div className="fixed top-0 inset-x-0 z-50 w-full px-4 sm:px-6 lg:px-10 pt-4 pointer-events-none">
        <header
          className={`max-w-[1280px] mx-auto w-full px-5 sm:px-6 py-3 rounded-full flex items-center justify-between pointer-events-auto border transition-all duration-500 ${
            isScrolled || isMenuOpen
              ? 'bg-black/60 backdrop-blur-xl border-white/[0.08] shadow-[0_8px_32px_0_rgba(0,0,0,0.6)] bg-gradient-to-r from-white/[0.04] to-white/[0.01]'
              : 'bg-transparent border-transparent'
          }`}
        >
          {/* Brand Logo */}
          <div className="flex-1 flex justify-start z-55 pl-2">
            <Link to="/" className="flex items-center select-none">
              <span className="text-base font-extrabold tracking-wider uppercase text-white">
                Chitra<span className="text-primary">AI</span>
              </span>
            </Link>
          </div>

          {/* Navigation Tabs (Desktop) */}
          <nav className="hidden md:flex items-center justify-center gap-7 flex-1">
            {navItems.map((item) => {
              const isActive = location.pathname === item.path;
              return (
                <Link
                  key={item.path}
                  to={item.path}
                  className={`text-[12px] font-semibold uppercase tracking-[0.15em] transition-colors duration-200 cursor-pointer ${
                    isActive ? 'text-white' : 'text-white/40 hover:text-white/80'
                  }`}
                >
                  {item.label}
                </Link>
              );
            })}
          </nav>

          {/* Actions (Desktop & Tablet) */}
          <div className="flex-1 flex items-center justify-end gap-2 sm:gap-4 z-55 pr-1">
            {/* Auth State (Desktop & Tablet) */}
            {user ? (
              <Link
                to="/profile"
                className="hidden sm:inline-flex items-center gap-2 cursor-pointer group"
              >
                <div className="w-7 h-7 rounded-full bg-gradient-to-tr from-rose-500 via-purple-600 to-blue-500 flex items-center justify-center text-white text-[10px] font-black uppercase shadow-sm group-hover:shadow-purple-500/20 transition-shadow overflow-hidden">
                  {profile?.avatar_url ? (
                    <img src={profile.avatar_url} alt="" className="w-full h-full object-cover" />
                  ) : (
                    (profile?.display_name?.[0] || user.email?.[0] || 'U').toUpperCase()
                  )}
                </div>
              </Link>
            ) : (
              <Link
                to="/profile"
                className="hidden sm:inline-block text-[12px] font-semibold uppercase tracking-[0.15em] text-white/40 hover:text-white/80 transition-colors duration-200 cursor-pointer"
              >
                Sign In
              </Link>
            )}

            {/* Pill button (Desktop & Tablet) */}
            <Link
              to="/search"
              className="hidden sm:inline-flex items-center gap-1 px-5 py-2.5 bg-white text-black text-[11px] font-bold uppercase tracking-[0.15em] rounded-full hover:bg-white/90 transition-all duration-200 cursor-pointer"
            >
              Get Started
              <ArrowUpRight className="w-3.5 h-3.5 stroke-[2.5]" />
            </Link>

            {/* Mobile Hamburger Button */}
            <button
              onClick={() => setIsMenuOpen(!isMenuOpen)}
              className="md:hidden w-8 h-8 rounded-full border border-white/10 hover:border-white/20 flex items-center justify-center text-muted-foreground hover:text-white transition-colors cursor-pointer bg-white/5"
            >
              {isMenuOpen ? <X className="w-4.5 h-4.5" /> : <Menu className="w-4.5 h-4.5" />}
            </button>
          </div>
        </header>

        {/* Mobile Menu Panel */}
        <AnimatePresence>
          {isMenuOpen && (
            <motion.div
              initial={{ opacity: 0, y: -15, height: 0 }}
              animate={{ opacity: 1, y: 0, height: 'auto' }}
              exit={{ opacity: 0, y: -15, height: 0 }}
              transition={{ duration: 0.25, ease: 'easeInOut' }}
              className="md:hidden mt-2 max-w-7xl mx-auto w-full px-6 py-6 rounded-3xl border border-white/10 bg-black/80 backdrop-blur-xl pointer-events-auto flex flex-col gap-6 shadow-[0_12px_40px_0_rgba(0,0,0,0.8)] bg-gradient-to-b from-white/5 to-transparent overflow-hidden"
            >
              <div className="flex flex-col gap-4">
                {navItems.map((item) => {
                  const isActive = location.pathname === item.path;
                  return (
                    <Link
                      key={item.path}
                      to={item.path}
                      className={`text-xs font-bold uppercase tracking-widest transition-colors duration-200 py-1.5 border-b border-white/5 cursor-pointer ${
                        isActive
                          ? 'text-white border-white/10'
                          : 'text-muted-foreground hover:text-white'
                      }`}
                    >
                      {item.label}
                    </Link>
                  );
                })}
              </div>

              {/* Extra Mobile Actions (Sign In / Get Started) */}
              <div className="flex flex-col sm:hidden gap-3 pt-2">
                <Link
                  to="/profile"
                  className="w-full text-center py-2.5 rounded-full border border-white/10 text-xs font-bold uppercase tracking-widest text-muted-foreground hover:text-white transition-colors cursor-pointer"
                >
                  {user ? (profile?.display_name || 'My Account') : 'Sign In'}
                </Link>
                <Link
                  to="/search"
                  className="w-full inline-flex items-center justify-center gap-0.5 py-2.5 bg-white text-black text-xs font-bold uppercase tracking-widest rounded-full hover:bg-white/90 transition-all duration-200 cursor-pointer"
                >
                  Get Started
                  <ArrowUpRight className="w-3.5 h-3.5 stroke-[2.5] -mr-0.5" />
                </Link>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>

      {/* Main viewport */}
      <main className="flex-1 relative z-10 w-full h-full pt-28">
        <Outlet />
      </main>

      {/* Footer */}
      {location.pathname === '/' && <Footer />}
    </div>
  );
};

export default AppShell;

import React from 'react';
import { Link } from 'react-router-dom';
import { Heart, Mail } from 'lucide-react';

export const Footer: React.FC = () => {
  return (
    <footer className="relative w-full border-t border-white/[0.04] bg-black/30 backdrop-blur-lg pt-16 pb-8 z-20 overflow-hidden">
      {/* Background ambient lighting/glow specifically for the footer */}
      <div className="absolute inset-0 pointer-events-none z-0">
        <div className="absolute bottom-0 left-1/2 -translate-x-1/2 w-[60vw] h-[180px] rounded-full bg-rose-500/5 blur-[100px] pointer-events-none" />
      </div>

      <div className="relative z-10 max-w-[1280px] mx-auto px-6 sm:px-10 lg:px-14">
        {/* Footer Top Grid */}
        <div className="grid grid-cols-2 md:grid-cols-5 gap-10 md:gap-14 pb-12 mb-12 border-b border-white/[0.04]">
          {/* Column 1: Info */}
          <div className="col-span-2 flex flex-col items-start gap-4">
            <Link to="/" className="flex items-center gap-2 select-none cursor-pointer">
              <span className="font-sans font-black tracking-tight text-white text-lg uppercase">
                CHITRA<span className="text-rose-500">AI</span>
              </span>
            </Link>
            <p
              className="text-[13px] leading-[1.6] text-white/40 max-w-[320px]"
              style={{ fontFamily: 'Inter, sans-serif' }}
            >
              Discover the stories that move you. ChitraAI leverages natural language semantic
              embedding matching to serve next-generation movie discovery.
            </p>
            {/* Social Links */}
            <div className="flex items-center gap-3.5 mt-2">
              <a
                href="https://github.com"
                target="_blank"
                rel="noreferrer"
                className="w-8 h-8 rounded-full border border-white/5 hover:border-white/20 bg-white/[0.02] hover:bg-white/5 text-white/55 hover:text-white flex items-center justify-center transition-all cursor-pointer"
              >
                <svg className="w-4 h-4 fill-current" viewBox="0 0 24 24">
                  <path d="M12 2C6.477 2 2 6.477 2 12c0 4.42 2.865 8.166 6.839 9.489.5.092.682-.217.682-.482 0-.237-.008-.866-.013-1.7-2.782.603-3.369-1.34-3.369-1.34-.454-1.156-1.11-1.462-1.11-1.462-.908-.62.069-.608.069-.608 1.003.07 1.531 1.03 1.531 1.03.892 1.529 2.341 1.087 2.91.831.092-.646.35-1.086.636-1.336-2.22-.253-4.555-1.11-4.555-4.943 0-1.091.39-1.984 1.029-2.683-.103-.253-.446-1.27.098-2.647 0 0 .84-.269 2.75 1.025A9.564 9.564 0 0112 6.844c.85.004 1.705.115 2.504.337 1.909-1.294 2.747-1.025 2.747-1.025.546 1.377.203 2.394.1 2.647.64.699 1.028 1.592 1.028 2.683 0 3.842-2.339 4.687-4.566 4.935.359.309.678.919.678 1.852 0 1.336-.012 2.415-.012 2.743 0 .267.18.579.688.481C19.137 20.162 22 16.418 22 12c0-5.523-4.477-10-10-10z" />
                </svg>
              </a>
              <a
                href="https://twitter.com"
                target="_blank"
                rel="noreferrer"
                className="w-8 h-8 rounded-full border border-white/5 hover:border-white/20 bg-white/[0.02] hover:bg-white/5 text-white/55 hover:text-white flex items-center justify-center transition-all cursor-pointer"
              >
                <svg className="w-3.5 h-3.5 fill-current" viewBox="0 0 24 24">
                  <path d="M18.244 2.25h3.308l-7.227 8.26 8.502 11.24H16.17l-5.214-6.817L4.99 21.75H1.68l7.73-8.835L1.254 2.25H8.08l4.713 6.231zm-1.161 17.52h1.833L7.084 4.126H5.117z" />
                </svg>
              </a>
              <a
                href="https://linkedin.com"
                target="_blank"
                rel="noreferrer"
                className="w-8 h-8 rounded-full border border-white/5 hover:border-white/20 bg-white/[0.02] hover:bg-white/5 text-white/55 hover:text-white flex items-center justify-center transition-all cursor-pointer"
              >
                <svg className="w-4 h-4 fill-current" viewBox="0 0 24 24">
                  <path d="M19 0h-14c-2.761 0-5 2.239-5 5v14c0 2.761 2.239 5 5 5h14c2.762 0 5-2.239 5-5v-14c0-2.761-2.238-5-5-5zm-11 19h-3v-11h3v11zm-1.5-12.268c-.966 0-1.75-.779-1.75-1.75s.784-1.75 1.75-1.75 1.75.779 1.75 1.75-.784 1.75-1.75 1.75zm13.5 12.268h-3v-5.604c0-3.368-4-3.113-4 0v5.604h-3v-11h3v1.765c1.396-2.586 7-2.777 7 2.476v6.759z" />
                </svg>
              </a>
            </div>
          </div>

          {/* Column 2: Product */}
          <div className="flex flex-col items-start gap-3.5">
            <span
              className="text-[10px] font-bold uppercase tracking-[0.2em] text-white/80"
              style={{ fontFamily: 'Plus Jakarta Sans, sans-serif' }}
            >
              Product
            </span>
            <ul className="flex flex-col items-start gap-2.5">
              <li>
                <Link
                  to="/search"
                  className="text-[13px] text-white/45 hover:text-white transition-colors cursor-pointer"
                  style={{ fontFamily: 'Inter, sans-serif' }}
                >
                  Product
                </Link>
              </li>
              <li>
                <Link
                  to="/dashboard"
                  className="text-[13px] text-white/45 hover:text-white transition-colors cursor-pointer"
                  style={{ fontFamily: 'Inter, sans-serif' }}
                >
                  Dashboard
                </Link>
              </li>
            </ul>
          </div>

          {/* Column 3: Resources */}
          <div className="flex flex-col items-start gap-3.5">
            <span
              className="text-[10px] font-bold uppercase tracking-[0.2em] text-white/80"
              style={{ fontFamily: 'Plus Jakarta Sans, sans-serif' }}
            >
              Resources
            </span>
            <ul className="flex flex-col items-start gap-2.5">
              <li>
                <a
                  href="#"
                  className="text-[13px] text-white/45 hover:text-white transition-colors cursor-pointer"
                  style={{ fontFamily: 'Inter, sans-serif' }}
                >
                  Documentation
                </a>
              </li>
              <li>
                <a
                  href="#"
                  className="text-[13px] text-white/45 hover:text-white transition-colors cursor-pointer"
                  style={{ fontFamily: 'Inter, sans-serif' }}
                >
                  API Status
                </a>
              </li>
              <li>
                <a
                  href="#"
                  className="text-[13px] text-white/45 hover:text-white transition-colors cursor-pointer"
                  style={{ fontFamily: 'Inter, sans-serif' }}
                >
                  Release Notes
                </a>
              </li>
            </ul>
          </div>

          {/* Column 4: Contact & Legal */}
          <div className="flex flex-col items-start gap-3.5">
            <span
              className="text-[10px] font-bold uppercase tracking-[0.2em] text-white/80"
              style={{ fontFamily: 'Plus Jakarta Sans, sans-serif' }}
            >
              Legal
            </span>
            <ul className="flex flex-col items-start gap-2.5">
              <li>
                <a
                  href="#"
                  className="text-[13px] text-white/45 hover:text-white transition-colors cursor-pointer"
                  style={{ fontFamily: 'Inter, sans-serif' }}
                >
                  Privacy Policy
                </a>
              </li>
              <li>
                <a
                  href="#"
                  className="text-[13px] text-white/45 hover:text-white transition-colors cursor-pointer"
                  style={{ fontFamily: 'Inter, sans-serif' }}
                >
                  Terms of Service
                </a>
              </li>
              <li>
                <a
                  href="#"
                  className="text-[13px] text-white/45 hover:text-white transition-colors cursor-pointer flex items-center gap-1.5"
                  style={{ fontFamily: 'Inter, sans-serif' }}
                >
                  <Mail className="w-3.5 h-3.5" />
                  Support
                </a>
              </li>
            </ul>
          </div>
        </div>

        {/* Footer Bottom Block */}
        <div className="flex flex-col sm:flex-row items-center justify-between gap-4">
          <span className="text-[11px] text-white/30" style={{ fontFamily: 'Inter, sans-serif' }}>
            © {new Date().getFullYear()} ChitraAI. All rights reserved.
          </span>
          <span
            className="text-[11px] text-white/30 flex items-center gap-1.5"
            style={{ fontFamily: 'Inter, sans-serif' }}
          >
            Made with <Heart className="w-3.5 h-3.5 text-rose-500 fill-current" /> by Antigravity AI
          </span>
        </div>
      </div>
    </footer>
  );
};

export default Footer;

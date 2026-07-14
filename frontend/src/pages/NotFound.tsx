import React from 'react';
import { Link } from 'react-router-dom';
import { Film, Home } from 'lucide-react';
import { motion } from 'framer-motion';

export const NotFound: React.FC = () => {
  return (
    <div className="min-h-[70vh] flex flex-col items-center justify-center text-center px-4 relative select-none">
      <div className="absolute inset-0 bg-[radial-gradient(circle_at_center,rgba(229,9,20,0.06),transparent_65%)] pointer-events-none" />

      <div className="relative z-10 max-w-lg w-full flex flex-col items-center gap-6">
        <motion.div
          initial={{ rotate: -8, scale: 0.9, opacity: 0 }}
          animate={{ rotate: 0, scale: 1, opacity: 1 }}
          transition={{ type: 'spring', stiffness: 200, damping: 15 }}
          className="relative"
        >
          <div className="text-9xl font-black tracking-widest text-primary/10 select-none">404</div>
          <Film className="w-16 h-16 text-primary absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 opacity-80" />
        </motion.div>

        <div className="space-y-2">
          <h1 className="text-3xl font-extrabold uppercase tracking-wider text-white">
            Scene Missing
          </h1>
          <p className="text-muted-foreground text-sm max-w-md mx-auto leading-relaxed">
            The script doesn't have a scene for this location. It was either left on the cutting
            room floor or edited out.
          </p>
        </div>

        <Link
          to="/"
          className="inline-flex items-center gap-2 px-6 py-3 bg-secondary border border-border hover:bg-muted text-white rounded-md transition-all duration-300 transform hover:scale-105 active:scale-95 text-sm font-semibold uppercase tracking-wider cursor-pointer"
        >
          <Home className="w-4 h-4" />
          Back to Lobby
        </Link>
      </div>
    </div>
  );
};

export default NotFound;

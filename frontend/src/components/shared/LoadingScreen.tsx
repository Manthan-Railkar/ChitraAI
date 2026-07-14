import React, { useEffect, useState } from 'react';
import { motion } from 'framer-motion';

export const LoadingScreen: React.FC = () => {
  const [dots, setDots] = useState('');

  useEffect(() => {
    const interval = setInterval(() => {
      setDots((prev) => (prev.length >= 3 ? '' : prev + '.'));
    }, 450);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="fixed inset-0 z-50 flex flex-col items-center justify-center bg-black select-none">
      {/* Light leak / radial spotlight backdrop */}
      <div className="absolute inset-0 bg-[radial-gradient(circle_at_center,rgba(229,9,20,0.15),transparent_60%)] pointer-events-none" />

      <div className="relative z-10 flex flex-col items-center gap-6">
        {/* Cinematic Pulse Logo */}
        <motion.div
          initial={{ scale: 0.85, opacity: 0 }}
          animate={{
            scale: [0.95, 1.05, 0.98, 1.02, 0.95],
            opacity: 1,
            filter: [
              'drop-shadow(0 0 12px rgba(229,9,20,0.15))',
              'drop-shadow(0 0 35px rgba(229,9,20,0.5))',
              'drop-shadow(0 0 18px rgba(229,9,20,0.25))',
              'drop-shadow(0 0 28px rgba(229,9,20,0.4))',
              'drop-shadow(0 0 12px rgba(229,9,20,0.15))',
            ],
          }}
          transition={{
            duration: 4,
            repeat: Infinity,
            ease: 'easeInOut',
          }}
          className="flex items-center gap-2"
        >
          <span className="text-4xl md:text-5xl font-black tracking-wider uppercase text-primary">
            Chitra<span className="text-white">AI</span>
          </span>
        </motion.div>

        {/* Progress Bar & Subtext */}
        <div className="flex flex-col items-center gap-2">
          <motion.p
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.4 }}
            className="text-xs font-semibold uppercase tracking-widest text-muted-foreground/80"
          >
            Configuring Projector{dots}
          </motion.p>

          <div className="w-48 h-[2px] bg-secondary overflow-hidden rounded-full mt-2">
            <motion.div
              initial={{ x: '-100%' }}
              animate={{ x: '100%' }}
              transition={{
                duration: 2.0,
                repeat: Infinity,
                ease: 'easeInOut',
              }}
              className="w-full h-full bg-primary"
            />
          </div>
        </div>
      </div>
    </div>
  );
};

export default LoadingScreen;

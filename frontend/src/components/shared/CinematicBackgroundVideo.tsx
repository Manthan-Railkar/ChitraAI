import React, { useRef, useState, useEffect } from 'react';
import backgroundVideo from '@/assets/videos/cinematic-background.mp4';
import { APP_CONFIG } from '@/config';

interface CinematicBackgroundVideoProps {
  poster?: string;
  overlayOpacity?: number;
  videoOpacity?: number;
  className?: string;
}

export const CinematicBackgroundVideo: React.FC<CinematicBackgroundVideoProps> = ({
  poster = APP_CONFIG.imageFallbacks.backdrop,
  overlayOpacity = 0.4,
  videoOpacity = 0.9,
  className = '',
}) => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const [hasError, setHasError] = useState(false);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    if (videoRef.current) {
      // Force play if browser autoplays gets blocked
      videoRef.current.play().catch((err) => {
        console.warn('Cinematic video autoplay failed, waiting for interaction:', err);
      });
    }
  }, []);

  const handleVideoError = () => {
    console.error('Failed to load cinematic background video, falling back to static poster.');
    setHasError(true);
    setIsLoading(false);
  };

  const handleVideoLoaded = () => {
    setIsLoading(false);
  };

  return (
    <div
      className={`absolute inset-0 w-full h-full overflow-hidden select-none bg-black ${className}`}
    >
      {/* Fallback Poster Image or Video Playback Error Fallback */}
      {(hasError || isLoading) && (
        <img
          src={poster}
          alt="Cinematic background fallback"
          className="absolute inset-0 w-full h-full object-cover transition-opacity duration-1000 opacity-60 z-0"
        />
      )}

      {/* Looping Cinematic Video */}
      {!hasError && (
        <video
          ref={videoRef}
          src={backgroundVideo}
          autoPlay
          muted
          loop
          playsInline
          preload="auto"
          onError={handleVideoError}
          onLoadedData={handleVideoLoaded}
          style={{ opacity: isLoading ? 0 : videoOpacity }}
          className="absolute inset-0 w-full h-full object-cover transition-opacity duration-1000 z-0"
        />
      )}

      {/* Cinematic Ambient Overlays */}
      {/* 1. Dark tone overlay */}
      <div
        className="absolute inset-0 bg-black pointer-events-none z-1"
        style={{ opacity: overlayOpacity }}
      />

      {/* 2. Soft radial vignette for framing */}
      <div className="absolute inset-0 bg-[radial-gradient(circle_at_center,rgba(0,0,0,0.15)_0%,rgba(0,0,0,0.8)_100%)] pointer-events-none z-2" />

      {/* 3. Deep bottom gradient for readability of UI overlays */}
      <div className="absolute inset-0 bg-gradient-to-t from-black via-black/40 to-transparent pointer-events-none z-2" />

      {/* GPU Accelerated Dynamic Film Grain */}
      <div className="absolute inset-0 pointer-events-none z-3 overflow-hidden opacity-[0.035] mix-blend-overlay">
        <svg className="w-full h-full">
          <filter id="noiseFilter">
            <feTurbulence
              type="fractalNoise"
              baseFrequency="0.75"
              numOctaves="3"
              stitchTiles="stitch"
            />
            <feColorMatrix type="matrix" values="0 0 0 0 0   0 0 0 0 0   0 0 0 0 0  0 0 0 0.8 0" />
          </filter>
          <rect width="100%" height="100%" filter="url(#noiseFilter)" />
        </svg>
      </div>
    </div>
  );
};

export default CinematicBackgroundVideo;

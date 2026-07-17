import React from 'react';
import { Check, X } from 'lucide-react';
import { useNavigate } from 'react-router-dom';

interface GuestLimitModalProps {
  onClose: () => void;
}

export const GuestLimitModal: React.FC<GuestLimitModalProps> = ({ onClose }) => {
  const navigate = useNavigate();

  const handleCreateAccount = () => {
    onClose();
    navigate('/profile');
  };

  return (
    <div className="fixed inset-0 z-[200] flex items-center justify-center p-4 bg-black/80 backdrop-blur-md animate-fade-in">
      <div
        role="dialog"
        aria-modal="true"
        aria-labelledby="guest-limit-title"
        className="relative w-full max-w-md rounded-3xl border border-white/[0.1] bg-zinc-950/95 p-6 sm:p-8 shadow-2xl"
      >
        <button
          onClick={onClose}
          aria-label="Close"
          className="absolute right-4 top-4 rounded-full border border-white/10 p-2 text-white/50 transition-colors hover:text-white hover:border-white/20 cursor-pointer"
        >
          <X className="w-4 h-4" />
        </button>

        <div className="flex flex-col gap-5">
          <div className="w-11 h-11 rounded-2xl border border-rose-500/20 bg-rose-500/10 flex items-center justify-center">
            <span className="text-lg">✦</span>
          </div>
          <div className="flex flex-col gap-2">
            <h2 id="guest-limit-title" className="text-xl font-black tracking-tight text-white">
              You&apos;ve reached your free search limit
            </h2>
            <p className="text-sm leading-relaxed text-white/55">
              Create a free ChitraAI account to continue discovering personalized AI-powered movie
              recommendations.
            </p>
          </div>
          <ul className="flex flex-col gap-2 text-sm text-white/70">
            {[
              'Unlimited AI searches',
              'Save favourite movies',
              'Personalized recommendations',
              'Access future premium features',
            ].map((benefit) => (
              <li key={benefit} className="flex items-center gap-2">
                <Check className="w-4 h-4 text-rose-400 shrink-0" />
                {benefit}
              </li>
            ))}
          </ul>
          <div className="flex flex-col-reverse sm:flex-row gap-3 pt-1">
            <button
              onClick={onClose}
              className="flex-1 rounded-full border border-white/10 px-4 py-3 text-xs font-bold uppercase tracking-wider text-white/70 transition-colors hover:border-white/20 hover:text-white cursor-pointer"
            >
              Maybe Later
            </button>
            <button
              onClick={handleCreateAccount}
              className="flex-1 rounded-full bg-white px-4 py-3 text-xs font-bold uppercase tracking-wider text-black transition-colors hover:bg-rose-500 hover:text-white cursor-pointer"
            >
              Create Free Account
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default GuestLimitModal;

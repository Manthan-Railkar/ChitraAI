import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '@/contexts/AuthContext';
import { Mail, Lock, User, LogOut, Trash2, Eye, EyeOff, Loader2, AlertTriangle, ArrowLeft, KeyRound } from 'lucide-react';
import { toast } from 'sonner';
import Beams from '@/components/reactbits/Beams';

type AuthView = 'welcome' | 'sign-in' | 'sign-up' | 'forgot-password';

export const Profile: React.FC = () => {
  const { user, profile, isLoading, signInWithEmail, signUpWithEmail, signInWithGoogle, signOut, deleteAccount } = useAuth();
  const navigate = useNavigate();

  const [view, setView] = useState<AuthView>('welcome');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [displayName, setDisplayName] = useState('');
  const [showPassword, setShowPassword] = useState(false);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [formError, setFormError] = useState<string | null>(null);
  const [showDeleteConfirm, setShowDeleteConfirm] = useState(false);

  if (isLoading) {
    return (
      <div className="fixed inset-0 bg-black flex items-center justify-center">
        <Loader2 className="w-8 h-8 text-rose-500 animate-spin" />
      </div>
    );
  }

  const resetForm = () => {
    setEmail('');
    setPassword('');
    setDisplayName('');
    setFormError(null);
    setShowPassword(false);
  };

  const handleGoogleSignIn = async () => {
    setFormError(null);
    const { error } = await signInWithGoogle();
    if (error) setFormError(error.message);
  };

  const handleEmailSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setFormError(null);
    setIsSubmitting(true);
    try {
      if (view === 'sign-in') {
        const { error } = await signInWithEmail(email, password);
        if (error) {
          setFormError(error.message);
        } else {
          toast.success('Welcome back!');
          navigate('/');
        }
      } else if (view === 'sign-up') {
        const { error } = await signUpWithEmail(email, password, displayName);
        if (error) {
          setFormError(error.message);
        } else {
          toast.success('Account created! Check your email for verification.');
          resetForm();
          setView('sign-in');
        }
      }
    } finally {
      setIsSubmitting(false);
    }
  };

  // ── Authenticated View ──
  if (user) {
    return (
      <div className="fixed inset-0 bg-black overflow-hidden">
        {/* Beams Background */}
        <div className="absolute inset-0 z-0 opacity-90">
          <Beams
            beamWidth={3}
            beamHeight={30}
            beamNumber={20}
            lightColor="#e11d48"
            speed={2}
            noiseIntensity={1.75}
            scale={0.2}
            rotation={30}
          />
        </div>

        {/* Back Button */}
        <button
          onClick={() => navigate('/')}
          className="absolute top-6 left-6 z-20 inline-flex items-center gap-2 text-[11px] font-bold text-white/40 hover:text-white uppercase tracking-widest transition-colors cursor-pointer"
        >
          <ArrowLeft className="w-3.5 h-3.5" />
          Back
        </button>

        {/* Profile Card */}
        <div className="relative z-10 flex items-center justify-center min-h-screen px-4 py-12">
          <div className="w-full max-w-[440px] rounded-3xl border border-white/[0.08] bg-[#0a0a0f]/80 backdrop-blur-2xl p-8 flex flex-col items-center gap-6 shadow-[0_24px_64px_rgba(0,0,0,0.5)]">
            {/* Avatar */}
            <div className="relative">
              <div className="w-20 h-20 rounded-full bg-gradient-to-tr from-rose-500 via-purple-600 to-blue-500 flex items-center justify-center text-white text-2xl font-black uppercase shadow-xl shadow-purple-500/20 overflow-hidden">
                {profile?.avatar_url ? (
                  <img src={profile.avatar_url} alt="Avatar" className="w-full h-full object-cover" />
                ) : (
                  (profile?.display_name?.[0] || user.email?.[0] || 'U').toUpperCase()
                )}
              </div>
              <div className="absolute -bottom-1 -right-1 w-6 h-6 rounded-full bg-emerald-500 border-[3px] border-[#0a0a0f] flex items-center justify-center">
                <div className="w-1.5 h-1.5 rounded-full bg-white" />
              </div>
            </div>

            {/* User Info */}
            <div className="flex flex-col items-center gap-1.5 text-center">
              <h2
                className="text-xl font-black text-white uppercase tracking-tight"
                style={{ fontFamily: 'Plus Jakarta Sans, sans-serif' }}
              >
                {profile?.display_name || 'ChitraAI User'}
              </h2>
              <p className="text-xs text-white/50 font-medium" style={{ fontFamily: 'Inter, sans-serif' }}>
                {user.email}
              </p>
              <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full border border-white/[0.08] bg-white/[0.02] text-[10px] font-bold text-white/40 uppercase tracking-widest mt-1">
                <div className="w-1.5 h-1.5 rounded-full bg-emerald-500" />
                Active Session
              </span>
            </div>

            {/* Account Details */}
            <div className="w-full border-t border-white/[0.06] pt-4 flex flex-col gap-3">
              <div className="flex items-center justify-between px-1">
                <span className="text-[10px] font-bold text-white/30 uppercase tracking-widest">Member Since</span>
                <span className="text-xs text-white/60 font-semibold">
                  {profile?.created_at
                    ? new Date(profile.created_at).toLocaleDateString('en-US', { month: 'long', year: 'numeric' })
                    : 'Unknown'}
                </span>
              </div>
              <div className="flex items-center justify-between px-1">
                <span className="text-[10px] font-bold text-white/30 uppercase tracking-widest">Auth Provider</span>
                <span className="text-xs text-white/60 font-semibold capitalize">
                  {user.app_metadata?.provider || 'Email'}
                </span>
              </div>
            </div>

            {/* Actions */}
            <div className="w-full flex flex-col gap-3 pt-1">
              <button
                onClick={async () => {
                  await signOut();
                  toast.success('Signed out successfully');
                }}
                className="w-full inline-flex items-center justify-center gap-2 py-3 rounded-xl border border-white/[0.08] hover:border-white/15 bg-white/[0.03] text-white text-[11px] font-bold uppercase tracking-wider hover:bg-white/[0.06] transition-all cursor-pointer"
              >
                <LogOut className="w-3.5 h-3.5" />
                Sign Out
              </button>

              {!showDeleteConfirm ? (
                <button
                  onClick={() => setShowDeleteConfirm(true)}
                  className="w-full inline-flex items-center justify-center gap-2 py-3 rounded-xl border border-rose-500/10 hover:border-rose-500/30 bg-transparent text-rose-400/60 hover:text-rose-400 text-[11px] font-bold uppercase tracking-wider hover:bg-rose-500/5 transition-all cursor-pointer"
                >
                  <Trash2 className="w-3.5 h-3.5" />
                  Delete Account
                </button>
              ) : (
                <div className="rounded-xl border border-rose-500/20 bg-rose-500/5 p-4 flex flex-col gap-3 animate-fade-in">
                  <div className="flex items-center gap-2 text-rose-400">
                    <AlertTriangle className="w-4 h-4" />
                    <span className="text-xs font-bold uppercase tracking-wider">Confirm Deletion</span>
                  </div>
                  <p className="text-[11px] text-white/50 leading-relaxed" style={{ fontFamily: 'Inter, sans-serif' }}>
                    This will permanently delete your profile, all favourites, and sign you out. This action cannot be undone.
                  </p>
                  <div className="flex gap-2">
                    <button
                      onClick={async () => {
                        const { error } = await deleteAccount();
                        if (error) toast.error(error);
                        else toast.success('Account deleted successfully');
                        setShowDeleteConfirm(false);
                      }}
                      className="flex-1 py-2 rounded-full bg-rose-500 text-white text-[10px] font-bold uppercase tracking-wider hover:bg-rose-600 transition-all cursor-pointer"
                    >
                      Yes, Delete
                    </button>
                    <button
                      onClick={() => setShowDeleteConfirm(false)}
                      className="flex-1 py-2 rounded-full border border-white/10 text-white/60 text-[10px] font-bold uppercase tracking-wider hover:bg-white/[0.04] transition-all cursor-pointer"
                    >
                      Cancel
                    </button>
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    );
  }

  // ── Unauthenticated: Multi-step Auth Flow ──
  return (
    <div className="fixed inset-0 bg-black overflow-hidden">
      {/* Beams Background */}
      <div className="absolute inset-0 z-0 opacity-80">
        <Beams
          beamWidth={3}
          beamHeight={30}
          beamNumber={20}
          lightColor="#e11d48"
          speed={2}
          noiseIntensity={1.75}
          scale={0.2}
          rotation={30}
        />
      </div>

      {/* Back to Home */}
      <button
        onClick={() => navigate('/')}
        className="absolute top-6 left-6 z-20 inline-flex items-center gap-2 text-[11px] font-bold text-white/40 hover:text-white uppercase tracking-widest transition-colors cursor-pointer"
      >
        <ArrowLeft className="w-3.5 h-3.5" />
        Back
      </button>

      {/* Auth Card */}
      <div className="relative z-10 flex items-center justify-center min-h-screen px-4 py-12">
        <div className="w-full max-w-[440px] flex flex-col items-center gap-7">

          {/* ═══ WELCOME VIEW ═══ */}
          {view === 'welcome' && (
            <div className="w-full rounded-3xl border border-white/[0.08] bg-[#0a0a0f]/80 backdrop-blur-2xl p-8 sm:p-10 flex flex-col items-center gap-7 shadow-[0_24px_64px_rgba(0,0,0,0.5)] animate-fade-in">

              {/* Heading */}
              <div className="flex flex-col items-center gap-2 text-center">
                <h1
                  className="text-2xl sm:text-[1.75rem] font-black text-white tracking-tight"
                  style={{ fontFamily: 'Plus Jakarta Sans, sans-serif' }}
                >
                  Welcome to Chitra<span className="text-rose-500">AI</span>
                </h1>
                <p className="text-xs text-white/40 max-w-[280px] leading-relaxed" style={{ fontFamily: 'Inter, sans-serif' }}>
                  Please enter your details to create your account
                </p>
              </div>

              {/* OAuth Buttons */}
              <div className="w-full flex flex-col gap-3">
                <button
                  onClick={handleGoogleSignIn}
                  className="w-full inline-flex items-center justify-center gap-3 py-3.5 rounded-xl border border-white/[0.08] bg-white/[0.03] text-white text-[12px] font-semibold hover:bg-white/[0.06] hover:border-white/15 transition-all cursor-pointer"
                  style={{ fontFamily: 'Inter, sans-serif' }}
                >
                  <svg className="w-4 h-4" viewBox="0 0 24 24">
                    <path fill="#4285F4" d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92a5.06 5.06 0 0 1-2.2 3.32v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.1z" />
                    <path fill="#34A853" d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z" />
                    <path fill="#FBBC05" d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z" />
                    <path fill="#EA4335" d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z" />
                  </svg>
                  Continue with Google
                </button>
              </div>

              {/* Divider */}
              <div className="w-full flex items-center gap-3">
                <div className="flex-1 h-px bg-white/[0.06]" />
                <span className="text-[10px] font-bold text-white/20 uppercase tracking-widest">or</span>
                <div className="flex-1 h-px bg-white/[0.06]" />
              </div>

              {/* Continue with Email */}
              <button
                onClick={() => { resetForm(); setView('sign-up'); }}
                className="w-full py-3.5 rounded-xl bg-white text-black text-[12px] font-bold hover:bg-white/90 transition-all cursor-pointer shadow-lg"
                style={{ fontFamily: 'Inter, sans-serif' }}
              >
                Continue with Email
              </button>

              {/* Toggle */}
              <p className="text-xs text-white/40" style={{ fontFamily: 'Inter, sans-serif' }}>
                Already have an account?{' '}
                <button
                  onClick={() => { resetForm(); setView('sign-in'); }}
                  className="text-white font-bold hover:text-rose-400 transition-colors cursor-pointer"
                >
                  Sign In
                </button>
              </p>

              {formError && (
                <div className="w-full rounded-xl border border-rose-500/20 bg-rose-500/5 px-4 py-2.5 text-[11px] text-rose-400 font-medium animate-fade-in" style={{ fontFamily: 'Inter, sans-serif' }}>
                  {formError}
                </div>
              )}
            </div>
          )}

          {/* ═══ SIGN IN VIEW ═══ */}
          {view === 'sign-in' && (
            <div className="w-full rounded-3xl border border-white/[0.08] bg-[#0a0a0f]/80 backdrop-blur-2xl p-8 sm:p-10 flex flex-col items-center gap-7 shadow-[0_24px_64px_rgba(0,0,0,0.5)] animate-fade-in">

              <div className="flex flex-col items-center gap-2 text-center">
                <h1
                  className="text-2xl sm:text-[1.75rem] font-black text-white tracking-tight"
                  style={{ fontFamily: 'Plus Jakarta Sans, sans-serif' }}
                >
                  Welcome to Chitra<span className="text-rose-500">AI</span>
                </h1>
                <p className="text-xs text-white/40 max-w-[280px] leading-relaxed" style={{ fontFamily: 'Inter, sans-serif' }}>
                  Sign in to access your favourites
                </p>
              </div>

              {/* OAuth */}
              <button
                onClick={handleGoogleSignIn}
                className="w-full inline-flex items-center justify-center gap-3 py-3.5 rounded-xl border border-white/[0.08] bg-white/[0.03] text-white text-[12px] font-semibold hover:bg-white/[0.06] hover:border-white/15 transition-all cursor-pointer"
                style={{ fontFamily: 'Inter, sans-serif' }}
              >
                <svg className="w-4 h-4" viewBox="0 0 24 24">
                  <path fill="#4285F4" d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92a5.06 5.06 0 0 1-2.2 3.32v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.1z" />
                  <path fill="#34A853" d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z" />
                  <path fill="#FBBC05" d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z" />
                  <path fill="#EA4335" d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z" />
                </svg>
                Continue with Google
              </button>

              {/* Divider */}
              <div className="w-full flex items-center gap-3">
                <div className="flex-1 h-px bg-white/[0.06]" />
                <span className="text-[10px] font-bold text-white/20 uppercase tracking-widest">or</span>
                <div className="flex-1 h-px bg-white/[0.06]" />
              </div>

              {/* Email / Password Form */}
              <form onSubmit={handleEmailSubmit} className="w-full flex flex-col gap-4">
                <div className="relative group">
                  <Mail className="absolute left-4 top-1/2 -translate-y-1/2 w-4 h-4 text-white/25 group-focus-within:text-white/50 transition-colors" />
                  <input
                    type="email"
                    value={email}
                    onChange={(e) => setEmail(e.target.value)}
                    placeholder="Email"
                    required
                    className="w-full bg-white/[0.03] border border-white/[0.08] rounded-xl pl-11 pr-4 py-3.5 text-[13px] text-white placeholder-white/25 outline-none focus:border-white/20 transition-colors"
                    style={{ fontFamily: 'Inter, sans-serif' }}
                  />
                </div>

                <div className="relative group">
                  <Lock className="absolute left-4 top-1/2 -translate-y-1/2 w-4 h-4 text-white/25 group-focus-within:text-white/50 transition-colors" />
                  <input
                    type={showPassword ? 'text' : 'password'}
                    value={password}
                    onChange={(e) => setPassword(e.target.value)}
                    placeholder="Password"
                    required
                    minLength={6}
                    className="w-full bg-white/[0.03] border border-white/[0.08] rounded-xl pl-11 pr-11 py-3.5 text-[13px] text-white placeholder-white/25 outline-none focus:border-white/20 transition-colors"
                    style={{ fontFamily: 'Inter, sans-serif' }}
                  />
                  <button
                    type="button"
                    onClick={() => setShowPassword(!showPassword)}
                    className="absolute right-4 top-1/2 -translate-y-1/2 text-white/25 hover:text-white/50 transition-colors cursor-pointer"
                  >
                    {showPassword ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                  </button>
                </div>

                {/* Forgot Password Link */}
                <div className="flex justify-end">
                  <button
                    type="button"
                    onClick={() => { setFormError(null); setView('forgot-password'); }}
                    className="text-[11px] text-white/30 hover:text-white/60 font-medium transition-colors cursor-pointer"
                    style={{ fontFamily: 'Inter, sans-serif' }}
                  >
                    Forgot password?
                  </button>
                </div>

                {formError && (
                  <div className="rounded-xl border border-rose-500/20 bg-rose-500/5 px-4 py-2.5 text-[11px] text-rose-400 font-medium animate-fade-in" style={{ fontFamily: 'Inter, sans-serif' }}>
                    {formError}
                  </div>
                )}

                <button
                  type="submit"
                  disabled={isSubmitting}
                  className="w-full py-3.5 rounded-xl bg-white text-black text-[12px] font-bold hover:bg-white/90 transition-all cursor-pointer shadow-lg disabled:opacity-50 disabled:pointer-events-none flex items-center justify-center gap-2"
                  style={{ fontFamily: 'Inter, sans-serif' }}
                >
                  {isSubmitting && <Loader2 className="w-3.5 h-3.5 animate-spin" />}
                  Sign In
                </button>
              </form>

              {/* Toggle */}
              <p className="text-xs text-white/40" style={{ fontFamily: 'Inter, sans-serif' }}>
                Don't have an account?{' '}
                <button
                  onClick={() => { resetForm(); setView('sign-up'); }}
                  className="text-white font-bold hover:text-rose-400 transition-colors cursor-pointer"
                >
                  Sign Up
                </button>
              </p>
            </div>
          )}

          {/* ═══ SIGN UP VIEW ═══ */}
          {view === 'sign-up' && (
            <div className="w-full rounded-3xl border border-white/[0.08] bg-[#0a0a0f]/80 backdrop-blur-2xl p-8 sm:p-10 flex flex-col items-center gap-7 shadow-[0_24px_64px_rgba(0,0,0,0.5)] animate-fade-in">

              <div className="flex flex-col items-center gap-2 text-center">
                <h1
                  className="text-2xl sm:text-[1.75rem] font-black text-white tracking-tight"
                  style={{ fontFamily: 'Plus Jakarta Sans, sans-serif' }}
                >
                  Welcome to Chitra<span className="text-rose-500">AI</span>
                </h1>
                <p className="text-xs text-white/40 max-w-[280px] leading-relaxed" style={{ fontFamily: 'Inter, sans-serif' }}>
                  Please enter your details to create your account
                </p>
              </div>

              {/* OAuth */}
              <button
                onClick={handleGoogleSignIn}
                className="w-full inline-flex items-center justify-center gap-3 py-3.5 rounded-xl border border-white/[0.08] bg-white/[0.03] text-white text-[12px] font-semibold hover:bg-white/[0.06] hover:border-white/15 transition-all cursor-pointer"
                style={{ fontFamily: 'Inter, sans-serif' }}
              >
                <svg className="w-4 h-4" viewBox="0 0 24 24">
                  <path fill="#4285F4" d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92a5.06 5.06 0 0 1-2.2 3.32v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.1z" />
                  <path fill="#34A853" d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z" />
                  <path fill="#FBBC05" d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z" />
                  <path fill="#EA4335" d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z" />
                </svg>
                Continue with Google
              </button>

              {/* Divider */}
              <div className="w-full flex items-center gap-3">
                <div className="flex-1 h-px bg-white/[0.06]" />
                <span className="text-[10px] font-bold text-white/20 uppercase tracking-widest">or</span>
                <div className="flex-1 h-px bg-white/[0.06]" />
              </div>

              {/* Sign Up Form */}
              <form onSubmit={handleEmailSubmit} className="w-full flex flex-col gap-4">
                <div className="relative group">
                  <User className="absolute left-4 top-1/2 -translate-y-1/2 w-4 h-4 text-white/25 group-focus-within:text-white/50 transition-colors" />
                  <input
                    type="text"
                    value={displayName}
                    onChange={(e) => setDisplayName(e.target.value)}
                    placeholder="Display Name"
                    className="w-full bg-white/[0.03] border border-white/[0.08] rounded-xl pl-11 pr-4 py-3.5 text-[13px] text-white placeholder-white/25 outline-none focus:border-white/20 transition-colors"
                    style={{ fontFamily: 'Inter, sans-serif' }}
                  />
                </div>

                <div className="relative group">
                  <Mail className="absolute left-4 top-1/2 -translate-y-1/2 w-4 h-4 text-white/25 group-focus-within:text-white/50 transition-colors" />
                  <input
                    type="email"
                    value={email}
                    onChange={(e) => setEmail(e.target.value)}
                    placeholder="Email"
                    required
                    className="w-full bg-white/[0.03] border border-white/[0.08] rounded-xl pl-11 pr-4 py-3.5 text-[13px] text-white placeholder-white/25 outline-none focus:border-white/20 transition-colors"
                    style={{ fontFamily: 'Inter, sans-serif' }}
                  />
                </div>

                <div className="relative group">
                  <Lock className="absolute left-4 top-1/2 -translate-y-1/2 w-4 h-4 text-white/25 group-focus-within:text-white/50 transition-colors" />
                  <input
                    type={showPassword ? 'text' : 'password'}
                    value={password}
                    onChange={(e) => setPassword(e.target.value)}
                    placeholder="Password"
                    required
                    minLength={6}
                    className="w-full bg-white/[0.03] border border-white/[0.08] rounded-xl pl-11 pr-11 py-3.5 text-[13px] text-white placeholder-white/25 outline-none focus:border-white/20 transition-colors"
                    style={{ fontFamily: 'Inter, sans-serif' }}
                  />
                  <button
                    type="button"
                    onClick={() => setShowPassword(!showPassword)}
                    className="absolute right-4 top-1/2 -translate-y-1/2 text-white/25 hover:text-white/50 transition-colors cursor-pointer"
                  >
                    {showPassword ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                  </button>
                </div>

                {formError && (
                  <div className="rounded-xl border border-rose-500/20 bg-rose-500/5 px-4 py-2.5 text-[11px] text-rose-400 font-medium animate-fade-in" style={{ fontFamily: 'Inter, sans-serif' }}>
                    {formError}
                  </div>
                )}

                <button
                  type="submit"
                  disabled={isSubmitting}
                  className="w-full py-3.5 rounded-xl bg-white text-black text-[12px] font-bold hover:bg-white/90 transition-all cursor-pointer shadow-lg disabled:opacity-50 disabled:pointer-events-none flex items-center justify-center gap-2"
                  style={{ fontFamily: 'Inter, sans-serif' }}
                >
                  {isSubmitting && <Loader2 className="w-3.5 h-3.5 animate-spin" />}
                  Create Account
                </button>
              </form>

              {/* Toggle */}
              <p className="text-xs text-white/40" style={{ fontFamily: 'Inter, sans-serif' }}>
                Already have an account?{' '}
                <button
                  onClick={() => { resetForm(); setView('sign-in'); }}
                  className="text-white font-bold hover:text-rose-400 transition-colors cursor-pointer"
                >
                  Sign In
                </button>
              </p>
            </div>
          )}

          {/* ═══ FORGOT PASSWORD VIEW ═══ */}
          {view === 'forgot-password' && (
            <div className="w-full rounded-3xl border border-white/[0.08] bg-[#0a0a0f]/80 backdrop-blur-2xl p-8 sm:p-10 flex flex-col items-center gap-7 shadow-[0_24px_64px_rgba(0,0,0,0.5)] animate-fade-in">
              {/* Icon */}
              <div className="w-14 h-14 rounded-2xl bg-white/[0.03] border border-white/[0.08] flex items-center justify-center">
                <KeyRound className="w-6 h-6 text-white/50" />
              </div>

              <div className="flex flex-col items-center gap-2 text-center">
                <h1
                  className="text-xl sm:text-2xl font-black text-white tracking-tight"
                  style={{ fontFamily: 'Plus Jakarta Sans, sans-serif' }}
                >
                  Reset Your Password
                </h1>
                <p className="text-xs text-white/40 max-w-[280px] leading-relaxed" style={{ fontFamily: 'Inter, sans-serif' }}>
                  Enter your email, and we'll send you instructions to create a new password.
                </p>
              </div>

              <form
                onSubmit={async (e) => {
                  e.preventDefault();
                  setFormError(null);
                  setIsSubmitting(true);
                  try {
                    const { default: supabase } = await import('@/lib/supabaseClient');
                    const { error } = await supabase.auth.resetPasswordForEmail(email, {
                      redirectTo: window.location.origin + '/profile',
                    });
                    if (error) {
                      setFormError(error.message);
                    } else {
                      toast.success('Password reset email sent! Check your inbox.');
                      resetForm();
                      setView('sign-in');
                    }
                  } finally {
                    setIsSubmitting(false);
                  }
                }}
                className="w-full flex flex-col gap-4"
              >
                <div className="relative group">
                  <Mail className="absolute left-4 top-1/2 -translate-y-1/2 w-4 h-4 text-white/25 group-focus-within:text-white/50 transition-colors" />
                  <input
                    type="email"
                    value={email}
                    onChange={(e) => setEmail(e.target.value)}
                    placeholder="Email"
                    required
                    className="w-full bg-white/[0.03] border border-white/[0.08] rounded-xl pl-11 pr-4 py-3.5 text-[13px] text-white placeholder-white/25 outline-none focus:border-white/20 transition-colors"
                    style={{ fontFamily: 'Inter, sans-serif' }}
                  />
                </div>

                {formError && (
                  <div className="rounded-xl border border-rose-500/20 bg-rose-500/5 px-4 py-2.5 text-[11px] text-rose-400 font-medium animate-fade-in" style={{ fontFamily: 'Inter, sans-serif' }}>
                    {formError}
                  </div>
                )}

                <button
                  type="submit"
                  disabled={isSubmitting}
                  className="w-full py-3.5 rounded-xl bg-white text-black text-[12px] font-bold hover:bg-white/90 transition-all cursor-pointer shadow-lg disabled:opacity-50 disabled:pointer-events-none flex items-center justify-center gap-2"
                  style={{ fontFamily: 'Inter, sans-serif' }}
                >
                  {isSubmitting && <Loader2 className="w-3.5 h-3.5 animate-spin" />}
                  Continue with Email
                </button>
              </form>

              <p className="text-xs text-white/40" style={{ fontFamily: 'Inter, sans-serif' }}>
                Remember your password?{' '}
                <button
                  onClick={() => { resetForm(); setView('sign-in'); }}
                  className="text-white font-bold hover:text-rose-400 transition-colors cursor-pointer"
                >
                  Back to Login
                </button>
              </p>
            </div>
          )}

        </div>
      </div>
    </div>
  );
};

export default Profile;

import React, { createContext, useContext, useEffect, useState, useCallback } from 'react';
import { supabase } from '@/lib/supabaseClient';
import type { User, Session, AuthError } from '@supabase/supabase-js';

// ── Profile shape (mirrors public.profiles table) ──
export interface UserProfile {
  id: string;
  email: string | null;
  display_name: string | null;
  avatar_url: string | null;
  created_at: string;
  updated_at: string;
}

// ── Context value ──
interface AuthContextValue {
  user: User | null;
  session: Session | null;
  profile: UserProfile | null;
  isLoading: boolean;
  signInWithEmail: (email: string, password: string) => Promise<{ error: AuthError | null }>;
  signUpWithEmail: (email: string, password: string, displayName?: string) => Promise<{ error: AuthError | null }>;
  signInWithGoogle: () => Promise<{ error: AuthError | null }>;
  signOut: () => Promise<void>;
  deleteAccount: () => Promise<{ error: string | null }>;
  refreshProfile: () => Promise<void>;
}

const AuthContext = createContext<AuthContextValue | undefined>(undefined);

// ── Provider ──
export const AuthProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [user, setUser] = useState<User | null>(null);
  const [session, setSession] = useState<Session | null>(null);
  const [profile, setProfile] = useState<UserProfile | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  // Fetch profile from public.profiles
  const fetchProfile = useCallback(async (userId: string, currentUser: User) => {
    try {
      const { data, error } = await supabase
        .from('profiles')
        .select('*')
        .eq('id', userId)
        .maybeSingle();

      const googleName = currentUser.user_metadata?.full_name;
      const googleAvatar = currentUser.user_metadata?.avatar_url;

      if (!data) {
        // Profile does not exist yet, create one
        const newProfile = {
          id: userId,
          email: currentUser.email,
          display_name: googleName || currentUser.email?.split('@')[0] || 'User',
          avatar_url: googleAvatar || null,
        };
        const { data: insertedData, error: insertError } = await supabase
          .from('profiles')
          .insert(newProfile)
          .select()
          .single();

        if (insertError) {
          console.error('[Auth] Profile creation error:', insertError.message);
          // Fallback to local state if db insert fails
          setProfile({
            id: userId,
            email: currentUser.email ?? null,
            display_name: newProfile.display_name,
            avatar_url: newProfile.avatar_url,
            created_at: new Date().toISOString(),
            updated_at: new Date().toISOString(),
          });
        } else {
          setProfile(insertedData as UserProfile);
        }
      } else {
        // Profile exists. Let's see if we should sync Google metadata
        const currentProfile = data as UserProfile;
        const isGoogleProvider = 
          currentUser.app_metadata?.provider === 'google' || 
          currentUser.identities?.some(id => id.provider === 'google');

        if (isGoogleProvider && googleName && currentProfile.display_name !== googleName) {
          const { data: updatedData } = await supabase
            .from('profiles')
            .update({
              display_name: googleName,
              avatar_url: googleAvatar || currentProfile.avatar_url,
            })
            .eq('id', userId)
            .select()
            .single();

          if (updatedData) {
            setProfile(updatedData as UserProfile);
            return;
          }
        }
        setProfile(currentProfile);
      }
    } catch (e) {
      console.error('[Auth] Error in fetchProfile:', e);
      setProfile(null);
    }
  }, []);

  const refreshProfile = useCallback(async () => {
    if (user) {
      await fetchProfile(user.id, user);
    }
  }, [user, fetchProfile]);

  // Listen for auth state changes
  useEffect(() => {
    // Get initial session
    supabase.auth.getSession().then(({ data: { session: initialSession } }) => {
      setSession(initialSession);
      setUser(initialSession?.user ?? null);
      if (initialSession?.user) {
        fetchProfile(initialSession.user.id, initialSession.user);
      }
      setIsLoading(false);
    });

    // Subscribe to auth changes
    const { data: { subscription } } = supabase.auth.onAuthStateChange(
      async (_event, newSession) => {
        setSession(newSession);
        setUser(newSession?.user ?? null);
        if (newSession?.user) {
          await fetchProfile(newSession.user.id, newSession.user);
        } else {
          setProfile(null);
        }
        setIsLoading(false);
      }
    );

    return () => {
      subscription.unsubscribe();
    };
  }, [fetchProfile]);

  // ── Auth methods ──

  const signInWithEmail = async (email: string, password: string) => {
    const { error } = await supabase.auth.signInWithPassword({ email, password });
    return { error };
  };

  const signUpWithEmail = async (email: string, password: string, displayName?: string) => {
    const { error } = await supabase.auth.signUp({
      email,
      password,
      options: {
        data: {
          full_name: displayName || email.split('@')[0],
        },
      },
    });
    return { error };
  };

  const signInWithGoogle = async () => {
    const { error } = await supabase.auth.signInWithOAuth({
      provider: 'google',
      options: {
        redirectTo: window.location.origin + '/profile',
      },
    });
    return { error };
  };

  const signOut = async () => {
    await supabase.auth.signOut();
    setUser(null);
    setSession(null);
    setProfile(null);
  };

  const deleteAccount = async (): Promise<{ error: string | null }> => {
    if (!user) return { error: 'No user is signed in.' };

    try {
      // Delete profile (cascades will handle favourites via FK)
      const { error: profileError } = await supabase
        .from('profiles')
        .delete()
        .eq('id', user.id);

      if (profileError) {
        return { error: `Failed to delete profile: ${profileError.message}` };
      }

      // Sign out after deletion
      await supabase.auth.signOut();
      setUser(null);
      setSession(null);
      setProfile(null);

      return { error: null };
    } catch (err: any) {
      return { error: err.message || 'An unexpected error occurred.' };
    }
  };

  return (
    <AuthContext.Provider
      value={{
        user,
        session,
        profile,
        isLoading,
        signInWithEmail,
        signUpWithEmail,
        signInWithGoogle,
        signOut,
        deleteAccount,
        refreshProfile,
      }}
    >
      {children}
    </AuthContext.Provider>
  );
};

// ── Hook ──
export const useAuth = (): AuthContextValue => {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
};

export default AuthProvider;

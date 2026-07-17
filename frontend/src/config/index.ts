export const APP_CONFIG = {
  appName: 'ChitraAI',
  appDescription: 'AI-Powered Semantic Movie Recommendations platform.',
  apiBaseUrl:
    import.meta.env.VITE_API_BASE_URL ||
    (import.meta.env.DEV ? 'http://127.0.0.1:8000/api/v1' : '/api/v1'),
  defaultSearchLimit: 12,
  imageFallbacks: {
    backdrop:
      'https://images.unsplash.com/photo-1536440136628-849c177e76a1?q=80&w=1200&auto=format&fit=crop',
    poster:
      'https://images.unsplash.com/photo-1440404653325-ab127d49abc1?q=80&w=600&auto=format&fit=crop',
  },
  localStorageKeys: {
    favorites: 'chitra_ai_favorites',
    searchHistory: 'chitra_ai_search_history',
    theme: 'chitra_ai_theme',
  },
};

export default APP_CONFIG;

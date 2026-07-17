import { useCallback, useState } from 'react';

export const GUEST_SEARCH_LIMIT = 5;
const GUEST_SEARCH_COUNT_KEY = 'guest_search_count';

const getStoredSearchCount = (): number => {
  if (typeof window === 'undefined') return 0;

  const value = Number.parseInt(window.localStorage.getItem(GUEST_SEARCH_COUNT_KEY) ?? '0', 10);
  return Number.isFinite(value) && value > 0 ? Math.min(value, GUEST_SEARCH_LIMIT) : 0;
};

/** Persists the number of successful recommendation responses made while signed out. */
export const useGuestSearchLimit = () => {
  const [guestSearchCount, setGuestSearchCount] = useState(getStoredSearchCount);

  const recordSuccessfulGuestSearch = useCallback(() => {
    setGuestSearchCount((currentCount) => {
      const nextCount = Math.min(currentCount + 1, GUEST_SEARCH_LIMIT);
      window.localStorage.setItem(GUEST_SEARCH_COUNT_KEY, String(nextCount));
      return nextCount;
    });
  }, []);

  return {
    guestSearchCount,
    isGuestSearchLimitReached: guestSearchCount >= GUEST_SEARCH_LIMIT,
    recordSuccessfulGuestSearch,
  };
};

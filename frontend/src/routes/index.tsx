/* eslint-disable react-refresh/only-export-components */
import React, { Suspense } from 'react';
import { createBrowserRouter } from 'react-router-dom';
import AppShell from '@/layouts/AppShell';
import LoadingScreen from '@/components/shared/LoadingScreen';

// Lazy load page views
const Home = React.lazy(() => import('@/pages/Home'));
const Search = React.lazy(() => import('@/pages/Search'));
const Recommendations = React.lazy(() => import('@/pages/Recommendations'));
const Favorites = React.lazy(() => import('@/pages/Favorites'));
const Profile = React.lazy(() => import('@/pages/Profile'));
const Dashboard = React.lazy(() => import('@/pages/Dashboard'));
const MovieDetails = React.lazy(() => import('@/pages/MovieDetails'));
const NotFound = React.lazy(() => import('@/pages/NotFound'));

// Wrapper utility for Lazy Suspense routing
const withSuspense = (Component: React.ComponentType) => (
  <Suspense fallback={<LoadingScreen />}>
    <Component />
  </Suspense>
);

export const router = createBrowserRouter([
  {
    path: '/',
    element: <AppShell />,
    children: [
      {
        index: true,
        element: withSuspense(Home),
      },
      {
        path: 'search',
        element: withSuspense(Search),
      },
      {
        path: 'recommendations',
        element: withSuspense(Recommendations),
      },
      {
        path: 'favorites',
        element: withSuspense(Favorites),
      },
      {
        path: 'profile',
        element: withSuspense(Profile),
      },
      {
        path: 'dashboard',
        element: withSuspense(Dashboard),
      },
      {
        path: 'movie/:id',
        element: withSuspense(MovieDetails),
      },
      {
        path: '*',
        element: withSuspense(NotFound),
      },
    ],
  },
]);

export default router;

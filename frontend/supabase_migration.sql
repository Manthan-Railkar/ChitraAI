-- ============================================================
-- ChitraAI Supabase Migration
-- Run this SQL in your Supabase SQL Editor (https://supabase.com/dashboard)
-- ============================================================

-- 1. Profiles Table
-- Stores user display names and avatars. Auto-created on signup via trigger.
create table if not exists public.profiles (
  id uuid references auth.users(id) on delete cascade primary key,
  email text,
  display_name text,
  avatar_url text,
  created_at timestamptz default now() not null,
  updated_at timestamptz default now() not null
);

-- Enable RLS on profiles
alter table public.profiles enable row level security;

-- Profiles RLS policies
create policy "Users can view their own profile"
  on public.profiles for select
  using (auth.uid() = id);

create policy "Users can update their own profile"
  on public.profiles for update
  using (auth.uid() = id);

create policy "Users can insert their own profile"
  on public.profiles for insert
  with check (auth.uid() = id);

create policy "Users can delete their own profile"
  on public.profiles for delete
  using (auth.uid() = id);

-- Auto-create profile on new user signup
create or replace function public.handle_new_user()
returns trigger as $$
begin
  insert into public.profiles (id, email, display_name, avatar_url)
  values (
    new.id,
    new.email,
    coalesce(new.raw_user_meta_data->>'full_name', split_part(new.email, '@', 1)),
    coalesce(new.raw_user_meta_data->>'avatar_url', null)
  );
  return new;
end;
$$ language plpgsql security definer;

-- Trigger to call the function after a new user signs up
drop trigger if exists on_auth_user_created on auth.users;
create trigger on_auth_user_created
  after insert on auth.users
  for each row execute procedure public.handle_new_user();


-- 2. Favourites Table
-- Stores the movies a user has favourited.
create table if not exists public.favourites (
  id uuid default gen_random_uuid() primary key,
  user_id uuid references auth.users(id) on delete cascade not null,
  movie_id text not null,
  title text not null,
  poster_path text,
  genres text[] default '{}',
  rating_value real,
  release_year integer,
  overview text,
  runtime_minutes integer,
  created_at timestamptz default now() not null,
  unique(user_id, movie_id)
);

-- Enable RLS on favourites
alter table public.favourites enable row level security;

-- Favourites RLS policies
create policy "Users can view their own favourites"
  on public.favourites for select
  using (auth.uid() = user_id);

create policy "Users can insert their own favourites"
  on public.favourites for insert
  with check (auth.uid() = user_id);

create policy "Users can delete their own favourites"
  on public.favourites for delete
  using (auth.uid() = user_id);


-- 3. Updated_at auto-update trigger for profiles
create or replace function public.update_updated_at_column()
returns trigger as $$
begin
  new.updated_at = now();
  return new;
end;
$$ language plpgsql;

drop trigger if exists set_updated_at on public.profiles;
create trigger set_updated_at
  before update on public.profiles
  for each row execute procedure public.update_updated_at_column();

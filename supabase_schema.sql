-- Fixed schema with correct table creation order
-- Create base tables first, then tables with dependencies

-- 1. Create users table first (no dependencies)
CREATE TABLE public.users (
  id uuid NOT NULL DEFAULT gen_random_uuid(),
  email text NOT NULL UNIQUE,
  name text,
  created_at timestamp with time zone DEFAULT now(),
  CONSTRAINT users_pkey PRIMARY KEY (id)
);

-- 2. Create workspaces table (depends only on users)
CREATE TABLE public.workspaces (
  id uuid NOT NULL DEFAULT gen_random_uuid(),
  name text NOT NULL,
  owner_id uuid NOT NULL,
  description text,
  created_at timestamp with time zone DEFAULT now(),
  updated_at timestamp with time zone DEFAULT now(),
  CONSTRAINT workspaces_pkey PRIMARY KEY (id),
  CONSTRAINT workspaces_owner_id_fkey FOREIGN KEY (owner_id) REFERENCES public.users(id)
);

-- 3. Create files table (depends on users and workspaces)
CREATE TABLE public.files (
  id uuid NOT NULL DEFAULT gen_random_uuid(),
  workspace_id uuid NOT NULL,
  filename text NOT NULL,
  file_type text DEFAULT 'markdown'::text,
  content text DEFAULT ''::text,
  created_by uuid NOT NULL,
  created_at timestamp with time zone DEFAULT now(),
  updated_at timestamp with time zone DEFAULT now(),
  CONSTRAINT files_pkey PRIMARY KEY (id),
  CONSTRAINT files_workspace_id_fkey FOREIGN KEY (workspace_id) REFERENCES public.workspaces(id),
  CONSTRAINT files_created_by_fkey FOREIGN KEY (created_by) REFERENCES public.users(id)
);

-- 4. Create remaining tables (all depend on users and/or workspaces and/or files)
CREATE TABLE public.chat_messages (
  id uuid NOT NULL DEFAULT gen_random_uuid(),
  workspace_id uuid NOT NULL,
  sender_id uuid NOT NULL,
  content text NOT NULL,
  is_ai boolean DEFAULT false,
  created_at timestamp with time zone DEFAULT now(),
  CONSTRAINT chat_messages_pkey PRIMARY KEY (id),
  CONSTRAINT chat_messages_sender_id_fkey FOREIGN KEY (sender_id) REFERENCES public.users(id),
  CONSTRAINT chat_messages_workspace_id_fkey FOREIGN KEY (workspace_id) REFERENCES public.workspaces(id)
);

CREATE TABLE public.collaborators (
  id uuid NOT NULL DEFAULT gen_random_uuid(),
  workspace_id uuid NOT NULL,
  user_id uuid NOT NULL,
  role text NOT NULL DEFAULT 'editor'::text CHECK (role = ANY (ARRAY['owner'::text, 'editor'::text, 'viewer'::text])),
  invited_at timestamp with time zone DEFAULT now(),
  CONSTRAINT collaborators_pkey PRIMARY KEY (id),
  CONSTRAINT collaborators_workspace_id_fkey FOREIGN KEY (workspace_id) REFERENCES public.workspaces(id),
  CONSTRAINT collaborators_user_id_fkey FOREIGN KEY (user_id) REFERENCES public.users(id)
);

CREATE TABLE public.file_versions (
  id uuid NOT NULL DEFAULT gen_random_uuid(),
  file_id uuid NOT NULL,
  version_number integer NOT NULL,
  content text NOT NULL,
  change_summary text,
  created_by uuid NOT NULL,
  created_at timestamp with time zone DEFAULT now(),
  CONSTRAINT file_versions_pkey PRIMARY KEY (id),
  CONSTRAINT file_versions_created_by_fkey FOREIGN KEY (created_by) REFERENCES public.users(id),
  CONSTRAINT file_versions_file_id_fkey FOREIGN KEY (file_id) REFERENCES public.files(id)
);

CREATE TABLE public.workspace_activity (
  id uuid NOT NULL DEFAULT gen_random_uuid(),
  workspace_id uuid NOT NULL,
  user_id uuid NOT NULL,
  action text NOT NULL,
  target_user_id uuid,
  details jsonb,
  created_at timestamp with time zone DEFAULT now(),
  CONSTRAINT workspace_activity_pkey PRIMARY KEY (id),
  CONSTRAINT workspace_activity_target_user_id_fkey FOREIGN KEY (target_user_id) REFERENCES public.users(id),
  CONSTRAINT workspace_activity_user_id_fkey FOREIGN KEY (user_id) REFERENCES public.users(id),
  CONSTRAINT workspace_activity_workspace_id_fkey FOREIGN KEY (workspace_id) REFERENCES public.workspaces(id)
);

CREATE TABLE public.workspace_invitations (
  id uuid NOT NULL DEFAULT gen_random_uuid(),
  workspace_id uuid NOT NULL,
  inviter_id uuid NOT NULL,
  invitee_email text NOT NULL,
  invitee_id uuid,
  role text NOT NULL DEFAULT 'editor'::text CHECK (role = ANY (ARRAY['owner'::text, 'editor'::text, 'viewer'::text])),
  status text NOT NULL DEFAULT 'pending'::text CHECK (status = ANY (ARRAY['pending'::text, 'accepted'::text, 'declined'::text, 'expired'::text])),
  invitation_token uuid DEFAULT gen_random_uuid(),
  expires_at timestamp with time zone DEFAULT (now() + '7 days'::interval),
  created_at timestamp with time zone DEFAULT now(),
  updated_at timestamp with time zone DEFAULT now(),
  CONSTRAINT workspace_invitations_pkey PRIMARY KEY (id),
  CONSTRAINT workspace_invitations_workspace_id_fkey FOREIGN KEY (workspace_id) REFERENCES public.workspaces(id),
  CONSTRAINT workspace_invitations_inviter_id_fkey FOREIGN KEY (inviter_id) REFERENCES public.users(id),
  CONSTRAINT workspace_invitations_invitee_id_fkey FOREIGN KEY (invitee_id) REFERENCES public.users(id)
);

CREATE TABLE public.workspace_requests (
  id uuid NOT NULL DEFAULT gen_random_uuid(),
  workspace_id uuid NOT NULL,
  requester_id uuid NOT NULL,
  message text,
  status text NOT NULL DEFAULT 'pending'::text CHECK (status = ANY (ARRAY['pending'::text, 'approved'::text, 'rejected'::text])),
  reviewed_by uuid,
  reviewed_at timestamp with time zone,
  created_at timestamp with time zone DEFAULT now(),
  CONSTRAINT workspace_requests_pkey PRIMARY KEY (id),
  CONSTRAINT workspace_requests_requester_id_fkey FOREIGN KEY (requester_id) REFERENCES public.users(id),
  CONSTRAINT workspace_requests_reviewed_by_fkey FOREIGN KEY (reviewed_by) REFERENCES public.users(id),
  CONSTRAINT workspace_requests_workspace_id_fkey FOREIGN KEY (workspace_id) REFERENCES public.workspaces(id)
);
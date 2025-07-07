--
-- PostgreSQL database dump
--

-- Dumped from database version 15.13 (Debian 15.13-1.pgdg120+1)
-- Dumped by pg_dump version 15.13 (Homebrew)

SET statement_timeout = 0;
SET lock_timeout = 0;
SET idle_in_transaction_session_timeout = 0;
SET client_encoding = 'UTF8';
SET standard_conforming_strings = on;
SELECT pg_catalog.set_config('search_path', '', false);
SET check_function_bodies = false;
SET xmloption = content;
SET client_min_messages = warning;
SET row_security = off;

SET default_tablespace = '';

SET default_table_access_method = heap;

--
-- Name: api_usage_logs; Type: TABLE; Schema: public; Owner: eindr
--

CREATE TABLE public.api_usage_logs (
    id integer NOT NULL,
    customer_id integer NOT NULL,
    end_point character varying(255),
    method character varying(10),
    status_code integer,
    response_time_ms integer,
    request_size_bytes integer,
    ip_address inet,
    user_agent character varying(255),
    created_at timestamp without time zone DEFAULT CURRENT_TIMESTAMP
);


ALTER TABLE public.api_usage_logs OWNER TO eindr;

--
-- Name: api_usage_logs_id_seq; Type: SEQUENCE; Schema: public; Owner: eindr
--

ALTER TABLE public.api_usage_logs ALTER COLUMN id ADD GENERATED ALWAYS AS IDENTITY (
    SEQUENCE NAME public.api_usage_logs_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1
);


--
-- Name: chat_messages; Type: TABLE; Schema: public; Owner: eindr
--

CREATE TABLE public.chat_messages (
    id integer NOT NULL,
    conversions_id integer NOT NULL,
    role character varying(50),
    description text,
    token_count integer,
    model_used character varying(50),
    response_time_ms integer,
    created_at timestamp without time zone DEFAULT CURRENT_TIMESTAMP
);


ALTER TABLE public.chat_messages OWNER TO eindr;

--
-- Name: chat_messages_id_seq; Type: SEQUENCE; Schema: public; Owner: eindr
--

ALTER TABLE public.chat_messages ALTER COLUMN id ADD GENERATED ALWAYS AS IDENTITY (
    SEQUENCE NAME public.chat_messages_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1
);


--
-- Name: condition_states; Type: TABLE; Schema: public; Owner: eindr
--

CREATE TABLE public.condition_states (
    id integer NOT NULL,
    label character varying(255)
);


ALTER TABLE public.condition_states OWNER TO eindr;

--
-- Name: condition_states_id_seq; Type: SEQUENCE; Schema: public; Owner: eindr
--

ALTER TABLE public.condition_states ALTER COLUMN id ADD GENERATED ALWAYS AS IDENTITY (
    SEQUENCE NAME public.condition_states_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1
);


--
-- Name: conversions; Type: TABLE; Schema: public; Owner: eindr
--

CREATE TABLE public.conversions (
    id integer NOT NULL,
    customer_id integer NOT NULL,
    title character varying(255),
    is_active boolean DEFAULT true,
    created_at timestamp without time zone DEFAULT CURRENT_TIMESTAMP,
    updated_at timestamp without time zone DEFAULT CURRENT_TIMESTAMP,
    last_message_at timestamp without time zone
);


ALTER TABLE public.conversions OWNER TO eindr;

--
-- Name: conversions_id_seq; Type: SEQUENCE; Schema: public; Owner: eindr
--

ALTER TABLE public.conversions ALTER COLUMN id ADD GENERATED ALWAYS AS IDENTITY (
    SEQUENCE NAME public.conversions_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1
);


--
-- Name: customer_preferences; Type: TABLE; Schema: public; Owner: eindr
--

CREATE TABLE public.customer_preferences (
    id integer NOT NULL,
    customer_id integer NOT NULL,
    allow_friends boolean DEFAULT true,
    received_shared_notes boolean DEFAULT true,
    notification_sound character varying(255),
    language_id integer,
    chat_history_enabled boolean DEFAULT true,
    theme character varying(50),
    email_notifications boolean DEFAULT true,
    push_notifications boolean DEFAULT true,
    notification_frequency character varying(50),
    auto_backup boolean DEFAULT true,
    data_retention_days integer DEFAULT 30,
    updated_at timestamp without time zone DEFAULT CURRENT_TIMESTAMP
);


ALTER TABLE public.customer_preferences OWNER TO eindr;

--
-- Name: customer_preferences_id_seq; Type: SEQUENCE; Schema: public; Owner: eindr
--

ALTER TABLE public.customer_preferences ALTER COLUMN id ADD GENERATED ALWAYS AS IDENTITY (
    SEQUENCE NAME public.customer_preferences_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1
);


--
-- Name: customer_sessions; Type: TABLE; Schema: public; Owner: eindr
--

CREATE TABLE public.customer_sessions (
    id integer NOT NULL,
    customer_id integer NOT NULL,
    session_token character varying(255) NOT NULL,
    ip_address inet,
    user_agent character varying(255),
    expires_at timestamp without time zone,
    created_at timestamp without time zone DEFAULT CURRENT_TIMESTAMP
);


ALTER TABLE public.customer_sessions OWNER TO eindr;

--
-- Name: customer_sessions_id_seq; Type: SEQUENCE; Schema: public; Owner: eindr
--

ALTER TABLE public.customer_sessions ALTER COLUMN id ADD GENERATED ALWAYS AS IDENTITY (
    SEQUENCE NAME public.customer_sessions_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1
);


--
-- Name: customers; Type: TABLE; Schema: public; Owner: eindr
--

CREATE TABLE public.customers (
    id integer NOT NULL,
    email character varying(255) NOT NULL,
    password_hash character varying(255) NOT NULL,
    is_verified boolean DEFAULT false,
    is_active boolean DEFAULT true,
    created_at timestamp without time zone DEFAULT CURRENT_TIMESTAMP,
    updated_at timestamp without time zone DEFAULT CURRENT_TIMESTAMP,
    last_login timestamp without time zone,
    login_attempts integer DEFAULT 0,
    locked_until timestamp without time zone,
    subscription_plan_id integer
);


ALTER TABLE public.customers OWNER TO eindr;

--
-- Name: customers_devices; Type: TABLE; Schema: public; Owner: eindr
--

CREATE TABLE public.customers_devices (
    id integer NOT NULL,
    customer_id integer NOT NULL,
    device_token character varying(255),
    device_type character varying(50),
    device_name character varying(255),
    device_model character varying(255),
    os_version character varying(50),
    app_version character varying(50),
    is_active boolean DEFAULT true,
    last_seen timestamp without time zone,
    register_at timestamp without time zone DEFAULT CURRENT_TIMESTAMP
);


ALTER TABLE public.customers_devices OWNER TO eindr;

--
-- Name: customers_devices_id_seq; Type: SEQUENCE; Schema: public; Owner: eindr
--

ALTER TABLE public.customers_devices ALTER COLUMN id ADD GENERATED ALWAYS AS IDENTITY (
    SEQUENCE NAME public.customers_devices_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1
);


--
-- Name: customers_id_seq; Type: SEQUENCE; Schema: public; Owner: eindr
--

ALTER TABLE public.customers ALTER COLUMN id ADD GENERATED ALWAYS AS IDENTITY (
    SEQUENCE NAME public.customers_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1
);


--
-- Name: customers_profiles; Type: TABLE; Schema: public; Owner: eindr
--

CREATE TABLE public.customers_profiles (
    id integer NOT NULL,
    customer_id integer NOT NULL,
    full_name character varying(255),
    user_name character varying(255),
    bio text,
    avatar_url character varying(255),
    phone_number character varying(20),
    date_of_birth date,
    timezone_id integer,
    language_id integer,
    subscription_plan_id integer,
    country character varying(255),
    city character varying(255),
    created_at timestamp without time zone DEFAULT CURRENT_TIMESTAMP,
    updated_at timestamp without time zone DEFAULT CURRENT_TIMESTAMP
);


ALTER TABLE public.customers_profiles OWNER TO eindr;

--
-- Name: customers_profiles_id_seq; Type: SEQUENCE; Schema: public; Owner: eindr
--

ALTER TABLE public.customers_profiles ALTER COLUMN id ADD GENERATED ALWAYS AS IDENTITY (
    SEQUENCE NAME public.customers_profiles_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1
);


--
-- Name: friend_permissions; Type: TABLE; Schema: public; Owner: eindr
--

CREATE TABLE public.friend_permissions (
    id integer NOT NULL,
    customer_id integer NOT NULL,
    friend_id integer NOT NULL,
    auto_accept_reminders boolean DEFAULT false,
    auto_accept_notes boolean DEFAULT false,
    can_view_schedule boolean DEFAULT true,
    can_create_reminders boolean DEFAULT true,
    can_view_ledger boolean DEFAULT true,
    can_view_activity boolean DEFAULT true,
    notification_level character varying(50),
    updated_at timestamp without time zone DEFAULT CURRENT_TIMESTAMP
);


ALTER TABLE public.friend_permissions OWNER TO eindr;

--
-- Name: friend_permissions_id_seq; Type: SEQUENCE; Schema: public; Owner: eindr
--

ALTER TABLE public.friend_permissions ALTER COLUMN id ADD GENERATED ALWAYS AS IDENTITY (
    SEQUENCE NAME public.friend_permissions_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1
);


--
-- Name: friend_request_history; Type: TABLE; Schema: public; Owner: eindr
--

CREATE TABLE public.friend_request_history (
    id integer NOT NULL,
    requester_id integer NOT NULL,
    requested_id integer NOT NULL,
    action character varying(50),
    message text,
    ip_address inet,
    created_at timestamp without time zone DEFAULT CURRENT_TIMESTAMP
);


ALTER TABLE public.friend_request_history OWNER TO eindr;

--
-- Name: friend_request_history_id_seq; Type: SEQUENCE; Schema: public; Owner: eindr
--

ALTER TABLE public.friend_request_history ALTER COLUMN id ADD GENERATED ALWAYS AS IDENTITY (
    SEQUENCE NAME public.friend_request_history_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1
);


--
-- Name: friendships; Type: TABLE; Schema: public; Owner: eindr
--

CREATE TABLE public.friendships (
    id integer NOT NULL,
    customer_id integer NOT NULL,
    friend_id integer NOT NULL,
    status character varying(50),
    initiated_by character varying(50),
    message text,
    created_at timestamp without time zone DEFAULT CURRENT_TIMESTAMP,
    updated_at timestamp without time zone DEFAULT CURRENT_TIMESTAMP,
    accepted_at timestamp without time zone
);


ALTER TABLE public.friendships OWNER TO eindr;

--
-- Name: friendships_id_seq; Type: SEQUENCE; Schema: public; Owner: eindr
--

ALTER TABLE public.friendships ALTER COLUMN id ADD GENERATED ALWAYS AS IDENTITY (
    SEQUENCE NAME public.friendships_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1
);


--
-- Name: label_codes; Type: TABLE; Schema: public; Owner: eindr
--

CREATE TABLE public.label_codes (
    id integer NOT NULL,
    name character varying(255),
    label_group_id integer
);


ALTER TABLE public.label_codes OWNER TO eindr;

--
-- Name: label_codes_id_seq; Type: SEQUENCE; Schema: public; Owner: eindr
--

ALTER TABLE public.label_codes ALTER COLUMN id ADD GENERATED ALWAYS AS IDENTITY (
    SEQUENCE NAME public.label_codes_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1
);


--
-- Name: label_groups; Type: TABLE; Schema: public; Owner: eindr
--

CREATE TABLE public.label_groups (
    id integer NOT NULL,
    group_name character varying(255)
);


ALTER TABLE public.label_groups OWNER TO eindr;

--
-- Name: label_groups_id_seq; Type: SEQUENCE; Schema: public; Owner: eindr
--

ALTER TABLE public.label_groups ALTER COLUMN id ADD GENERATED ALWAYS AS IDENTITY (
    SEQUENCE NAME public.label_groups_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1
);


--
-- Name: language_label; Type: TABLE; Schema: public; Owner: eindr
--

CREATE TABLE public.language_label (
    id integer NOT NULL,
    language_id integer NOT NULL,
    label_code_id integer NOT NULL,
    label_text character varying(255)
);


ALTER TABLE public.language_label OWNER TO eindr;

--
-- Name: language_label_id_seq; Type: SEQUENCE; Schema: public; Owner: eindr
--

ALTER TABLE public.language_label ALTER COLUMN id ADD GENERATED ALWAYS AS IDENTITY (
    SEQUENCE NAME public.language_label_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1
);


--
-- Name: languages; Type: TABLE; Schema: public; Owner: eindr
--

CREATE TABLE public.languages (
    id integer NOT NULL,
    name character varying(255),
    direction character varying(10),
    is_active boolean DEFAULT true,
    lang_code character varying(10),
    icon character varying(50)
);


ALTER TABLE public.languages OWNER TO eindr;

--
-- Name: languages_id_seq; Type: SEQUENCE; Schema: public; Owner: eindr
--

ALTER TABLE public.languages ALTER COLUMN id ADD GENERATED ALWAYS AS IDENTITY (
    SEQUENCE NAME public.languages_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1
);


--
-- Name: ledger_direction; Type: TABLE; Schema: public; Owner: eindr
--

CREATE TABLE public.ledger_direction (
    id integer NOT NULL,
    label character varying(100)
);


ALTER TABLE public.ledger_direction OWNER TO eindr;

--
-- Name: ledger_direction_id_seq; Type: SEQUENCE; Schema: public; Owner: eindr
--

ALTER TABLE public.ledger_direction ALTER COLUMN id ADD GENERATED ALWAYS AS IDENTITY (
    SEQUENCE NAME public.ledger_direction_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1
);


--
-- Name: ledger_entries; Type: TABLE; Schema: public; Owner: eindr
--

CREATE TABLE public.ledger_entries (
    id integer NOT NULL,
    customer_id integer NOT NULL,
    friend_id integer NOT NULL,
    amount numeric(10,2),
    ledger_direction_id integer NOT NULL,
    notes text,
    created_at timestamp without time zone DEFAULT CURRENT_TIMESTAMP,
    updated_at timestamp without time zone DEFAULT CURRENT_TIMESTAMP
);


ALTER TABLE public.ledger_entries OWNER TO eindr;

--
-- Name: ledger_entries_id_seq; Type: SEQUENCE; Schema: public; Owner: eindr
--

ALTER TABLE public.ledger_entries ALTER COLUMN id ADD GENERATED ALWAYS AS IDENTITY (
    SEQUENCE NAME public.ledger_entries_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1
);


--
-- Name: login_attempts; Type: TABLE; Schema: public; Owner: eindr
--

CREATE TABLE public.login_attempts (
    id integer NOT NULL,
    customer_id integer,
    email character varying(255) NOT NULL,
    ip_address inet,
    user_agent character varying(255),
    is_success boolean DEFAULT false,
    failure_reason character varying(255),
    attempted_at timestamp without time zone DEFAULT CURRENT_TIMESTAMP
);


ALTER TABLE public.login_attempts OWNER TO eindr;

--
-- Name: login_attempts_id_seq; Type: SEQUENCE; Schema: public; Owner: eindr
--

ALTER TABLE public.login_attempts ALTER COLUMN id ADD GENERATED ALWAYS AS IDENTITY (
    SEQUENCE NAME public.login_attempts_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1
);


--
-- Name: note_shares; Type: TABLE; Schema: public; Owner: eindr
--

CREATE TABLE public.note_shares (
    id integer NOT NULL,
    note_id integer NOT NULL,
    owner_customer_id integer NOT NULL,
    shared_with_customer_id integer NOT NULL,
    can_edit boolean DEFAULT false,
    can_comment boolean DEFAULT true,
    status character varying(50),
    shared_at timestamp without time zone DEFAULT CURRENT_TIMESTAMP,
    responded_at timestamp without time zone
);


ALTER TABLE public.note_shares OWNER TO eindr;

--
-- Name: note_shares_id_seq; Type: SEQUENCE; Schema: public; Owner: eindr
--

ALTER TABLE public.note_shares ALTER COLUMN id ADD GENERATED ALWAYS AS IDENTITY (
    SEQUENCE NAME public.note_shares_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1
);


--
-- Name: notes; Type: TABLE; Schema: public; Owner: eindr
--

CREATE TABLE public.notes (
    id integer NOT NULL,
    customer_id integer NOT NULL,
    title character varying(255),
    description text,
    content_type character varying(50),
    is_shared boolean DEFAULT false,
    is_favorite boolean DEFAULT false,
    is_pinned boolean DEFAULT false,
    created_at timestamp without time zone DEFAULT CURRENT_TIMESTAMP,
    updated_at timestamp without time zone DEFAULT CURRENT_TIMESTAMP,
    last_accessed timestamp without time zone
);


ALTER TABLE public.notes OWNER TO eindr;

--
-- Name: notes_id_seq; Type: SEQUENCE; Schema: public; Owner: eindr
--

ALTER TABLE public.notes ALTER COLUMN id ADD GENERATED ALWAYS AS IDENTITY (
    SEQUENCE NAME public.notes_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1
);


--
-- Name: priority_levels; Type: TABLE; Schema: public; Owner: eindr
--

CREATE TABLE public.priority_levels (
    id integer NOT NULL,
    label character varying(255),
    rank integer
);


ALTER TABLE public.priority_levels OWNER TO eindr;

--
-- Name: priority_levels_id_seq; Type: SEQUENCE; Schema: public; Owner: eindr
--

ALTER TABLE public.priority_levels ALTER COLUMN id ADD GENERATED ALWAYS AS IDENTITY (
    SEQUENCE NAME public.priority_levels_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1
);


--
-- Name: reminder_notifications; Type: TABLE; Schema: public; Owner: eindr
--

CREATE TABLE public.reminder_notifications (
    id integer NOT NULL,
    reminder_id integer NOT NULL,
    customer_id integer NOT NULL,
    notification_type character varying(50),
    status character varying(50),
    scheduled_at timestamp without time zone,
    sent_at timestamp without time zone,
    delivered_at timestamp without time zone,
    delivery_attempts integer,
    last_attempt_at timestamp without time zone,
    failure_reason text,
    created_at timestamp without time zone DEFAULT CURRENT_TIMESTAMP
);


ALTER TABLE public.reminder_notifications OWNER TO eindr;

--
-- Name: reminder_notifications_id_seq; Type: SEQUENCE; Schema: public; Owner: eindr
--

ALTER TABLE public.reminder_notifications ALTER COLUMN id ADD GENERATED ALWAYS AS IDENTITY (
    SEQUENCE NAME public.reminder_notifications_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1
);


--
-- Name: reminder_shares; Type: TABLE; Schema: public; Owner: eindr
--

CREATE TABLE public.reminder_shares (
    id integer NOT NULL,
    reminder_id integer NOT NULL,
    owner_customer_id integer NOT NULL,
    shared_with_customer_id integer NOT NULL,
    can_edit boolean DEFAULT false,
    can_complete boolean DEFAULT true,
    can_reshedule boolean DEFAULT true,
    status character varying(50),
    shared_at timestamp without time zone DEFAULT CURRENT_TIMESTAMP,
    responded_at timestamp without time zone
);


ALTER TABLE public.reminder_shares OWNER TO eindr;

--
-- Name: reminder_shares_id_seq; Type: SEQUENCE; Schema: public; Owner: eindr
--

ALTER TABLE public.reminder_shares ALTER COLUMN id ADD GENERATED ALWAYS AS IDENTITY (
    SEQUENCE NAME public.reminder_shares_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1
);


--
-- Name: reminders; Type: TABLE; Schema: public; Owner: eindr
--

CREATE TABLE public.reminders (
    id integer NOT NULL,
    customer_id integer NOT NULL,
    title text,
    description text,
    "time" timestamp without time zone,
    repeat_pattern_id integer,
    timezone_id integer,
    is_shared boolean DEFAULT false,
    is_active boolean DEFAULT true,
    next_occurrence timestamp without time zone,
    occurrence_count integer DEFAULT 0,
    max_occurrence integer,
    priority_id integer,
    created_at timestamp without time zone DEFAULT CURRENT_TIMESTAMP,
    updated_at timestamp without time zone DEFAULT CURRENT_TIMESTAMP,
    is_completed boolean DEFAULT false,
    completed_at timestamp without time zone
);


ALTER TABLE public.reminders OWNER TO eindr;

--
-- Name: reminders_id_seq; Type: SEQUENCE; Schema: public; Owner: eindr
--

ALTER TABLE public.reminders ALTER COLUMN id ADD GENERATED ALWAYS AS IDENTITY (
    SEQUENCE NAME public.reminders_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1
);


--
-- Name: subscription_plans; Type: TABLE; Schema: public; Owner: eindr
--

CREATE TABLE public.subscription_plans (
    id integer NOT NULL,
    plan_name character varying(255) NOT NULL,
    price numeric(10,2),
    billing_interval character varying(50),
    max_seats integer,
    description text,
    is_active boolean DEFAULT true,
    created_at timestamp without time zone DEFAULT CURRENT_TIMESTAMP
);


ALTER TABLE public.subscription_plans OWNER TO eindr;

--
-- Name: subscription_plans_id_seq; Type: SEQUENCE; Schema: public; Owner: eindr
--

ALTER TABLE public.subscription_plans ALTER COLUMN id ADD GENERATED ALWAYS AS IDENTITY (
    SEQUENCE NAME public.subscription_plans_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1
);


--
-- Name: timezones; Type: TABLE; Schema: public; Owner: eindr
--

CREATE TABLE public.timezones (
    id integer NOT NULL,
    name character varying(255),
    gmt_offset interval,
    created_at timestamp without time zone DEFAULT CURRENT_TIMESTAMP
);


ALTER TABLE public.timezones OWNER TO eindr;

--
-- Name: timezones_id_seq; Type: SEQUENCE; Schema: public; Owner: eindr
--

ALTER TABLE public.timezones ALTER COLUMN id ADD GENERATED ALWAYS AS IDENTITY (
    SEQUENCE NAME public.timezones_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1
);


--
-- Data for Name: api_usage_logs; Type: TABLE DATA; Schema: public; Owner: eindr
--

COPY public.api_usage_logs (id, customer_id, end_point, method, status_code, response_time_ms, request_size_bytes, ip_address, user_agent, created_at) FROM stdin;
\.


--
-- Data for Name: chat_messages; Type: TABLE DATA; Schema: public; Owner: eindr
--

COPY public.chat_messages (id, conversions_id, role, description, token_count, model_used, response_time_ms, created_at) FROM stdin;
\.


--
-- Data for Name: condition_states; Type: TABLE DATA; Schema: public; Owner: eindr
--

COPY public.condition_states (id, label) FROM stdin;
\.


--
-- Data for Name: conversions; Type: TABLE DATA; Schema: public; Owner: eindr
--

COPY public.conversions (id, customer_id, title, is_active, created_at, updated_at, last_message_at) FROM stdin;
\.


--
-- Data for Name: customer_preferences; Type: TABLE DATA; Schema: public; Owner: eindr
--

COPY public.customer_preferences (id, customer_id, allow_friends, received_shared_notes, notification_sound, language_id, chat_history_enabled, theme, email_notifications, push_notifications, notification_frequency, auto_backup, data_retention_days, updated_at) FROM stdin;
\.


--
-- Data for Name: customer_sessions; Type: TABLE DATA; Schema: public; Owner: eindr
--

COPY public.customer_sessions (id, customer_id, session_token, ip_address, user_agent, expires_at, created_at) FROM stdin;
\.


--
-- Data for Name: customers; Type: TABLE DATA; Schema: public; Owner: eindr
--

COPY public.customers (id, email, password_hash, is_verified, is_active, created_at, updated_at, last_login, login_attempts, locked_until, subscription_plan_id) FROM stdin;
1	afnann@example.com	$2b$12$QvTDR9VukfrDOnthbL79KeqUWiZ05wGfLpgQiG2dfhiMS8rH5Jmd.	f	t	2025-07-06 10:43:43.128601	2025-07-06 10:44:49.931235	2025-07-06 10:44:50.168704	0	\N	\N
\.


--
-- Data for Name: customers_devices; Type: TABLE DATA; Schema: public; Owner: eindr
--

COPY public.customers_devices (id, customer_id, device_token, device_type, device_name, device_model, os_version, app_version, is_active, last_seen, register_at) FROM stdin;
\.


--
-- Data for Name: customers_profiles; Type: TABLE DATA; Schema: public; Owner: eindr
--

COPY public.customers_profiles (id, customer_id, full_name, user_name, bio, avatar_url, phone_number, date_of_birth, timezone_id, language_id, subscription_plan_id, country, city, created_at, updated_at) FROM stdin;
\.


--
-- Data for Name: friend_permissions; Type: TABLE DATA; Schema: public; Owner: eindr
--

COPY public.friend_permissions (id, customer_id, friend_id, auto_accept_reminders, auto_accept_notes, can_view_schedule, can_create_reminders, can_view_ledger, can_view_activity, notification_level, updated_at) FROM stdin;
\.


--
-- Data for Name: friend_request_history; Type: TABLE DATA; Schema: public; Owner: eindr
--

COPY public.friend_request_history (id, requester_id, requested_id, action, message, ip_address, created_at) FROM stdin;
\.


--
-- Data for Name: friendships; Type: TABLE DATA; Schema: public; Owner: eindr
--

COPY public.friendships (id, customer_id, friend_id, status, initiated_by, message, created_at, updated_at, accepted_at) FROM stdin;
\.


--
-- Data for Name: label_codes; Type: TABLE DATA; Schema: public; Owner: eindr
--

COPY public.label_codes (id, name, label_group_id) FROM stdin;
\.


--
-- Data for Name: label_groups; Type: TABLE DATA; Schema: public; Owner: eindr
--

COPY public.label_groups (id, group_name) FROM stdin;
\.


--
-- Data for Name: language_label; Type: TABLE DATA; Schema: public; Owner: eindr
--

COPY public.language_label (id, language_id, label_code_id, label_text) FROM stdin;
\.


--
-- Data for Name: languages; Type: TABLE DATA; Schema: public; Owner: eindr
--

COPY public.languages (id, name, direction, is_active, lang_code, icon) FROM stdin;
\.


--
-- Data for Name: ledger_direction; Type: TABLE DATA; Schema: public; Owner: eindr
--

COPY public.ledger_direction (id, label) FROM stdin;
\.


--
-- Data for Name: ledger_entries; Type: TABLE DATA; Schema: public; Owner: eindr
--

COPY public.ledger_entries (id, customer_id, friend_id, amount, ledger_direction_id, notes, created_at, updated_at) FROM stdin;
\.


--
-- Data for Name: login_attempts; Type: TABLE DATA; Schema: public; Owner: eindr
--

COPY public.login_attempts (id, customer_id, email, ip_address, user_agent, is_success, failure_reason, attempted_at) FROM stdin;
\.


--
-- Data for Name: note_shares; Type: TABLE DATA; Schema: public; Owner: eindr
--

COPY public.note_shares (id, note_id, owner_customer_id, shared_with_customer_id, can_edit, can_comment, status, shared_at, responded_at) FROM stdin;
\.


--
-- Data for Name: notes; Type: TABLE DATA; Schema: public; Owner: eindr
--

COPY public.notes (id, customer_id, title, description, content_type, is_shared, is_favorite, is_pinned, created_at, updated_at, last_accessed) FROM stdin;
\.


--
-- Data for Name: priority_levels; Type: TABLE DATA; Schema: public; Owner: eindr
--

COPY public.priority_levels (id, label, rank) FROM stdin;
\.


--
-- Data for Name: reminder_notifications; Type: TABLE DATA; Schema: public; Owner: eindr
--

COPY public.reminder_notifications (id, reminder_id, customer_id, notification_type, status, scheduled_at, sent_at, delivered_at, delivery_attempts, last_attempt_at, failure_reason, created_at) FROM stdin;
\.


--
-- Data for Name: reminder_shares; Type: TABLE DATA; Schema: public; Owner: eindr
--

COPY public.reminder_shares (id, reminder_id, owner_customer_id, shared_with_customer_id, can_edit, can_complete, can_reshedule, status, shared_at, responded_at) FROM stdin;
\.


--
-- Data for Name: reminders; Type: TABLE DATA; Schema: public; Owner: eindr
--

COPY public.reminders (id, customer_id, title, description, "time", repeat_pattern_id, timezone_id, is_shared, is_active, next_occurrence, occurrence_count, max_occurrence, priority_id, created_at, updated_at, is_completed, completed_at) FROM stdin;
\.


--
-- Data for Name: subscription_plans; Type: TABLE DATA; Schema: public; Owner: eindr
--

COPY public.subscription_plans (id, plan_name, price, billing_interval, max_seats, description, is_active, created_at) FROM stdin;
\.


--
-- Data for Name: timezones; Type: TABLE DATA; Schema: public; Owner: eindr
--

COPY public.timezones (id, name, gmt_offset, created_at) FROM stdin;
\.


--
-- Name: api_usage_logs_id_seq; Type: SEQUENCE SET; Schema: public; Owner: eindr
--

SELECT pg_catalog.setval('public.api_usage_logs_id_seq', 1, false);


--
-- Name: chat_messages_id_seq; Type: SEQUENCE SET; Schema: public; Owner: eindr
--

SELECT pg_catalog.setval('public.chat_messages_id_seq', 1, false);


--
-- Name: condition_states_id_seq; Type: SEQUENCE SET; Schema: public; Owner: eindr
--

SELECT pg_catalog.setval('public.condition_states_id_seq', 1, false);


--
-- Name: conversions_id_seq; Type: SEQUENCE SET; Schema: public; Owner: eindr
--

SELECT pg_catalog.setval('public.conversions_id_seq', 1, false);


--
-- Name: customer_preferences_id_seq; Type: SEQUENCE SET; Schema: public; Owner: eindr
--

SELECT pg_catalog.setval('public.customer_preferences_id_seq', 1, false);


--
-- Name: customer_sessions_id_seq; Type: SEQUENCE SET; Schema: public; Owner: eindr
--

SELECT pg_catalog.setval('public.customer_sessions_id_seq', 1, false);


--
-- Name: customers_devices_id_seq; Type: SEQUENCE SET; Schema: public; Owner: eindr
--

SELECT pg_catalog.setval('public.customers_devices_id_seq', 1, false);


--
-- Name: customers_id_seq; Type: SEQUENCE SET; Schema: public; Owner: eindr
--

SELECT pg_catalog.setval('public.customers_id_seq', 1, true);


--
-- Name: customers_profiles_id_seq; Type: SEQUENCE SET; Schema: public; Owner: eindr
--

SELECT pg_catalog.setval('public.customers_profiles_id_seq', 1, false);


--
-- Name: friend_permissions_id_seq; Type: SEQUENCE SET; Schema: public; Owner: eindr
--

SELECT pg_catalog.setval('public.friend_permissions_id_seq', 1, false);


--
-- Name: friend_request_history_id_seq; Type: SEQUENCE SET; Schema: public; Owner: eindr
--

SELECT pg_catalog.setval('public.friend_request_history_id_seq', 1, false);


--
-- Name: friendships_id_seq; Type: SEQUENCE SET; Schema: public; Owner: eindr
--

SELECT pg_catalog.setval('public.friendships_id_seq', 1, false);


--
-- Name: label_codes_id_seq; Type: SEQUENCE SET; Schema: public; Owner: eindr
--

SELECT pg_catalog.setval('public.label_codes_id_seq', 1, false);


--
-- Name: label_groups_id_seq; Type: SEQUENCE SET; Schema: public; Owner: eindr
--

SELECT pg_catalog.setval('public.label_groups_id_seq', 1, false);


--
-- Name: language_label_id_seq; Type: SEQUENCE SET; Schema: public; Owner: eindr
--

SELECT pg_catalog.setval('public.language_label_id_seq', 1, false);


--
-- Name: languages_id_seq; Type: SEQUENCE SET; Schema: public; Owner: eindr
--

SELECT pg_catalog.setval('public.languages_id_seq', 1, false);


--
-- Name: ledger_direction_id_seq; Type: SEQUENCE SET; Schema: public; Owner: eindr
--

SELECT pg_catalog.setval('public.ledger_direction_id_seq', 1, false);


--
-- Name: ledger_entries_id_seq; Type: SEQUENCE SET; Schema: public; Owner: eindr
--

SELECT pg_catalog.setval('public.ledger_entries_id_seq', 1, false);


--
-- Name: login_attempts_id_seq; Type: SEQUENCE SET; Schema: public; Owner: eindr
--

SELECT pg_catalog.setval('public.login_attempts_id_seq', 1, false);


--
-- Name: note_shares_id_seq; Type: SEQUENCE SET; Schema: public; Owner: eindr
--

SELECT pg_catalog.setval('public.note_shares_id_seq', 1, false);


--
-- Name: notes_id_seq; Type: SEQUENCE SET; Schema: public; Owner: eindr
--

SELECT pg_catalog.setval('public.notes_id_seq', 1, false);


--
-- Name: priority_levels_id_seq; Type: SEQUENCE SET; Schema: public; Owner: eindr
--

SELECT pg_catalog.setval('public.priority_levels_id_seq', 1, false);


--
-- Name: reminder_notifications_id_seq; Type: SEQUENCE SET; Schema: public; Owner: eindr
--

SELECT pg_catalog.setval('public.reminder_notifications_id_seq', 1, false);


--
-- Name: reminder_shares_id_seq; Type: SEQUENCE SET; Schema: public; Owner: eindr
--

SELECT pg_catalog.setval('public.reminder_shares_id_seq', 1, false);


--
-- Name: reminders_id_seq; Type: SEQUENCE SET; Schema: public; Owner: eindr
--

SELECT pg_catalog.setval('public.reminders_id_seq', 1, false);


--
-- Name: subscription_plans_id_seq; Type: SEQUENCE SET; Schema: public; Owner: eindr
--

SELECT pg_catalog.setval('public.subscription_plans_id_seq', 1, false);


--
-- Name: timezones_id_seq; Type: SEQUENCE SET; Schema: public; Owner: eindr
--

SELECT pg_catalog.setval('public.timezones_id_seq', 1, false);


--
-- Name: api_usage_logs api_usage_logs_pkey; Type: CONSTRAINT; Schema: public; Owner: eindr
--

ALTER TABLE ONLY public.api_usage_logs
    ADD CONSTRAINT api_usage_logs_pkey PRIMARY KEY (id);


--
-- Name: chat_messages chat_messages_pkey; Type: CONSTRAINT; Schema: public; Owner: eindr
--

ALTER TABLE ONLY public.chat_messages
    ADD CONSTRAINT chat_messages_pkey PRIMARY KEY (id);


--
-- Name: condition_states condition_states_pkey; Type: CONSTRAINT; Schema: public; Owner: eindr
--

ALTER TABLE ONLY public.condition_states
    ADD CONSTRAINT condition_states_pkey PRIMARY KEY (id);


--
-- Name: conversions conversions_pkey; Type: CONSTRAINT; Schema: public; Owner: eindr
--

ALTER TABLE ONLY public.conversions
    ADD CONSTRAINT conversions_pkey PRIMARY KEY (id);


--
-- Name: customer_preferences customer_preferences_pkey; Type: CONSTRAINT; Schema: public; Owner: eindr
--

ALTER TABLE ONLY public.customer_preferences
    ADD CONSTRAINT customer_preferences_pkey PRIMARY KEY (id);


--
-- Name: customer_sessions customer_sessions_pkey; Type: CONSTRAINT; Schema: public; Owner: eindr
--

ALTER TABLE ONLY public.customer_sessions
    ADD CONSTRAINT customer_sessions_pkey PRIMARY KEY (id);


--
-- Name: customers_devices customers_devices_pkey; Type: CONSTRAINT; Schema: public; Owner: eindr
--

ALTER TABLE ONLY public.customers_devices
    ADD CONSTRAINT customers_devices_pkey PRIMARY KEY (id);


--
-- Name: customers customers_email_key; Type: CONSTRAINT; Schema: public; Owner: eindr
--

ALTER TABLE ONLY public.customers
    ADD CONSTRAINT customers_email_key UNIQUE (email);


--
-- Name: customers customers_pkey; Type: CONSTRAINT; Schema: public; Owner: eindr
--

ALTER TABLE ONLY public.customers
    ADD CONSTRAINT customers_pkey PRIMARY KEY (id);


--
-- Name: customers_profiles customers_profiles_phone_number_key; Type: CONSTRAINT; Schema: public; Owner: eindr
--

ALTER TABLE ONLY public.customers_profiles
    ADD CONSTRAINT customers_profiles_phone_number_key UNIQUE (phone_number);


--
-- Name: customers_profiles customers_profiles_pkey; Type: CONSTRAINT; Schema: public; Owner: eindr
--

ALTER TABLE ONLY public.customers_profiles
    ADD CONSTRAINT customers_profiles_pkey PRIMARY KEY (id);


--
-- Name: customers_profiles customers_profiles_user_name_key; Type: CONSTRAINT; Schema: public; Owner: eindr
--

ALTER TABLE ONLY public.customers_profiles
    ADD CONSTRAINT customers_profiles_user_name_key UNIQUE (user_name);


--
-- Name: friend_permissions friend_permissions_pkey; Type: CONSTRAINT; Schema: public; Owner: eindr
--

ALTER TABLE ONLY public.friend_permissions
    ADD CONSTRAINT friend_permissions_pkey PRIMARY KEY (id);


--
-- Name: friend_request_history friend_request_history_pkey; Type: CONSTRAINT; Schema: public; Owner: eindr
--

ALTER TABLE ONLY public.friend_request_history
    ADD CONSTRAINT friend_request_history_pkey PRIMARY KEY (id);


--
-- Name: friendships friendships_pkey; Type: CONSTRAINT; Schema: public; Owner: eindr
--

ALTER TABLE ONLY public.friendships
    ADD CONSTRAINT friendships_pkey PRIMARY KEY (id);


--
-- Name: label_codes label_codes_pkey; Type: CONSTRAINT; Schema: public; Owner: eindr
--

ALTER TABLE ONLY public.label_codes
    ADD CONSTRAINT label_codes_pkey PRIMARY KEY (id);


--
-- Name: label_groups label_groups_pkey; Type: CONSTRAINT; Schema: public; Owner: eindr
--

ALTER TABLE ONLY public.label_groups
    ADD CONSTRAINT label_groups_pkey PRIMARY KEY (id);


--
-- Name: language_label language_label_pkey; Type: CONSTRAINT; Schema: public; Owner: eindr
--

ALTER TABLE ONLY public.language_label
    ADD CONSTRAINT language_label_pkey PRIMARY KEY (id);


--
-- Name: languages languages_pkey; Type: CONSTRAINT; Schema: public; Owner: eindr
--

ALTER TABLE ONLY public.languages
    ADD CONSTRAINT languages_pkey PRIMARY KEY (id);


--
-- Name: ledger_direction ledger_direction_pkey; Type: CONSTRAINT; Schema: public; Owner: eindr
--

ALTER TABLE ONLY public.ledger_direction
    ADD CONSTRAINT ledger_direction_pkey PRIMARY KEY (id);


--
-- Name: ledger_entries ledger_entries_pkey; Type: CONSTRAINT; Schema: public; Owner: eindr
--

ALTER TABLE ONLY public.ledger_entries
    ADD CONSTRAINT ledger_entries_pkey PRIMARY KEY (id);


--
-- Name: login_attempts login_attempts_pkey; Type: CONSTRAINT; Schema: public; Owner: eindr
--

ALTER TABLE ONLY public.login_attempts
    ADD CONSTRAINT login_attempts_pkey PRIMARY KEY (id);


--
-- Name: note_shares note_shares_pkey; Type: CONSTRAINT; Schema: public; Owner: eindr
--

ALTER TABLE ONLY public.note_shares
    ADD CONSTRAINT note_shares_pkey PRIMARY KEY (id);


--
-- Name: notes notes_pkey; Type: CONSTRAINT; Schema: public; Owner: eindr
--

ALTER TABLE ONLY public.notes
    ADD CONSTRAINT notes_pkey PRIMARY KEY (id);


--
-- Name: priority_levels priority_levels_pkey; Type: CONSTRAINT; Schema: public; Owner: eindr
--

ALTER TABLE ONLY public.priority_levels
    ADD CONSTRAINT priority_levels_pkey PRIMARY KEY (id);


--
-- Name: reminder_notifications reminder_notifications_pkey; Type: CONSTRAINT; Schema: public; Owner: eindr
--

ALTER TABLE ONLY public.reminder_notifications
    ADD CONSTRAINT reminder_notifications_pkey PRIMARY KEY (id);


--
-- Name: reminder_shares reminder_shares_pkey; Type: CONSTRAINT; Schema: public; Owner: eindr
--

ALTER TABLE ONLY public.reminder_shares
    ADD CONSTRAINT reminder_shares_pkey PRIMARY KEY (id);


--
-- Name: reminders reminders_pkey; Type: CONSTRAINT; Schema: public; Owner: eindr
--

ALTER TABLE ONLY public.reminders
    ADD CONSTRAINT reminders_pkey PRIMARY KEY (id);


--
-- Name: subscription_plans subscription_plans_pkey; Type: CONSTRAINT; Schema: public; Owner: eindr
--

ALTER TABLE ONLY public.subscription_plans
    ADD CONSTRAINT subscription_plans_pkey PRIMARY KEY (id);


--
-- Name: timezones timezones_pkey; Type: CONSTRAINT; Schema: public; Owner: eindr
--

ALTER TABLE ONLY public.timezones
    ADD CONSTRAINT timezones_pkey PRIMARY KEY (id);


--
-- Name: api_usage_logs api_usage_logs_customer_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: eindr
--

ALTER TABLE ONLY public.api_usage_logs
    ADD CONSTRAINT api_usage_logs_customer_id_fkey FOREIGN KEY (customer_id) REFERENCES public.customers(id);


--
-- Name: chat_messages chat_messages_conversions_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: eindr
--

ALTER TABLE ONLY public.chat_messages
    ADD CONSTRAINT chat_messages_conversions_id_fkey FOREIGN KEY (conversions_id) REFERENCES public.conversions(id);


--
-- Name: conversions conversions_customer_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: eindr
--

ALTER TABLE ONLY public.conversions
    ADD CONSTRAINT conversions_customer_id_fkey FOREIGN KEY (customer_id) REFERENCES public.customers(id);


--
-- Name: customer_preferences customer_preferences_customer_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: eindr
--

ALTER TABLE ONLY public.customer_preferences
    ADD CONSTRAINT customer_preferences_customer_id_fkey FOREIGN KEY (customer_id) REFERENCES public.customers(id);


--
-- Name: customer_preferences customer_preferences_language_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: eindr
--

ALTER TABLE ONLY public.customer_preferences
    ADD CONSTRAINT customer_preferences_language_id_fkey FOREIGN KEY (language_id) REFERENCES public.languages(id);


--
-- Name: customer_sessions customer_sessions_customer_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: eindr
--

ALTER TABLE ONLY public.customer_sessions
    ADD CONSTRAINT customer_sessions_customer_id_fkey FOREIGN KEY (customer_id) REFERENCES public.customers(id);


--
-- Name: customers_devices customers_devices_customer_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: eindr
--

ALTER TABLE ONLY public.customers_devices
    ADD CONSTRAINT customers_devices_customer_id_fkey FOREIGN KEY (customer_id) REFERENCES public.customers(id);


--
-- Name: customers_profiles customers_profiles_customer_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: eindr
--

ALTER TABLE ONLY public.customers_profiles
    ADD CONSTRAINT customers_profiles_customer_id_fkey FOREIGN KEY (customer_id) REFERENCES public.customers(id);


--
-- Name: customers_profiles customers_profiles_language_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: eindr
--

ALTER TABLE ONLY public.customers_profiles
    ADD CONSTRAINT customers_profiles_language_id_fkey FOREIGN KEY (language_id) REFERENCES public.languages(id);


--
-- Name: customers_profiles customers_profiles_subscription_plan_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: eindr
--

ALTER TABLE ONLY public.customers_profiles
    ADD CONSTRAINT customers_profiles_subscription_plan_id_fkey FOREIGN KEY (subscription_plan_id) REFERENCES public.subscription_plans(id);


--
-- Name: customers_profiles customers_profiles_timezone_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: eindr
--

ALTER TABLE ONLY public.customers_profiles
    ADD CONSTRAINT customers_profiles_timezone_id_fkey FOREIGN KEY (timezone_id) REFERENCES public.timezones(id);


--
-- Name: customers customers_subscription_plan_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: eindr
--

ALTER TABLE ONLY public.customers
    ADD CONSTRAINT customers_subscription_plan_id_fkey FOREIGN KEY (subscription_plan_id) REFERENCES public.subscription_plans(id);


--
-- Name: friend_permissions friend_permissions_customer_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: eindr
--

ALTER TABLE ONLY public.friend_permissions
    ADD CONSTRAINT friend_permissions_customer_id_fkey FOREIGN KEY (customer_id) REFERENCES public.customers(id);


--
-- Name: friend_permissions friend_permissions_friend_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: eindr
--

ALTER TABLE ONLY public.friend_permissions
    ADD CONSTRAINT friend_permissions_friend_id_fkey FOREIGN KEY (friend_id) REFERENCES public.customers(id);


--
-- Name: friend_request_history friend_request_history_requested_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: eindr
--

ALTER TABLE ONLY public.friend_request_history
    ADD CONSTRAINT friend_request_history_requested_id_fkey FOREIGN KEY (requested_id) REFERENCES public.customers(id);


--
-- Name: friend_request_history friend_request_history_requester_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: eindr
--

ALTER TABLE ONLY public.friend_request_history
    ADD CONSTRAINT friend_request_history_requester_id_fkey FOREIGN KEY (requester_id) REFERENCES public.customers(id);


--
-- Name: friendships friendships_customer_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: eindr
--

ALTER TABLE ONLY public.friendships
    ADD CONSTRAINT friendships_customer_id_fkey FOREIGN KEY (customer_id) REFERENCES public.customers(id);


--
-- Name: friendships friendships_friend_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: eindr
--

ALTER TABLE ONLY public.friendships
    ADD CONSTRAINT friendships_friend_id_fkey FOREIGN KEY (friend_id) REFERENCES public.customers(id);


--
-- Name: label_codes label_codes_label_group_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: eindr
--

ALTER TABLE ONLY public.label_codes
    ADD CONSTRAINT label_codes_label_group_id_fkey FOREIGN KEY (label_group_id) REFERENCES public.label_groups(id);


--
-- Name: language_label language_label_label_code_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: eindr
--

ALTER TABLE ONLY public.language_label
    ADD CONSTRAINT language_label_label_code_id_fkey FOREIGN KEY (label_code_id) REFERENCES public.label_codes(id);


--
-- Name: language_label language_label_language_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: eindr
--

ALTER TABLE ONLY public.language_label
    ADD CONSTRAINT language_label_language_id_fkey FOREIGN KEY (language_id) REFERENCES public.languages(id);


--
-- Name: ledger_entries ledger_entries_customer_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: eindr
--

ALTER TABLE ONLY public.ledger_entries
    ADD CONSTRAINT ledger_entries_customer_id_fkey FOREIGN KEY (customer_id) REFERENCES public.customers(id);


--
-- Name: ledger_entries ledger_entries_friend_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: eindr
--

ALTER TABLE ONLY public.ledger_entries
    ADD CONSTRAINT ledger_entries_friend_id_fkey FOREIGN KEY (friend_id) REFERENCES public.customers(id);


--
-- Name: ledger_entries ledger_entries_ledger_direction_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: eindr
--

ALTER TABLE ONLY public.ledger_entries
    ADD CONSTRAINT ledger_entries_ledger_direction_id_fkey FOREIGN KEY (ledger_direction_id) REFERENCES public.ledger_direction(id);


--
-- Name: login_attempts login_attempts_customer_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: eindr
--

ALTER TABLE ONLY public.login_attempts
    ADD CONSTRAINT login_attempts_customer_id_fkey FOREIGN KEY (customer_id) REFERENCES public.customers(id);


--
-- Name: note_shares note_shares_note_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: eindr
--

ALTER TABLE ONLY public.note_shares
    ADD CONSTRAINT note_shares_note_id_fkey FOREIGN KEY (note_id) REFERENCES public.notes(id);


--
-- Name: note_shares note_shares_owner_customer_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: eindr
--

ALTER TABLE ONLY public.note_shares
    ADD CONSTRAINT note_shares_owner_customer_id_fkey FOREIGN KEY (owner_customer_id) REFERENCES public.customers(id);


--
-- Name: note_shares note_shares_shared_with_customer_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: eindr
--

ALTER TABLE ONLY public.note_shares
    ADD CONSTRAINT note_shares_shared_with_customer_id_fkey FOREIGN KEY (shared_with_customer_id) REFERENCES public.customers(id);


--
-- Name: notes notes_customer_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: eindr
--

ALTER TABLE ONLY public.notes
    ADD CONSTRAINT notes_customer_id_fkey FOREIGN KEY (customer_id) REFERENCES public.customers(id);


--
-- Name: reminder_notifications reminder_notifications_customer_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: eindr
--

ALTER TABLE ONLY public.reminder_notifications
    ADD CONSTRAINT reminder_notifications_customer_id_fkey FOREIGN KEY (customer_id) REFERENCES public.customers(id);


--
-- Name: reminder_notifications reminder_notifications_reminder_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: eindr
--

ALTER TABLE ONLY public.reminder_notifications
    ADD CONSTRAINT reminder_notifications_reminder_id_fkey FOREIGN KEY (reminder_id) REFERENCES public.reminders(id);


--
-- Name: reminder_shares reminder_shares_owner_customer_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: eindr
--

ALTER TABLE ONLY public.reminder_shares
    ADD CONSTRAINT reminder_shares_owner_customer_id_fkey FOREIGN KEY (owner_customer_id) REFERENCES public.customers(id);


--
-- Name: reminder_shares reminder_shares_reminder_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: eindr
--

ALTER TABLE ONLY public.reminder_shares
    ADD CONSTRAINT reminder_shares_reminder_id_fkey FOREIGN KEY (reminder_id) REFERENCES public.reminders(id);


--
-- Name: reminder_shares reminder_shares_shared_with_customer_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: eindr
--

ALTER TABLE ONLY public.reminder_shares
    ADD CONSTRAINT reminder_shares_shared_with_customer_id_fkey FOREIGN KEY (shared_with_customer_id) REFERENCES public.customers(id);


--
-- Name: reminders reminders_customer_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: eindr
--

ALTER TABLE ONLY public.reminders
    ADD CONSTRAINT reminders_customer_id_fkey FOREIGN KEY (customer_id) REFERENCES public.customers(id);


--
-- Name: reminders reminders_priority_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: eindr
--

ALTER TABLE ONLY public.reminders
    ADD CONSTRAINT reminders_priority_id_fkey FOREIGN KEY (priority_id) REFERENCES public.priority_levels(id);


--
-- Name: reminders reminders_timezone_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: eindr
--

ALTER TABLE ONLY public.reminders
    ADD CONSTRAINT reminders_timezone_id_fkey FOREIGN KEY (timezone_id) REFERENCES public.timezones(id);


--
-- Name: SCHEMA public; Type: ACL; Schema: -; Owner: pg_database_owner
--

GRANT USAGE ON SCHEMA public TO eindr;


--
-- PostgreSQL database dump complete
--


-- PostgreSQL DDL proposal. Never grant the inference role access to qa_private.
CREATE SCHEMA IF NOT EXISTS runtime;
CREATE SCHEMA IF NOT EXISTS qa_private;
CREATE TABLE IF NOT EXISTS runtime.telemetry (
 id BIGSERIAL PRIMARY KEY, node_id TEXT NOT NULL, boot_id TEXT NOT NULL,
 seq BIGINT NOT NULL, observed_at TIMESTAMPTZ NOT NULL, received_at TIMESTAMPTZ DEFAULT now(),
 fw_version TEXT NOT NULL, cal_version TEXT NOT NULL, payload JSONB NOT NULL,
 UNIQUE(node_id,boot_id,seq)
);
CREATE TABLE IF NOT EXISTS runtime.check_request (
 request_id UUID PRIMARY KEY, episode_id UUID NOT NULL, node_id TEXT NOT NULL,
 action TEXT NOT NULL CHECK (action IN ('RESAMPLE','COMPARE','PULSE_TEST','REQUEST_INSPECTION')),
 mode TEXT NOT NULL, policy_version TEXT NOT NULL, created_at TIMESTAMPTZ DEFAULT now(),
 expires_at TIMESTAMPTZ NOT NULL, observation_before JSONB NOT NULL,
 allowed_mask JSONB NOT NULL, limits JSONB NOT NULL, status TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS runtime.check_result (
 request_id UUID PRIMARY KEY REFERENCES runtime.check_request(request_id),
 started_at TIMESTAMPTZ, ended_at TIMESTAMPTZ, result JSONB NOT NULL
);
CREATE TABLE IF NOT EXISTS runtime.diagnosis (
 id BIGSERIAL PRIMARY KEY, episode_id UUID NOT NULL, timestamp TIMESTAMPTZ DEFAULT now(),
 state TEXT NOT NULL, cause TEXT, evidence JSONB NOT NULL, model_version TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS qa_private.ground_truth (
 episode_id UUID PRIMARY KEY, scenario TEXT NOT NULL, injection_method TEXT NOT NULL,
 injection_start TIMESTAMPTZ, reference_measurements JSONB, reviewed_by TEXT,
 released_at TIMESTAMPTZ
);
-- Deployment task: create separate roles and grant only runtime to model/ingest.
-- Runtime token/passwords belong in environment or secret management, never here.

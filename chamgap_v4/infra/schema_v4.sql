-- PostgreSQL design reference; not executed against a PostgreSQL instance here.
-- Deploy with separate admin/migration, ops writer, read-only model, and QA roles.
BEGIN;
CREATE SCHEMA IF NOT EXISTS ops;
CREATE SCHEMA IF NOT EXISTS qa;
CREATE TABLE IF NOT EXISTS ops.nodes (
 node_id text PRIMARY KEY, zone_id text NOT NULL, hardware_manifest jsonb NOT NULL,
 enabled boolean NOT NULL DEFAULT false, registered_at timestamptz NOT NULL DEFAULT now()
);
CREATE TABLE IF NOT EXISTS ops.telemetry (
 mode text NOT NULL CHECK(mode IN ('simulation','replay','shadow','live_supervised')),
 node_id text NOT NULL REFERENCES ops.nodes, boot_id text NOT NULL, seq bigint NOT NULL,
 measured_at timestamptz NOT NULL, received_at timestamptz NOT NULL,
 payload_sha256 text NOT NULL, payload jsonb NOT NULL,
 PRIMARY KEY(mode,node_id,boot_id,seq)
);
CREATE INDEX IF NOT EXISTS telemetry_node_time ON ops.telemetry(node_id,measured_at DESC);
CREATE TABLE IF NOT EXISTS ops.sessions (
 session_id text PRIMARY KEY, node_id text NOT NULL REFERENCES ops.nodes, boot_id text NOT NULL,
 opened_at timestamptz NOT NULL, closed_at timestamptz,
 config_sha256 text NOT NULL, maximum_ul bigint NOT NULL CHECK(maximum_ul>0),
 reserved_ul bigint NOT NULL DEFAULT 0 CHECK(reserved_ul>=0),
 CHECK(reserved_ul<=maximum_ul)
);
CREATE TABLE IF NOT EXISTS ops.check_requests (
 request_id text PRIMARY KEY, session_id text NOT NULL REFERENCES ops.sessions,
 body_sha256 text NOT NULL, action text NOT NULL,
 status text NOT NULL, requested_at timestamptz NOT NULL, expires_at timestamptz NOT NULL,
 dose_ul bigint NOT NULL CHECK(dose_ul>=0), body jsonb NOT NULL
);
CREATE TABLE IF NOT EXISTS ops.outbox (
 event_id text PRIMARY KEY, request_id text NOT NULL UNIQUE REFERENCES ops.check_requests,
 topic text NOT NULL, body jsonb NOT NULL, created_at timestamptz NOT NULL DEFAULT now(),
 expires_at timestamptz NOT NULL, published_at timestamptz,
 attempts integer NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS ops.check_results (
 request_id text PRIMARY KEY REFERENCES ops.check_requests,
 body_sha256 text NOT NULL, observed_at timestamptz NOT NULL, body jsonb NOT NULL
);
CREATE TABLE IF NOT EXISTS ops.decisions (
 decision_id text PRIMARY KEY, session_id text NOT NULL REFERENCES ops.sessions,
 observation_sha256 text NOT NULL, feature_schema_sha256 text NOT NULL,
 model_sha256 text, validator_sha256 text NOT NULL,
 created_at timestamptz NOT NULL, evidence jsonb NOT NULL,
 allowed_actions jsonb NOT NULL, recommended_action text, state text NOT NULL
);
CREATE TABLE IF NOT EXISTS qa.truth_sessions (
 session_id text PRIMARY KEY, family_id text NOT NULL, actual_fault text NOT NULL,
 injection_method text NOT NULL, started_at timestamptz NOT NULL, ended_at timestamptz,
 reference_manifest jsonb NOT NULL, sealed_sha256 text NOT NULL,
 released_at timestamptz
);
REVOKE ALL ON SCHEMA qa FROM PUBLIC;
REVOKE ALL ON ALL TABLES IN SCHEMA qa FROM PUBLIC;
COMMIT;
-- Create dedicated roles outside this script. Model users must not receive qa privileges.
-- INSERT...ON CONFLICT DO NOTHING is insufficient: compare hashes on duplicates.

-- 0001: tenants, API keys, model registry, cameras, events, jobs.

CREATE EXTENSION IF NOT EXISTS pgcrypto;

CREATE TABLE tenant (
    id              uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    name            text NOT NULL UNIQUE,
    -- NULL = every model; otherwise the Triton model names this tenant may call.
    model_allowlist text[],
    max_cameras     integer NOT NULL DEFAULT 100,
    created_at      timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE api_key (
    id            uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id     uuid NOT NULL REFERENCES tenant(id) ON DELETE CASCADE,
    name          text NOT NULL,
    -- Keys look like yp_<prefix>_<secret>. Only the sha256 of the secret is kept:
    -- the secret is 32 random bytes, so a fast hash is enough.
    prefix        text NOT NULL UNIQUE,
    secret_sha256 text NOT NULL,
    scopes        text[] NOT NULL DEFAULT ARRAY['infer'],
    rate_per_min  integer NOT NULL DEFAULT 600,
    created_at    timestamptz NOT NULL DEFAULT now(),
    last_used_at  timestamptz,
    revoked_at    timestamptz
);

-- One row per (model, version) in the Triton model repository.
CREATE TABLE model_version (
    id            bigserial PRIMARY KEY,
    model_name    text NOT NULL,
    version       integer NOT NULL,
    task          text NOT NULL CHECK (task IN ('detect', 'classify')),
    framework     text NOT NULL DEFAULT 'ultralytics',
    runtime       text NOT NULL DEFAULT 'tensorrt',
    precision     text NOT NULL DEFAULT 'fp16',
    imgsz         integer NOT NULL,
    max_batch     integer NOT NULL,
    names         jsonb NOT NULL,
    sha256        text,
    -- registered -> validated -> active -> retired; failed when the gate rejects it.
    status        text NOT NULL DEFAULT 'registered'
                  CHECK (status IN ('registered', 'validated', 'active', 'retired', 'failed')),
    gate_report   jsonb,
    created_at    timestamptz NOT NULL DEFAULT now(),
    activated_at  timestamptz,
    UNIQUE (model_name, version)
);
-- At most one active version per model.
CREATE UNIQUE INDEX model_version_one_active ON model_version (model_name) WHERE status = 'active';

CREATE TABLE camera (
    id           uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id    uuid NOT NULL REFERENCES tenant(id) ON DELETE CASCADE,
    name         text NOT NULL,
    url          text NOT NULL,
    enabled      boolean NOT NULL DEFAULT false,
    target_fps   real NOT NULL DEFAULT 5 CHECK (target_fps > 0 AND target_fps <= 30),
    -- Primary detector run on every sampled frame.
    model        text NOT NULL DEFAULT 'yolo26n_detection',
    -- Rules and triggered models; validated by the API (schemas.CameraConfig).
    config       jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at   timestamptz NOT NULL DEFAULT now(),
    updated_at   timestamptz NOT NULL DEFAULT now(),
    UNIQUE (tenant_id, name)
);

CREATE TABLE event (
    id           bigserial PRIMARY KEY,
    camera_id    uuid NOT NULL REFERENCES camera(id) ON DELETE CASCADE,
    ts           timestamptz NOT NULL,
    type         text NOT NULL,
    rule         text NOT NULL,
    track_id     integer,
    class_name   text,
    confidence   real,
    bbox         jsonb,
    data         jsonb NOT NULL DEFAULT '{}'::jsonb,
    -- Stream entry id, so a redelivered event is stored once.
    stream_id    text UNIQUE
);
CREATE INDEX event_camera_ts ON event (camera_id, ts DESC);

CREATE TABLE job (
    id               uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id        uuid NOT NULL REFERENCES tenant(id) ON DELETE CASCADE,
    kind             text NOT NULL DEFAULT 'video',
    status           text NOT NULL DEFAULT 'queued'
                     CHECK (status IN ('queued', 'processing', 'completed', 'failed', 'cancelled')),
    model            text NOT NULL,
    params           jsonb NOT NULL DEFAULT '{}'::jsonb,
    input_key        text NOT NULL,
    output_key       text,
    result_key       text,
    summary          jsonb,
    total_frames     integer,
    processed_frames integer NOT NULL DEFAULT 0,
    attempts         integer NOT NULL DEFAULT 0,
    error            text,
    created_at       timestamptz NOT NULL DEFAULT now(),
    started_at       timestamptz,
    finished_at      timestamptz
);
CREATE INDEX job_tenant_created ON job (tenant_id, created_at DESC);

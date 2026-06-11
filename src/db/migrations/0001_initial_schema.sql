-- ---------------------------------------------------------------------------
-- Schema inicial do agentic-tabular-pipeline (seção 11 do documento de apoio).
-- Usa JSONB para preservar flexibilidade e rastreabilidade dos eventos agentivos,
-- configurações, perfis, métricas e relatórios.
-- ---------------------------------------------------------------------------

CREATE EXTENSION IF NOT EXISTS "pgcrypto";  -- gen_random_uuid()

-- Configuração do experimento
CREATE TABLE IF NOT EXISTS experiments (
    id             UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name           TEXT NOT NULL,
    task_type      TEXT NOT NULL CHECK (task_type IN ('classification', 'regression', 'anomaly')),
    target_column  TEXT,
    primary_metric TEXT NOT NULL,
    status         TEXT NOT NULL DEFAULT 'created',
    created_at     TIMESTAMPTZ DEFAULT now(),
    config         JSONB NOT NULL
);

-- Datasets: para bases grandes guardar metadados, hash, schema e perfil (não a base inteira)
CREATE TABLE IF NOT EXISTS datasets (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    experiment_id       UUID REFERENCES experiments(id) ON DELETE CASCADE,
    name                TEXT NOT NULL,
    source_type         TEXT NOT NULL,
    source_uri          TEXT,
    content_hash        TEXT,
    schema_json         JSONB,
    profile_json        JSONB,
    quality_report_json JSONB,
    created_at          TIMESTAMPTZ DEFAULT now()
);

-- Eventos agentivos: toda decisão relevante gera um evento persistido (RNF03 - Auditabilidade)
CREATE TABLE IF NOT EXISTS agent_events (
    id            BIGSERIAL PRIMARY KEY,
    experiment_id UUID REFERENCES experiments(id) ON DELETE CASCADE,
    agent_name    TEXT NOT NULL,
    event_type    TEXT NOT NULL,
    input_json    JSONB,
    output_json   JSONB,
    rationale     TEXT,
    created_at    TIMESTAMPTZ DEFAULT now()
);

-- Execuções do pipeline: cada uma tem id único, versão de código, versão de dados e seed
CREATE TABLE IF NOT EXISTS pipeline_runs (
    id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    experiment_id UUID REFERENCES experiments(id) ON DELETE CASCADE,
    code_version  TEXT,
    data_version  TEXT,
    seed          INTEGER,
    status        TEXT NOT NULL DEFAULT 'running',
    started_at    TIMESTAMPTZ DEFAULT now(),
    finished_at   TIMESTAMPTZ,
    metrics_json  JSONB
);

-- Resultados por modelo/fold (base para o ranking — consultas da seção 23)
CREATE TABLE IF NOT EXISTS model_results (
    id              BIGSERIAL PRIMARY KEY,
    run_id          UUID REFERENCES pipeline_runs(id) ON DELETE CASCADE,
    model_name      TEXT NOT NULL,
    fold            INTEGER,
    hyperparameters JSONB,
    metrics_json    JSONB,
    fit_seconds     DOUBLE PRECISION,
    artifact_path   TEXT,
    artifact_hash   TEXT,
    created_at      TIMESTAMPTZ DEFAULT now()
);

-- Relatório técnico final gerado automaticamente
CREATE TABLE IF NOT EXISTS reports (
    id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    experiment_id UUID REFERENCES experiments(id) ON DELETE CASCADE,
    run_id        UUID REFERENCES pipeline_runs(id) ON DELETE CASCADE,
    format        TEXT NOT NULL DEFAULT 'markdown',
    content       TEXT,
    summary_json  JSONB,
    created_at    TIMESTAMPTZ DEFAULT now()
);

-- Índices para consultas de auditoria e ranking
CREATE INDEX IF NOT EXISTS idx_agent_events_experiment ON agent_events (experiment_id, created_at);
CREATE INDEX IF NOT EXISTS idx_model_results_run        ON model_results (run_id, model_name);
CREATE INDEX IF NOT EXISTS idx_model_results_metrics    ON model_results USING gin (metrics_json);

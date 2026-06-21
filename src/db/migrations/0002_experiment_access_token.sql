-- ---------------------------------------------------------------------------
-- Confidencialidade por experimento: cada experimento passa a ter um token de
-- acesso (capability token). Persistimos apenas o hash SHA-256 do token; o valor
-- em claro é exibido ao usuário uma única vez, na criação.
--
-- Experimentos criados antes desta migração ficam com access_token_hash nulo e,
-- portanto, inacessíveis pela API (sem token válido). Trate-os pelo painel
-- /admin ou atribua um token manualmente, se necessário.
-- ---------------------------------------------------------------------------

ALTER TABLE experiments ADD COLUMN IF NOT EXISTS access_token_hash TEXT;

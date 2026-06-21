# agentic-tabular-pipeline

Sistema agentivo *open source* para um pipeline completo de Aprendizado de Máquina sobre **dados tabulares**.
Trabalho em grupo da disciplina **MAQ020 - Inteligência Artificial**.

O sistema recebe uma base tabular e uma definição mínima da tarefa e conduz, de forma assistida e auditável,
as etapas de perfilamento, limpeza, engenharia de atributos, particionamento, treinamento cruzado de múltiplas
famílias de modelos (incluindo ao menos um autoencoder), avaliação comparativa e geração de relatório técnico.
Todas as interações dos agentes, configurações, decisões, métricas e metadados são persistidos em
**PostgreSQL** (preferencialmente em campos **JSONB**).

> Foco avaliativo: **robustez metodológica, reprodutibilidade e rastreabilidade** — não a maior acurácia absoluta.

## Fluxo agentivo

| # | Agente | Função |
|---|--------|--------|
| 1 | Formulação do Problema | Define tarefa, variável-alvo, métrica primária, critério de sucesso e restrições. |
| 2 | Ingestão e Perfilamento | Lê a base, infere schema, tipos, distribuições, cardinalidade, faltantes e desbalanceamento. |
| 3 | Qualidade e Limpeza | Propõe e executa limpeza reprodutível (faltantes, duplicatas, outliers, categorias raras). |
| 4 | Engenharia de Atributos | Transformações de numéricas, categóricas, datas e texto curto; evita *leakage*. |
| 5 | Split e Validação | Holdout, k-fold, stratified k-fold, group split ou time split. |
| 6 | Model Zoo (treino cruzado) | Treina famílias de modelos clássicos; registra folds, seeds, hiperparâmetros e métricas. |
| 7 | Autoencoders | Autoencoder tabular para representação, denoising ou detecção de anomalias. |
| 8 | Avaliação e Seleção | Compara métricas por fold e recomenda modelo por critério claro. |
| 9 | Relator e Auditor | Gera relatório técnico, *model card* e logs auditáveis. |

## Estrutura do repositório

```
agentic-tabular-pipeline/
  README.md
  LICENSE
  docker-compose.yml
  .env.example
  pyproject.toml
  data/
    raw/                 # não versionar dados sensíveis
    processed/
  configs/
    experiment_example.yaml
    model_zoo.yaml
  src/
    agents/              # um módulo por papel lógico do pipeline
    pipelines/           # preprocessing / training / evaluation
    db/                  # modelos ORM + migrations SQL
    api/                 # FastAPI
    ui/                  # Streamlit
  notebooks/
    demo_end_to_end.ipynb
  tests/
  reports/
    sample_report.md
```

## Como executar

### Com Docker Compose (recomendado)

```bash
cp .env.example .env          # ajuste as credenciais se necessário
docker compose up --build
```

Isso sobe o PostgreSQL, aplica as migrations e expõe a API (FastAPI, porta 8000) e a
interface (Streamlit, porta 8501). A UI conversa com a API via `API_URL`.

### Ambiente Python local

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .[dev]
# suba um PostgreSQL e exporte a variável de conexão:
export DATABASE_URL=postgresql://postgres:postgres@localhost:5432/agentic
psql "$DATABASE_URL" -f src/db/migrations/0001_initial_schema.sql
# gere a base sintética de demonstração (ver data/README.md):
python -m scripts.generate_demo_dataset
```

### Rodar um experimento

```bash
# pela linha de comando (persiste no PostgreSQL se DATABASE_URL estiver definido):
python -m src.pipelines.training --config configs/experiment_example.yaml

# ou pela API + interface web:
uvicorn src.api.main:app --reload          # backend em http://localhost:8000
API_URL=http://localhost:8000 streamlit run src/ui/app.py   # UI em http://localhost:8501
```

O notebook `notebooks/demo_end_to_end.ipynb` demonstra o fluxo completo e as
consultas de auditoria da seção 23 (roda em memória sem `DATABASE_URL`).

### Confidencialidade por token

Cada experimento recebe, na criação, um **token de acesso** (exibido uma única
vez). A API não oferece listagem aberta: para reabrir um experimento informa-se o
token (`POST /experiments/resolve`), enviado depois no cabeçalho
`X-Experiment-Token` das chamadas àquele experimento. No banco guarda-se apenas o
hash SHA-256 do token (ver `src/api/security.py`). O painel `/admin` continua sob
login (`ADMIN_USER`/`ADMIN_PASSWORD`) para o operador. Experimentos criados antes
desta versão ficam sem token e só são acessíveis via `/admin`.

## Stack open source

Python · pandas · scikit-learn · PyTorch · Optuna · FastAPI · Streamlit · PostgreSQL/JSONB · Docker.
As versões exatas são fixadas em `pyproject.toml`.

## Licença

Distribuído sob a licença **MIT** — ver [LICENSE](LICENSE).

"""API HTTP do pipeline (FastAPI).

Expõe endpoints para cadastrar experimentos, disparar execuções e consultar o
histórico de eventos agentivos e o ranking de modelos (consultas da seção 23).

O acesso a dados é isolado em :class:`~src.api.store.ExperimentStore` e injetado
via ``Depends(get_store)``, o que permite substituí-lo por um *fake* em memória
nos testes (``app.dependency_overrides``) sem exigir um PostgreSQL real.
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Any

from fastapi import BackgroundTasks, Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel

from src.api.store import ExperimentStore

app = FastAPI(title="agentic-tabular-pipeline", version="0.1.0")


class ExperimentCreate(BaseModel):
    name: str
    task_type: str
    target_column: str | None = None
    primary_metric: str
    config: dict[str, Any]


class TokenIn(BaseModel):
    token: str


@lru_cache(maxsize=1)
def _session_factory():
    """Fábrica de sessões a partir de ``DATABASE_URL`` (memorizada)."""
    from src.db.models import get_session_factory

    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        raise RuntimeError("DATABASE_URL não definido — API requer PostgreSQL.")
    return get_session_factory(database_url)


def get_store() -> ExperimentStore:
    """Dependência que fornece o ``ExperimentStore`` aos endpoints.

    Sobrescrita nos testes via ``app.dependency_overrides[get_store]``.
    """
    return ExperimentStore(_session_factory())


def require_access(
    experiment_id: str,
    x_experiment_token: str | None = Header(default=None),
    store: ExperimentStore = Depends(get_store),
) -> str:
    """Autoriza o acesso a um experimento pelo token de capacidade.

    Lê o token do cabeçalho ``X-Experiment-Token`` e o confere com o hash
    persistido. Falha com 401 — sem distinguir "token errado" de "experimento
    inexistente" — para não permitir enumeração de experimentos.
    """
    if not store.verify_access(experiment_id, x_experiment_token):
        raise HTTPException(
            status_code=401,
            detail="Token de acesso ausente ou inválido para este experimento.",
        )
    return experiment_id


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/experiments/resolve")
def resolve_experiment(
    payload: TokenIn, store: ExperimentStore = Depends(get_store)
) -> dict[str, Any]:
    """Recupera um experimento a partir do seu token (sem listagem aberta)."""
    exp = store.resolve_by_token(payload.token)
    if exp is None:
        raise HTTPException(status_code=404, detail="Token não corresponde a nenhum experimento.")
    return exp


@app.get("/experiments/{experiment_id}")
def get_experiment(
    experiment_id: str = Depends(require_access),
    store: ExperimentStore = Depends(get_store),
) -> dict[str, Any]:
    exp = store.get_experiment(experiment_id)
    if exp is None:
        raise HTTPException(status_code=404, detail="Experimento não encontrado.")
    return exp


@app.post("/experiments", status_code=201)
def create_experiment(
    payload: ExperimentCreate, store: ExperimentStore = Depends(get_store)
) -> dict[str, Any]:
    created = store.create_experiment(
        name=payload.name,
        task_type=payload.task_type,
        target_column=payload.target_column,
        primary_metric=payload.primary_metric,
        config=payload.config,
    )
    # ``access_token`` é devolvido uma única vez — o cliente deve guardá-lo.
    return {**created, "status": "created"}


@app.post("/experiments/{experiment_id}/run", status_code=202)
def run_experiment_endpoint(
    background_tasks: BackgroundTasks,
    experiment_id: str = Depends(require_access),
    store: ExperimentStore = Depends(get_store),
) -> dict[str, Any]:
    # Execução assíncrona: o pipeline pode levar minutos; respondemos 202 e
    # processamos em segundo plano, atualizando o status no banco ao final.
    background_tasks.add_task(store.execute, experiment_id)
    return {"experiment_id": experiment_id, "status": "running"}


@app.get("/experiments/{experiment_id}/events")
def list_events(
    experiment_id: str = Depends(require_access),
    store: ExperimentStore = Depends(get_store),
) -> list[dict[str, Any]]:
    return store.list_events(experiment_id)


@app.get("/experiments/{experiment_id}/runs")
def list_runs(
    experiment_id: str = Depends(require_access),
    store: ExperimentStore = Depends(get_store),
) -> list[dict[str, Any]]:
    return store.list_runs(experiment_id)


@app.get("/runs/{run_id}/ranking")
def model_ranking(
    run_id: str,
    x_experiment_token: str | None = Header(default=None),
    store: ExperimentStore = Depends(get_store),
) -> list[dict[str, Any]]:
    experiment_id = store.experiment_id_for_run(run_id)
    if experiment_id is None or not store.verify_access(experiment_id, x_experiment_token):
        raise HTTPException(
            status_code=401,
            detail="Token de acesso ausente ou inválido para este experimento.",
        )
    return store.ranking(run_id)


@app.get("/experiments/{experiment_id}/report")
def get_report(
    experiment_id: str = Depends(require_access),
    store: ExperimentStore = Depends(get_store),
) -> dict[str, Any]:
    report = store.get_report(experiment_id)
    if report is None:
        raise HTTPException(status_code=404, detail="Relatório ainda não gerado.")
    return report


@app.get("/experiments/{experiment_id}/profile")
def get_profile(
    experiment_id: str = Depends(require_access),
    store: ExperimentStore = Depends(get_store),
) -> dict[str, Any]:
    dataset = store.get_dataset(experiment_id)
    if dataset is None:
        raise HTTPException(status_code=404, detail="Perfil ainda não disponível.")
    return dataset


# ---------------------------------------------------------------------------
# Painel de administração (estilo Django Admin) em /admin.
#
# Montado apenas quando há PostgreSQL configurado: o SQLAdmin exige a engine real
# já na construção. Sem DATABASE_URL — como no ambiente de testes — o painel não é
# montado e a suíte segue rodando com o store em memória.
# ---------------------------------------------------------------------------
if os.environ.get("DATABASE_URL"):
    from src.api.admin import setup_admin
    from src.db.models import get_engine

    setup_admin(app, get_engine())

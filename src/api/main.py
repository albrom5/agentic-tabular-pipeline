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

from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException
from pydantic import BaseModel

from src.api.store import ExperimentStore

app = FastAPI(title="agentic-tabular-pipeline", version="0.1.0")


class ExperimentCreate(BaseModel):
    name: str
    task_type: str
    target_column: str | None = None
    primary_metric: str
    config: dict[str, Any]


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


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/experiments")
def list_experiments(
    store: ExperimentStore = Depends(get_store),
) -> list[dict[str, Any]]:
    return store.list_experiments()


@app.get("/experiments/{experiment_id}")
def get_experiment(
    experiment_id: str, store: ExperimentStore = Depends(get_store)
) -> dict[str, Any]:
    exp = store.get_experiment(experiment_id)
    if exp is None:
        raise HTTPException(status_code=404, detail="Experimento não encontrado.")
    return exp


@app.post("/experiments", status_code=201)
def create_experiment(
    payload: ExperimentCreate, store: ExperimentStore = Depends(get_store)
) -> dict[str, Any]:
    experiment_id = store.create_experiment(
        name=payload.name,
        task_type=payload.task_type,
        target_column=payload.target_column,
        primary_metric=payload.primary_metric,
        config=payload.config,
    )
    return {"experiment_id": experiment_id, "status": "created"}


@app.post("/experiments/{experiment_id}/run", status_code=202)
def run_experiment_endpoint(
    experiment_id: str,
    background_tasks: BackgroundTasks,
    store: ExperimentStore = Depends(get_store),
) -> dict[str, Any]:
    if store.get_experiment(experiment_id) is None:
        raise HTTPException(status_code=404, detail="Experimento não encontrado.")
    # Execução assíncrona: o pipeline pode levar minutos; respondemos 202 e
    # processamos em segundo plano, atualizando o status no banco ao final.
    background_tasks.add_task(store.execute, experiment_id)
    return {"experiment_id": experiment_id, "status": "running"}


@app.get("/experiments/{experiment_id}/events")
def list_events(
    experiment_id: str, store: ExperimentStore = Depends(get_store)
) -> list[dict[str, Any]]:
    if store.get_experiment(experiment_id) is None:
        raise HTTPException(status_code=404, detail="Experimento não encontrado.")
    return store.list_events(experiment_id)


@app.get("/experiments/{experiment_id}/runs")
def list_runs(
    experiment_id: str, store: ExperimentStore = Depends(get_store)
) -> list[dict[str, Any]]:
    if store.get_experiment(experiment_id) is None:
        raise HTTPException(status_code=404, detail="Experimento não encontrado.")
    return store.list_runs(experiment_id)


@app.get("/runs/{run_id}/ranking")
def model_ranking(
    run_id: str, store: ExperimentStore = Depends(get_store)
) -> list[dict[str, Any]]:
    return store.ranking(run_id)


@app.get("/experiments/{experiment_id}/report")
def get_report(
    experiment_id: str, store: ExperimentStore = Depends(get_store)
) -> dict[str, Any]:
    report = store.get_report(experiment_id)
    if report is None:
        raise HTTPException(status_code=404, detail="Relatório ainda não gerado.")
    return report


@app.get("/experiments/{experiment_id}/profile")
def get_profile(
    experiment_id: str, store: ExperimentStore = Depends(get_store)
) -> dict[str, Any]:
    dataset = store.get_dataset(experiment_id)
    if dataset is None:
        raise HTTPException(status_code=404, detail="Perfil ainda não disponível.")
    return dataset

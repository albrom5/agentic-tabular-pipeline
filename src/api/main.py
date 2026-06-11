"""API HTTP do pipeline (FastAPI).

Expõe endpoints para cadastrar experimentos, disparar execuções e consultar o
histórico de eventos agentivos e o ranking de modelos.
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="agentic-tabular-pipeline", version="0.1.0")


class ExperimentCreate(BaseModel):
    name: str
    task_type: str
    target_column: str | None = None
    primary_metric: str
    config: dict[str, Any]


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/experiments", status_code=201)
def create_experiment(payload: ExperimentCreate) -> dict[str, Any]:
    # TODO: persistir o experimento (src/db/models.py) e retornar o id.
    raise NotImplementedError


@app.post("/experiments/{experiment_id}/run")
def run_experiment(experiment_id: str) -> dict[str, Any]:
    # TODO: disparar src.pipelines.training.run_experiment de forma assíncrona.
    raise NotImplementedError


@app.get("/experiments/{experiment_id}/events")
def list_events(experiment_id: str) -> list[dict[str, Any]]:
    # TODO: retornar o histórico agentivo ordenado por created_at (seção 23).
    raise NotImplementedError


@app.get("/runs/{run_id}/ranking")
def model_ranking(run_id: str) -> list[dict[str, Any]]:
    # TODO: ranking agregado de modelos pela métrica primária (seção 23).
    raise NotImplementedError

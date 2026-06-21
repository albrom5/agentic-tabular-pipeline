"""Acesso a dados da API: cadastro de experimentos, execução e consultas.

Isola toda a interação com o PostgreSQL atrás de uma classe injetável, de modo
que os endpoints (``src/api/main.py``) possam ser testados com um *fake* em
memória, sem exigir um banco real. As consultas de leitura espelham as da
seção 23 do documento (histórico agentivo e ranking de modelos).
"""

from __future__ import annotations

import statistics
import uuid
from collections import defaultdict
from typing import Any

from sqlalchemy import select

from src.api.security import generate_token, hash_token, verify_token
from src.db.models import (
    AgentEvent,
    Dataset,
    Experiment,
    ModelResult,
    PipelineRun,
    Report,
)
from src.pipelines.persistence import RunRecorder
from src.pipelines.training import run_experiment

# Métricas em que menor é melhor (define a ordenação do ranking).
_MINIMIZE_METRICS = {"rmse", "mae", "mse", "mape", "logloss", "log_loss", "brier"}


class ExperimentStore:
    """Operações de persistência/consulta usadas pelos endpoints da API."""

    def __init__(self, session_factory) -> None:
        self._sf = session_factory
        self._recorder = RunRecorder(session_factory)

    # -- escrita ---------------------------------------------------------------

    def create_experiment(
        self,
        *,
        name: str,
        task_type: str,
        target_column: str | None,
        primary_metric: str,
        config: dict[str, Any],
    ) -> dict[str, str]:
        """Cadastra o experimento e devolve seu id e o token de acesso.

        O token (em claro) só existe aqui e na resposta ao usuário: no banco
        guardamos apenas o hash (RNF07/confidencialidade). Ver
        :mod:`src.api.security`.
        """
        token = generate_token()
        experiment_id = self._recorder.create_experiment(
            name=name,
            task_type=task_type,
            target_column=target_column,
            primary_metric=primary_metric,
            config=config,
            access_token_hash=hash_token(token),
        )
        # create_experiment marca como 'running'; aqui o experimento apenas
        # nasce cadastrado, então voltamos o status para 'created'.
        self._recorder.set_experiment_status(
            experiment_id=experiment_id, status="created"
        )
        return {"experiment_id": str(experiment_id), "access_token": token}

    def verify_access(self, experiment_id: str, token: str | None) -> bool:
        """True se ``token`` corresponde ao experimento (e ele existe)."""
        try:
            eid = uuid.UUID(experiment_id)
        except (ValueError, AttributeError):
            return False
        with self._sf() as session:
            exp = session.get(Experiment, eid)
            if exp is None:
                return False
            return verify_token(token, exp.access_token_hash)

    def resolve_by_token(self, token: str | None) -> dict[str, Any] | None:
        """Localiza um experimento pelo token, sem enumerar os demais.

        Substitui a antiga listagem aberta: o usuário informa o token e recupera
        apenas o experimento correspondente.
        """
        if not token:
            return None
        with self._sf() as session:
            exp = session.execute(
                select(Experiment).where(
                    Experiment.access_token_hash == hash_token(token)
                )
            ).scalar_one_or_none()
            if exp is None:
                return None
            return {
                "id": str(exp.id),
                "name": exp.name,
                "task_type": exp.task_type,
                "target_column": exp.target_column,
                "primary_metric": exp.primary_metric,
                "status": exp.status,
                "created_at": _iso(exp.created_at),
                "config": exp.config,
            }

    def experiment_id_for_run(self, run_id: str) -> str | None:
        """Id do experimento dono de um run (para autorizar o ranking)."""
        try:
            rid = uuid.UUID(run_id)
        except (ValueError, AttributeError):
            return None
        with self._sf() as session:
            run = session.get(PipelineRun, rid)
            return str(run.experiment_id) if run is not None else None

    def execute(self, experiment_id: str) -> None:
        """Executa o pipeline de ponta a ponta para um experimento cadastrado.

        Pensada para rodar em segundo plano (``BackgroundTasks``). Em caso de
        falha, marca o experimento como ``failed`` e propaga a exceção para o log.
        """
        exp = self.get_experiment(experiment_id)
        if exp is None:
            raise ValueError(f"Experimento {experiment_id} não encontrado.")
        eid = uuid.UUID(experiment_id)
        try:
            run_experiment(
                exp["config"],
                recorder=self._recorder,
                event_sink=self._recorder.event_sink,
                experiment_id=eid,
            )
        except Exception:
            self._recorder.set_experiment_status(experiment_id=eid, status="failed")
            raise

    # -- leitura ---------------------------------------------------------------
    # Não há listagem aberta de experimentos por confidencialidade: o acesso se
    # dá por token (ver ``resolve_by_token``), não por enumeração.

    def get_experiment(self, experiment_id: str) -> dict[str, Any] | None:
        try:
            eid = uuid.UUID(experiment_id)
        except (ValueError, AttributeError):
            return None
        with self._sf() as session:
            exp = session.get(Experiment, eid)
            if exp is None:
                return None
            return {
                "id": str(exp.id),
                "name": exp.name,
                "task_type": exp.task_type,
                "target_column": exp.target_column,
                "primary_metric": exp.primary_metric,
                "status": exp.status,
                "created_at": _iso(exp.created_at),
                "config": exp.config,
            }

    def list_events(self, experiment_id: str) -> list[dict[str, Any]]:
        """Histórico agentivo ordenado por ``created_at`` (seção 23)."""
        eid = uuid.UUID(experiment_id)
        with self._sf() as session:
            rows = session.execute(
                select(AgentEvent)
                .where(AgentEvent.experiment_id == eid)
                .order_by(AgentEvent.created_at, AgentEvent.id)
            ).scalars().all()
            return [
                {
                    "id": ev.id,
                    "agent_name": ev.agent_name,
                    "event_type": ev.event_type,
                    "rationale": ev.rationale,
                    "input_json": ev.input_json,
                    "output_json": ev.output_json,
                    "created_at": _iso(ev.created_at),
                }
                for ev in rows
            ]

    def list_runs(self, experiment_id: str) -> list[dict[str, Any]]:
        eid = uuid.UUID(experiment_id)
        with self._sf() as session:
            rows = session.execute(
                select(PipelineRun)
                .where(PipelineRun.experiment_id == eid)
                .order_by(PipelineRun.started_at)
            ).scalars().all()
            return [
                {
                    "id": str(r.id),
                    "status": r.status,
                    "seed": r.seed,
                    "code_version": r.code_version,
                    "data_version": r.data_version,
                    "started_at": _iso(r.started_at),
                    "finished_at": _iso(r.finished_at),
                    "metrics_json": r.metrics_json,
                }
                for r in rows
            ]

    def ranking(self, run_id: str) -> list[dict[str, Any]]:
        """Ranking agregado por modelo na métrica primária do experimento.

        Agrega as linhas por fold (``fold IS NOT NULL``) calculando média, desvio
        e número de folds — equivalente à consulta de ranking da seção 23.
        """
        rid = uuid.UUID(run_id)
        with self._sf() as session:
            run = session.get(PipelineRun, rid)
            if run is None:
                return []
            exp = session.get(Experiment, run.experiment_id)
            primary = exp.primary_metric if exp is not None else None
            rows = session.execute(
                select(ModelResult).where(
                    ModelResult.run_id == rid, ModelResult.fold.isnot(None)
                )
            ).scalars().all()
        triples = [(r.model_name, r.metrics_json, r.fit_seconds) for r in rows]
        return aggregate_ranking(triples, primary)

    def get_report(self, experiment_id: str) -> dict[str, Any] | None:
        """Relatório técnico mais recente do experimento (Markdown + resumo)."""
        eid = uuid.UUID(experiment_id)
        with self._sf() as session:
            report = session.execute(
                select(Report)
                .where(Report.experiment_id == eid)
                .order_by(Report.created_at.desc())
            ).scalars().first()
            if report is None:
                return None
            return {
                "format": report.format,
                "content": report.content,
                "summary_json": report.summary_json,
                "created_at": _iso(report.created_at),
            }

    def get_dataset(self, experiment_id: str) -> dict[str, Any] | None:
        """Perfil e relatório de qualidade persistidos do dataset (mais recente)."""
        eid = uuid.UUID(experiment_id)
        with self._sf() as session:
            ds = session.execute(
                select(Dataset)
                .where(Dataset.experiment_id == eid)
                .order_by(Dataset.created_at.desc())
            ).scalars().first()
            if ds is None:
                return None
            return {
                "name": ds.name,
                "source_type": ds.source_type,
                "source_uri": ds.source_uri,
                "content_hash": ds.content_hash,
                "profile_json": ds.profile_json,
                "quality_report_json": ds.quality_report_json,
            }


def aggregate_ranking(
    rows: list[tuple[str, dict[str, Any] | None, float | None]],
    primary_metric: str | None,
) -> list[dict[str, Any]]:
    """Agrega métricas por modelo (média, desvio, nº de folds) e ordena.

    ``rows`` são triplas ``(model_name, metrics_json, fit_seconds)`` por fold.
    A ordenação respeita a direção da métrica (menor é melhor para erros).
    Função pura, para ser testável sem um banco real.
    """
    scores: dict[str, list[float]] = defaultdict(list)
    times: dict[str, list[float]] = defaultdict(list)
    for name, metrics, fit_seconds in rows:
        value = (metrics or {}).get(primary_metric)
        if value is not None:
            scores[name].append(float(value))
            if fit_seconds is not None:
                times[name].append(float(fit_seconds))

    ranking = [
        {
            "model_name": name,
            "primary_metric": primary_metric,
            "mean": statistics.fmean(vals),
            "std": statistics.pstdev(vals) if len(vals) > 1 else 0.0,
            "folds": len(vals),
            "mean_fit_seconds": (
                statistics.fmean(times[name]) if times[name] else None
            ),
        }
        for name, vals in scores.items()
    ]
    minimize = (primary_metric or "").lower() in _MINIMIZE_METRICS
    ranking.sort(key=lambda d: d["mean"], reverse=not minimize)
    return ranking


def _iso(value: Any) -> str | None:
    return value.isoformat() if value is not None else None

"""Persistência de uma execução do pipeline no PostgreSQL.

Concentra a escrita das tabelas ``experiments``, ``pipeline_runs``, ``datasets``,
``model_results`` e ``reports`` (seção 11). Os eventos agentivos são gravados à
parte pelo :class:`~src.db.models.EventSink`, exposto aqui em ``self.event_sink``.

O orquestrador (``run_experiment``) recebe um ``RunRecorder`` por injeção; em
testes pode-se passar um *fake* com a mesma interface, evitando um banco real.
"""

from __future__ import annotations

import datetime
import uuid
from typing import Any

from src.db.models import (
    AgentEvent,  # noqa: F401 - documenta a tabela usada pelo EventSink
    Dataset,
    EventSink,
    Experiment,
    ModelResult,
    PipelineRun,
    Report,
)


class RunRecorder:
    """Grava o estado de um experimento/execução usando um ``session_factory``."""

    def __init__(self, session_factory) -> None:
        self._sf = session_factory
        self.event_sink = EventSink(session_factory)

    # -- experimento e execução ------------------------------------------------

    def create_experiment(
        self,
        *,
        name: str,
        task_type: str,
        target_column: str | None,
        primary_metric: str,
        config: dict[str, Any],
        experiment_id: uuid.UUID | None = None,
        access_token_hash: str | None = None,
    ) -> uuid.UUID:
        """Cria (ou atualiza) o experimento e o marca como ``running``.

        Quando ``experiment_id`` é informado — caso da API, que cadastra o
        experimento antes de disparar a execução — a linha existente é
        atualizada em vez de criar uma duplicata. O ``access_token_hash`` só é
        gravado na criação inicial: numa reexecução ele chega ``None`` e o hash
        já persistido é preservado (não sobrescrevemos a chave de acesso).
        """
        experiment_id = experiment_id or uuid.uuid4()
        with self._sf() as session:
            exp = session.get(Experiment, experiment_id)
            if exp is None:
                session.add(
                    Experiment(
                        id=experiment_id,
                        name=name,
                        task_type=task_type,
                        target_column=target_column,
                        primary_metric=primary_metric,
                        status="running",
                        config=config,
                        access_token_hash=access_token_hash,
                    )
                )
            else:
                exp.name = name
                exp.task_type = task_type
                exp.target_column = target_column
                exp.primary_metric = primary_metric
                exp.status = "running"
                exp.config = config
                if access_token_hash is not None:
                    exp.access_token_hash = access_token_hash
            session.commit()
        return experiment_id

    def create_run(
        self,
        *,
        experiment_id: uuid.UUID,
        seed: int | None,
        code_version: str | None,
        data_version: str | None,
    ) -> uuid.UUID:
        run_id = uuid.uuid4()
        with self._sf() as session:
            session.add(
                PipelineRun(
                    id=run_id,
                    experiment_id=experiment_id,
                    code_version=code_version,
                    data_version=data_version,
                    seed=seed,
                    status="running",
                )
            )
            session.commit()
        return run_id

    def finish_run(
        self, *, run_id: uuid.UUID, status: str, metrics_json: dict[str, Any] | None
    ) -> None:
        with self._sf() as session:
            run = session.get(PipelineRun, run_id)
            if run is not None:
                run.status = status
                run.finished_at = datetime.datetime.now(datetime.timezone.utc)
                run.metrics_json = metrics_json
            session.commit()

    def set_experiment_status(self, *, experiment_id: uuid.UUID, status: str) -> None:
        with self._sf() as session:
            exp = session.get(Experiment, experiment_id)
            if exp is not None:
                exp.status = status
            session.commit()

    # -- artefatos das etapas --------------------------------------------------

    def save_dataset(
        self,
        *,
        experiment_id: uuid.UUID,
        name: str,
        source_type: str,
        source_uri: str | None,
        content_hash: str | None,
        schema_json: dict[str, Any] | None,
        profile_json: dict[str, Any] | None,
        quality_report_json: dict[str, Any] | None,
    ) -> None:
        with self._sf() as session:
            session.add(
                Dataset(
                    experiment_id=experiment_id,
                    name=name,
                    source_type=source_type,
                    source_uri=source_uri,
                    content_hash=content_hash,
                    schema_json=schema_json,
                    profile_json=profile_json,
                    quality_report_json=quality_report_json,
                )
            )
            session.commit()

    def save_model_results(
        self,
        *,
        run_id: uuid.UUID,
        results: list[dict[str, Any]],
        artifacts: dict[str, dict[str, Any]] | None = None,
    ) -> int:
        """Grava uma linha por modelo/fold e uma linha agregada (fold = NULL)."""
        artifacts = artifacts or {}
        rows = 0
        with self._sf() as session:
            for res in results:
                name = res.get("model_name")
                params = res.get("hyperparameters")
                art = artifacts.get(name, {})
                for fm in res.get("fold_metrics", []):
                    session.add(
                        ModelResult(
                            run_id=run_id,
                            model_name=name,
                            fold=fm.get("fold"),
                            hyperparameters=params,
                            metrics_json=fm.get("metrics"),
                            fit_seconds=fm.get("fit_seconds"),
                        )
                    )
                    rows += 1
                # Linha agregada (média por fold), com eventual artefato.
                if res.get("metrics_mean") is not None or art:
                    session.add(
                        ModelResult(
                            run_id=run_id,
                            model_name=name,
                            fold=None,
                            hyperparameters=params,
                            metrics_json=res.get("metrics_mean"),
                            artifact_path=art.get("artifact_uri"),
                            artifact_hash=art.get("artifact_hash"),
                        )
                    )
                    rows += 1
            session.commit()
        return rows

    def save_report(
        self,
        *,
        experiment_id: uuid.UUID,
        run_id: uuid.UUID,
        content: str,
        summary_json: dict[str, Any] | None,
        fmt: str = "markdown",
    ) -> None:
        with self._sf() as session:
            session.add(
                Report(
                    experiment_id=experiment_id,
                    run_id=run_id,
                    format=fmt,
                    content=content,
                    summary_json=summary_json,
                )
            )
            session.commit()

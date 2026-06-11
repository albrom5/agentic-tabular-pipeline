"""Modelos ORM (SQLAlchemy) espelhando o schema de `migrations/0001_initial_schema.sql`.

Define também o :class:`EventSink`, usado pelos agentes para persistir eventos
agentivos de forma auditável (RNF03).
"""

from __future__ import annotations

import os
import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import (
    BigInteger,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    create_engine,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker


class Base(DeclarativeBase):
    pass


class Experiment(Base):
    __tablename__ = "experiments"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name: Mapped[str] = mapped_column(Text, nullable=False)
    task_type: Mapped[str] = mapped_column(Text, nullable=False)
    target_column: Mapped[str | None] = mapped_column(Text)
    primary_metric: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(Text, nullable=False, default="created")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    config: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)


class Dataset(Base):
    __tablename__ = "datasets"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    experiment_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("experiments.id", ondelete="CASCADE"))
    name: Mapped[str] = mapped_column(Text, nullable=False)
    source_type: Mapped[str] = mapped_column(Text, nullable=False)
    source_uri: Mapped[str | None] = mapped_column(Text)
    content_hash: Mapped[str | None] = mapped_column(Text)
    schema_json: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    profile_json: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    quality_report_json: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class AgentEvent(Base):
    __tablename__ = "agent_events"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    experiment_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("experiments.id", ondelete="CASCADE"))
    agent_name: Mapped[str] = mapped_column(Text, nullable=False)
    event_type: Mapped[str] = mapped_column(Text, nullable=False)
    input_json: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    output_json: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    rationale: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class PipelineRun(Base):
    __tablename__ = "pipeline_runs"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    experiment_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("experiments.id", ondelete="CASCADE"))
    code_version: Mapped[str | None] = mapped_column(Text)
    data_version: Mapped[str | None] = mapped_column(Text)
    seed: Mapped[int | None] = mapped_column(Integer)
    status: Mapped[str] = mapped_column(Text, nullable=False, default="running")
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    metrics_json: Mapped[dict[str, Any] | None] = mapped_column(JSONB)


class ModelResult(Base):
    __tablename__ = "model_results"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    run_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("pipeline_runs.id", ondelete="CASCADE"))
    model_name: Mapped[str] = mapped_column(Text, nullable=False)
    fold: Mapped[int | None] = mapped_column(Integer)
    hyperparameters: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    metrics_json: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    fit_seconds: Mapped[float | None] = mapped_column(Float)
    artifact_path: Mapped[str | None] = mapped_column(Text)
    artifact_hash: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class Report(Base):
    __tablename__ = "reports"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    experiment_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("experiments.id", ondelete="CASCADE"))
    run_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("pipeline_runs.id", ondelete="CASCADE"))
    format: Mapped[str] = mapped_column(String(16), nullable=False, default="markdown")
    content: Mapped[str | None] = mapped_column(Text)
    summary_json: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


def get_engine(database_url: str | None = None):
    """Cria um engine SQLAlchemy a partir de DATABASE_URL."""
    url = database_url or os.environ["DATABASE_URL"]
    return create_engine(url, future=True)


def get_session_factory(database_url: str | None = None):
    return sessionmaker(bind=get_engine(database_url), future=True)


class EventSink:
    """Persiste eventos agentivos. Injetado nos agentes via `BaseAgent(event_sink=...)`."""

    def __init__(self, session_factory) -> None:
        self._session_factory = session_factory

    def record_event(
        self,
        experiment_id: uuid.UUID,
        agent_name: str,
        event_type: str,
        input_json: dict[str, Any] | None = None,
        output_json: dict[str, Any] | None = None,
        rationale: str | None = None,
    ) -> None:
        with self._session_factory() as session:
            session.add(
                AgentEvent(
                    experiment_id=experiment_id,
                    agent_name=agent_name,
                    event_type=event_type,
                    input_json=input_json,
                    output_json=output_json,
                    rationale=rationale,
                )
            )
            session.commit()

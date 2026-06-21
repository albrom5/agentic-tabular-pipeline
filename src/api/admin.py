"""Painel de administração (estilo Django Admin) sobre os modelos SQLAlchemy.

Monta o `SQLAdmin <https://aminalaee.dev/sqladmin/>`_ no app FastAPI em ``/admin``,
gerando automaticamente listagem, detalhe e CRUD para todas as entidades do
schema (``src/db/models.py``). Dá uma visão administrativa transversal — todos os
experimentos, execuções, resultados e eventos agentivos — que a UI Streamlit,
focada em um experimento por vez, não oferece.

O acesso é protegido por um login simples cujas credenciais vêm de variáveis de
ambiente (``ADMIN_USER`` / ``ADMIN_PASSWORD``); o cookie de sessão é assinado com
``ADMIN_SECRET_KEY``.

A montagem é feita por :func:`setup_admin`, chamada condicionalmente em
``src/api/main.py`` apenas quando ``DATABASE_URL`` está definido (a engine real é
necessária). Sem banco — como no ambiente de testes — o ``/admin`` não é montado.
"""

from __future__ import annotations

import hmac
import os
import secrets

from sqladmin import Admin, ModelView
from sqladmin.authentication import AuthenticationBackend
from starlette.requests import Request

from src.db.models import (
    AgentEvent,
    Dataset,
    Experiment,
    ModelResult,
    PipelineRun,
    Report,
)


def check_credentials(username: str | None, password: str | None) -> bool:
    """Valida usuário/senha contra ``ADMIN_USER``/``ADMIN_PASSWORD`` (env).

    Usa :func:`hmac.compare_digest` para comparação em tempo constante (evita
    *timing attacks*). Lida lazy com o ambiente para que os testes possam
    sobrescrever as variáveis. Se as credenciais não estiverem configuradas, o
    acesso é negado (fail-closed).
    """
    expected_user = os.environ.get("ADMIN_USER")
    expected_password = os.environ.get("ADMIN_PASSWORD")
    if not expected_user or not expected_password:
        return False
    if username is None or password is None:
        return False
    return hmac.compare_digest(username, expected_user) and hmac.compare_digest(
        password, expected_password
    )


class AdminAuth(AuthenticationBackend):
    """Backend de autenticação por sessão do SQLAdmin (login com usuário/senha)."""

    async def login(self, request: Request) -> bool:
        form = await request.form()
        if check_credentials(form.get("username"), form.get("password")):
            request.session.update({"authenticated": True})
            return True
        return False

    async def logout(self, request: Request) -> bool:
        request.session.clear()
        return True

    async def authenticate(self, request: Request) -> bool:
        return bool(request.session.get("authenticated"))


# ---------------------------------------------------------------------------
# ModelViews — uma por entidade do schema
# ---------------------------------------------------------------------------

class ExperimentAdmin(ModelView, model=Experiment):
    name = "Experimento"
    name_plural = "Experimentos"
    icon = "fa-solid fa-flask"
    column_list = [
        Experiment.name,
        Experiment.task_type,
        Experiment.target_column,
        Experiment.primary_metric,
        Experiment.status,
        Experiment.created_at,
    ]
    column_searchable_list = [Experiment.name, Experiment.status, Experiment.task_type]
    column_sortable_list = [Experiment.name, Experiment.status, Experiment.created_at]
    column_default_sort = [(Experiment.created_at, True)]


class DatasetAdmin(ModelView, model=Dataset):
    name = "Dataset"
    name_plural = "Datasets"
    icon = "fa-solid fa-table"
    column_list = [
        Dataset.name,
        Dataset.source_type,
        Dataset.source_uri,
        Dataset.content_hash,
        Dataset.created_at,
    ]
    column_searchable_list = [Dataset.name, Dataset.source_type]
    column_sortable_list = [Dataset.name, Dataset.created_at]
    column_default_sort = [(Dataset.created_at, True)]


class AgentEventAdmin(ModelView, model=AgentEvent):
    name = "Evento agentivo"
    name_plural = "Eventos agentivos"
    icon = "fa-solid fa-robot"
    column_list = [
        AgentEvent.id,
        AgentEvent.agent_name,
        AgentEvent.event_type,
        AgentEvent.rationale,
        AgentEvent.created_at,
    ]
    column_searchable_list = [
        AgentEvent.agent_name,
        AgentEvent.event_type,
        AgentEvent.rationale,
    ]
    column_sortable_list = [AgentEvent.id, AgentEvent.agent_name, AgentEvent.created_at]
    column_default_sort = [(AgentEvent.created_at, True)]


class PipelineRunAdmin(ModelView, model=PipelineRun):
    name = "Execução"
    name_plural = "Execuções"
    icon = "fa-solid fa-play"
    column_list = [
        PipelineRun.id,
        PipelineRun.status,
        PipelineRun.seed,
        PipelineRun.code_version,
        PipelineRun.started_at,
        PipelineRun.finished_at,
    ]
    column_searchable_list = [PipelineRun.status, PipelineRun.code_version]
    column_sortable_list = [PipelineRun.status, PipelineRun.started_at]
    column_default_sort = [(PipelineRun.started_at, True)]


class ModelResultAdmin(ModelView, model=ModelResult):
    name = "Resultado de modelo"
    name_plural = "Resultados de modelos"
    icon = "fa-solid fa-trophy"
    column_list = [
        ModelResult.id,
        ModelResult.model_name,
        ModelResult.fold,
        ModelResult.fit_seconds,
        ModelResult.created_at,
    ]
    column_searchable_list = [ModelResult.model_name]
    column_sortable_list = [ModelResult.model_name, ModelResult.fold, ModelResult.created_at]
    column_default_sort = [(ModelResult.created_at, True)]


class ReportAdmin(ModelView, model=Report):
    name = "Relatório"
    name_plural = "Relatórios"
    icon = "fa-solid fa-file-lines"
    column_list = [
        Report.id,
        Report.format,
        Report.created_at,
    ]
    column_sortable_list = [Report.format, Report.created_at]
    column_default_sort = [(Report.created_at, True)]


_VIEWS = [
    ExperimentAdmin,
    DatasetAdmin,
    AgentEventAdmin,
    PipelineRunAdmin,
    ModelResultAdmin,
    ReportAdmin,
]


def setup_admin(app, engine) -> Admin:
    """Instancia o SQLAdmin no ``app`` FastAPI e registra todas as views.

    ``engine`` deve ser a engine SQLAlchemy do PostgreSQL (ver
    ``src.db.models.get_engine``). O ``secret_key`` do cookie de sessão vem de
    ``ADMIN_SECRET_KEY``; na ausência, é gerado um efêmero (sessões não persistem
    entre reinícios, o que é aceitável para um painel administrativo).
    """
    secret_key = os.environ.get("ADMIN_SECRET_KEY") or secrets.token_hex(32)
    admin = Admin(
        app,
        engine,
        authentication_backend=AdminAuth(secret_key=secret_key),
        title="Agentic Tabular · Admin",
    )
    for view in _VIEWS:
        admin.add_view(view)
    return admin

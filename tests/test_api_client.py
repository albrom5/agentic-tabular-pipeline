"""Testes do cliente HTTP da UI (``src.ui.api_client``).

Exercita o ``ApiClient`` contra a própria aplicação FastAPI via ``ASGITransport``
(sem servidor nem banco reais), com o store substituído pelo ``FakeStore``.
"""

from __future__ import annotations

import httpx
import pytest
from fastapi.testclient import TestClient

from src.api.main import app, get_store
from src.ui.api_client import ApiClient, ApiError
from tests.test_api import FakeStore


@pytest.fixture()
def api() -> ApiClient:
    fake = FakeStore()
    app.dependency_overrides[get_store] = lambda: fake
    # O TestClient da FastAPI é um httpx.Client; injeta-se direto no ApiClient,
    # exercitando a serialização HTTP real sem servidor nem banco.
    client = TestClient(app)
    api = ApiClient(base_url="http://testserver", client=client)
    api._fake = fake  # exposto para asserts
    yield api
    client.close()
    app.dependency_overrides.clear()


class TestApiClient:
    def test_health(self, api):
        assert api.health() is True

    def test_create_and_get(self, api):
        created = api.create_experiment(
            name="demo", task_type="classification", target_column="default",
            primary_metric="macro_f1", config={"experiment": {}},
        )
        eid = created["experiment_id"]
        # O token devolvido na criação autoriza as chamadas seguintes.
        assert created["access_token"]
        assert api.get_experiment(eid)["status"] == "created"

    def test_resolve_experiment_by_token(self, api):
        created = api.create_experiment(
            name="demo", task_type="classification", target_column="default",
            primary_metric="macro_f1", config={"experiment": {}},
        )
        resolved = api.resolve_experiment(created["access_token"])
        assert resolved["id"] == created["experiment_id"]

    def test_run(self, api):
        eid = api.create_experiment(
            name="d", task_type="classification", target_column="t",
            primary_metric="macro_f1", config={},
        )["experiment_id"]
        resp = api.run_experiment(eid)
        assert resp["status"] == "running"
        assert api._fake.executed == [eid]

    def test_get_report_missing_returns_none(self, api):
        eid = api.create_experiment(
            name="d", task_type="classification", target_column="t",
            primary_metric="macro_f1", config={},
        )["experiment_id"]
        assert api.get_report(eid) is None  # 404 → None, sem levantar
        assert api.get_profile(eid) is None

    def test_get_report_present(self, api):
        eid = api.create_experiment(
            name="d", task_type="classification", target_column="t",
            primary_metric="macro_f1", config={},
        )["experiment_id"]
        api._fake.reports[eid] = {"content": "# R", "summary_json": {"best_model": "rf"}}
        assert api.get_report(eid)["summary_json"]["best_model"] == "rf"

    def test_offline_health_false(self):
        # base_url inacessível → health() degrada para False
        client = httpx.Client(base_url="http://127.0.0.1:1", timeout=0.2)
        offline = ApiClient(base_url="http://127.0.0.1:1", client=client)
        assert offline.health() is False
        with pytest.raises(ApiError):
            offline.resolve_experiment("qualquer-token")
        client.close()

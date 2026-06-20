"""Testes da API FastAPI.

Os endpoints são exercitados com o ``TestClient`` e um ``FakeStore`` em memória
injetado por ``app.dependency_overrides`` — sem PostgreSQL real. A lógica de
agregação do ranking é testada à parte pela função pura ``aggregate_ranking``.
"""

from __future__ import annotations

import uuid
from collections import defaultdict

import pytest
from fastapi.testclient import TestClient

from src.api.main import app, get_store
from src.api.store import aggregate_ranking


class FakeStore:
    """Implementação em memória da interface usada pelos endpoints."""

    def __init__(self) -> None:
        self.experiments: dict[str, dict] = {}
        self.events: dict[str, list[dict]] = defaultdict(list)
        self.runs: dict[str, list[dict]] = defaultdict(list)
        self.rankings: dict[str, list[dict]] = {}
        self.reports: dict[str, dict] = {}
        self.datasets: dict[str, dict] = {}
        self.executed: list[str] = []

    def list_experiments(self):
        return list(self.experiments.values())

    def create_experiment(self, *, name, task_type, target_column, primary_metric, config):
        eid = str(uuid.uuid4())
        self.experiments[eid] = {
            "id": eid, "name": name, "task_type": task_type,
            "target_column": target_column, "primary_metric": primary_metric,
            "status": "created", "config": config,
        }
        return eid

    def get_experiment(self, experiment_id):
        return self.experiments.get(experiment_id)

    def execute(self, experiment_id):
        self.executed.append(experiment_id)
        self.experiments[experiment_id]["status"] = "completed"

    def list_events(self, experiment_id):
        return self.events[experiment_id]

    def list_runs(self, experiment_id):
        return self.runs[experiment_id]

    def ranking(self, run_id):
        return self.rankings.get(run_id, [])

    def get_report(self, experiment_id):
        return self.reports.get(experiment_id)

    def get_dataset(self, experiment_id):
        return self.datasets.get(experiment_id)


@pytest.fixture()
def store() -> FakeStore:
    fake = FakeStore()
    app.dependency_overrides[get_store] = lambda: fake
    yield fake
    app.dependency_overrides.clear()


@pytest.fixture()
def client() -> TestClient:
    return TestClient(app)


_PAYLOAD = {
    "name": "demo",
    "task_type": "classification",
    "target_column": "default",
    "primary_metric": "macro_f1",
    "config": {"experiment": {"name": "demo"}},
}


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

class TestHealth:
    def test_health_ok(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}


class TestCreateExperiment:
    def test_creates_and_returns_id(self, client, store):
        resp = client.post("/experiments", json=_PAYLOAD)
        assert resp.status_code == 201
        body = resp.json()
        assert body["status"] == "created"
        assert body["experiment_id"] in store.experiments

    def test_validation_error_when_missing_field(self, client, store):
        bad = {k: v for k, v in _PAYLOAD.items() if k != "primary_metric"}
        resp = client.post("/experiments", json=bad)
        assert resp.status_code == 422


class TestRunExperiment:
    def test_run_unknown_returns_404(self, client, store):
        resp = client.post(f"/experiments/{uuid.uuid4()}/run")
        assert resp.status_code == 404

    def test_run_known_schedules_execution(self, client, store):
        eid = client.post("/experiments", json=_PAYLOAD).json()["experiment_id"]
        resp = client.post(f"/experiments/{eid}/run")
        assert resp.status_code == 202
        assert resp.json()["status"] == "running"
        # BackgroundTasks rodam após a resposta no TestClient.
        assert store.executed == [eid]
        assert store.experiments[eid]["status"] == "completed"


class TestEvents:
    def test_events_unknown_returns_404(self, client, store):
        resp = client.get(f"/experiments/{uuid.uuid4()}/events")
        assert resp.status_code == 404

    def test_events_returns_history(self, client, store):
        eid = client.post("/experiments", json=_PAYLOAD).json()["experiment_id"]
        store.events[eid] = [
            {"agent_name": "Agente de Limpeza", "event_type": "cleaning_decision"}
        ]
        resp = client.get(f"/experiments/{eid}/events")
        assert resp.status_code == 200
        assert resp.json()[0]["agent_name"] == "Agente de Limpeza"


class TestListAndGet:
    def test_list_experiments(self, client, store):
        client.post("/experiments", json=_PAYLOAD)
        resp = client.get("/experiments")
        assert resp.status_code == 200
        assert len(resp.json()) == 1
        assert resp.json()[0]["name"] == "demo"

    def test_get_experiment_detail(self, client, store):
        eid = client.post("/experiments", json=_PAYLOAD).json()["experiment_id"]
        resp = client.get(f"/experiments/{eid}")
        assert resp.status_code == 200
        assert resp.json()["status"] == "created"

    def test_get_experiment_unknown_404(self, client, store):
        assert client.get(f"/experiments/{uuid.uuid4()}").status_code == 404


class TestReportAndProfile:
    def test_report_unknown_404(self, client, store):
        assert client.get(f"/experiments/{uuid.uuid4()}/report").status_code == 404

    def test_report_returns_markdown(self, client, store):
        eid = client.post("/experiments", json=_PAYLOAD).json()["experiment_id"]
        store.reports[eid] = {"content": "# Relatório", "summary_json": {"best_model": "rf"}}
        resp = client.get(f"/experiments/{eid}/report")
        assert resp.status_code == 200
        assert resp.json()["content"].startswith("# Relatório")

    def test_profile_unknown_404(self, client, store):
        assert client.get(f"/experiments/{uuid.uuid4()}/profile").status_code == 404

    def test_profile_returns_dataset(self, client, store):
        eid = client.post("/experiments", json=_PAYLOAD).json()["experiment_id"]
        store.datasets[eid] = {"name": "credit", "profile_json": {"n_rows": 1000}}
        resp = client.get(f"/experiments/{eid}/profile")
        assert resp.status_code == 200
        assert resp.json()["profile_json"]["n_rows"] == 1000


class TestRanking:
    def test_ranking_returns_store_payload(self, client, store):
        run_id = str(uuid.uuid4())
        store.rankings[run_id] = [{"model_name": "rf", "mean": 0.9}]
        resp = client.get(f"/runs/{run_id}/ranking")
        assert resp.status_code == 200
        assert resp.json()[0]["model_name"] == "rf"


# ---------------------------------------------------------------------------
# Agregação do ranking (função pura)
# ---------------------------------------------------------------------------

class TestAggregateRanking:
    def _rows(self):
        return [
            ("rf", {"macro_f1": 0.90}, 1.0),
            ("rf", {"macro_f1": 0.80}, 3.0),
            ("logreg", {"macro_f1": 0.70}, 0.5),
            ("logreg", {"macro_f1": 0.60}, 0.5),
        ]

    def test_groups_and_orders_descending_for_score(self):
        ranking = aggregate_ranking(self._rows(), "macro_f1")
        assert [r["model_name"] for r in ranking] == ["rf", "logreg"]
        assert ranking[0]["mean"] == pytest.approx(0.85)
        assert ranking[0]["folds"] == 2
        assert ranking[0]["std"] == pytest.approx(0.05)
        assert ranking[0]["mean_fit_seconds"] == pytest.approx(2.0)

    def test_orders_ascending_for_error_metric(self):
        rows = [
            ("a", {"rmse": 1.0}, None),
            ("a", {"rmse": 3.0}, None),
            ("b", {"rmse": 0.5}, None),
        ]
        ranking = aggregate_ranking(rows, "rmse")
        assert [r["model_name"] for r in ranking] == ["b", "a"]

    def test_ignores_rows_without_primary_metric(self):
        rows = [("a", {"accuracy": 0.9}, None), ("b", {"macro_f1": 0.5}, None)]
        ranking = aggregate_ranking(rows, "macro_f1")
        assert [r["model_name"] for r in ranking] == ["b"]

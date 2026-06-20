"""Cliente HTTP da UI Streamlit para a API FastAPI do pipeline.

Encapsula as chamadas REST (``API_URL``) usadas pela interface: cadastro de
experimentos, disparo de execução e consultas de eventos, ranking, perfil e
relatório. Mantém a UI desacoplada do backend e fácil de testar/instrumentar.
"""

from __future__ import annotations

import os
from typing import Any

import httpx

DEFAULT_TIMEOUT = 30.0


class ApiError(RuntimeError):
    """Falha de comunicação ou resposta de erro da API."""


class ApiClient:
    def __init__(
        self,
        base_url: str | None = None,
        timeout: float = DEFAULT_TIMEOUT,
        client: httpx.Client | None = None,
    ) -> None:
        self.base_url = (base_url or os.environ.get("API_URL", "http://localhost:8000")).rstrip("/")
        # ``client`` injetável permite testar via ASGITransport sem servidor real.
        self._client = client or httpx.Client(base_url=self.base_url, timeout=timeout)

    # -- infraestrutura --------------------------------------------------------

    def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        try:
            resp = self._client.request(method, path, **kwargs)
        except httpx.HTTPError as exc:  # rede indisponível, timeout, etc.
            raise ApiError(f"Não foi possível contatar a API em {self.base_url}: {exc}") from exc
        if resp.status_code >= 400:
            detail = _safe_detail(resp)
            raise ApiError(f"{method} {path} → {resp.status_code}: {detail}")
        if resp.content:
            return resp.json()
        return None

    def health(self) -> bool:
        try:
            return self._request("GET", "/health").get("status") == "ok"
        except ApiError:
            return False

    # -- experimentos ----------------------------------------------------------

    def list_experiments(self) -> list[dict[str, Any]]:
        return self._request("GET", "/experiments") or []

    def get_experiment(self, experiment_id: str) -> dict[str, Any]:
        return self._request("GET", f"/experiments/{experiment_id}")

    def create_experiment(
        self,
        *,
        name: str,
        task_type: str,
        target_column: str | None,
        primary_metric: str,
        config: dict[str, Any],
    ) -> dict[str, Any]:
        payload = {
            "name": name,
            "task_type": task_type,
            "target_column": target_column,
            "primary_metric": primary_metric,
            "config": config,
        }
        return self._request("POST", "/experiments", json=payload)

    def run_experiment(self, experiment_id: str) -> dict[str, Any]:
        return self._request("POST", f"/experiments/{experiment_id}/run")

    # -- consultas -------------------------------------------------------------

    def list_events(self, experiment_id: str) -> list[dict[str, Any]]:
        return self._request("GET", f"/experiments/{experiment_id}/events") or []

    def list_runs(self, experiment_id: str) -> list[dict[str, Any]]:
        return self._request("GET", f"/experiments/{experiment_id}/runs") or []

    def ranking(self, run_id: str) -> list[dict[str, Any]]:
        return self._request("GET", f"/runs/{run_id}/ranking") or []

    def get_profile(self, experiment_id: str) -> dict[str, Any] | None:
        try:
            return self._request("GET", f"/experiments/{experiment_id}/profile")
        except ApiError:
            return None

    def get_report(self, experiment_id: str) -> dict[str, Any] | None:
        try:
            return self._request("GET", f"/experiments/{experiment_id}/report")
        except ApiError:
            return None


def _safe_detail(resp: httpx.Response) -> str:
    try:
        return resp.json().get("detail", resp.text)
    except Exception:  # noqa: BLE001 - corpo não-JSON
        return resp.text

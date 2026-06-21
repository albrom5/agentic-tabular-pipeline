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
        # Tokens de acesso por experimento (capability tokens), enviados no
        # cabeçalho ``X-Experiment-Token`` das chamadas àquele experimento.
        self._tokens: dict[str, str] = {}

    # -- tokens de acesso ------------------------------------------------------

    def register_token(self, experiment_id: str, token: str) -> None:
        """Memoriza o token de um experimento para as próximas chamadas."""
        if experiment_id and token:
            self._tokens[experiment_id] = token

    def _auth_headers(self, experiment_id: str) -> dict[str, str]:
        token = self._tokens.get(experiment_id)
        return {"X-Experiment-Token": token} if token else {}

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

    def resolve_experiment(self, token: str) -> dict[str, Any]:
        """Recupera um experimento pelo token e memoriza o token para reuso."""
        exp = self._request("POST", "/experiments/resolve", json={"token": token})
        self.register_token(exp["id"], token)
        return exp

    def get_experiment(self, experiment_id: str) -> dict[str, Any]:
        return self._request(
            "GET", f"/experiments/{experiment_id}",
            headers=self._auth_headers(experiment_id),
        )

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
        created = self._request("POST", "/experiments", json=payload)
        # Guarda o token recém-gerado para autorizar as chamadas seguintes.
        self.register_token(created["experiment_id"], created.get("access_token", ""))
        return created

    def run_experiment(self, experiment_id: str) -> dict[str, Any]:
        return self._request(
            "POST", f"/experiments/{experiment_id}/run",
            headers=self._auth_headers(experiment_id),
        )

    # -- consultas -------------------------------------------------------------

    def list_events(self, experiment_id: str) -> list[dict[str, Any]]:
        return self._request(
            "GET", f"/experiments/{experiment_id}/events",
            headers=self._auth_headers(experiment_id),
        ) or []

    def list_runs(self, experiment_id: str) -> list[dict[str, Any]]:
        return self._request(
            "GET", f"/experiments/{experiment_id}/runs",
            headers=self._auth_headers(experiment_id),
        ) or []

    def ranking(self, run_id: str, experiment_id: str) -> list[dict[str, Any]]:
        return self._request(
            "GET", f"/runs/{run_id}/ranking",
            headers=self._auth_headers(experiment_id),
        ) or []

    def get_profile(self, experiment_id: str) -> dict[str, Any] | None:
        try:
            return self._request(
                "GET", f"/experiments/{experiment_id}/profile",
                headers=self._auth_headers(experiment_id),
            )
        except ApiError:
            return None

    def get_report(self, experiment_id: str) -> dict[str, Any] | None:
        try:
            return self._request(
                "GET", f"/experiments/{experiment_id}/report",
                headers=self._auth_headers(experiment_id),
            )
        except ApiError:
            return None


def _safe_detail(resp: httpx.Response) -> str:
    try:
        return resp.json().get("detail", resp.text)
    except Exception:  # noqa: BLE001 - corpo não-JSON
        return resp.text

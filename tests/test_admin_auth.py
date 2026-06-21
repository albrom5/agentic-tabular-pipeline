"""Testes da autenticação do painel de administração (``src/api/admin.py``).

Exercitam a validação de credenciais e o backend de sessão do SQLAdmin sem exigir
PostgreSQL — as ``ModelView``/``setup_admin`` dependem da engine real e ficam fora
deste escopo. As variáveis ``ADMIN_USER``/``ADMIN_PASSWORD`` são injetadas via
``monkeypatch``.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from src.api.admin import AdminAuth, check_credentials


class TestCheckCredentials:
    @pytest.fixture(autouse=True)
    def _env(self, monkeypatch):
        monkeypatch.setenv("ADMIN_USER", "admin")
        monkeypatch.setenv("ADMIN_PASSWORD", "s3cr3t")

    def test_accepts_valid_credentials(self):
        assert check_credentials("admin", "s3cr3t") is True

    def test_rejects_wrong_password(self):
        assert check_credentials("admin", "errada") is False

    def test_rejects_wrong_user(self):
        assert check_credentials("root", "s3cr3t") is False

    def test_rejects_none(self):
        assert check_credentials(None, None) is False

    def test_fail_closed_when_env_missing(self, monkeypatch):
        monkeypatch.delenv("ADMIN_USER", raising=False)
        monkeypatch.delenv("ADMIN_PASSWORD", raising=False)
        assert check_credentials("admin", "s3cr3t") is False


class _FakeRequest:
    """Stub de ``starlette.requests.Request`` com form e sessão controláveis."""

    def __init__(self, form: dict[str, str]) -> None:
        self._form = form
        self.session: dict[str, object] = {}

    async def form(self):
        return self._form


class TestAdminAuthBackend:
    @pytest.fixture(autouse=True)
    def _env(self, monkeypatch):
        monkeypatch.setenv("ADMIN_USER", "admin")
        monkeypatch.setenv("ADMIN_PASSWORD", "s3cr3t")

    @pytest.fixture()
    def backend(self) -> AdminAuth:
        return AdminAuth(secret_key="test-secret")

    def test_login_sets_session_on_success(self, backend):
        req = _FakeRequest({"username": "admin", "password": "s3cr3t"})
        assert asyncio.run(backend.login(req)) is True
        assert req.session.get("authenticated") is True

    def test_login_fails_on_bad_credentials(self, backend):
        req = _FakeRequest({"username": "admin", "password": "errada"})
        assert asyncio.run(backend.login(req)) is False
        assert "authenticated" not in req.session

    def test_authenticate_requires_session_flag(self, backend):
        anon = SimpleNamespace(session={})
        authed = SimpleNamespace(session={"authenticated": True})
        assert asyncio.run(backend.authenticate(anon)) is False
        assert asyncio.run(backend.authenticate(authed)) is True

    def test_logout_clears_session(self, backend):
        req = SimpleNamespace(session={"authenticated": True})
        assert asyncio.run(backend.logout(req)) is True
        assert req.session == {}

"""Tokens de acesso por experimento (confidencialidade dos dados).

Cada experimento recebe, na criação, um *token* de alta entropia que funciona
como uma chave de capacidade (*capability token*): quem o possui acessa os dados
daquele experimento; quem não o possui não consegue nem enumerá-lo. O token é
exibido ao usuário uma única vez (na criação) — no banco persistimos apenas o seu
hash SHA-256, de modo que um vazamento do banco (ou do painel ``/admin``) não
revele os tokens em claro.

A entropia do token (``secrets.token_urlsafe(32)`` ≈ 256 bits) torna a busca por
força bruta inviável, dispensando o *salt*/KDF que seria obrigatório para senhas
de baixa entropia escolhidas por humanos.
"""

from __future__ import annotations

import hashlib
import hmac
import secrets


def generate_token() -> str:
    """Gera um token de acesso novo e imprevisível (mostrado uma única vez)."""
    return secrets.token_urlsafe(32)


def hash_token(token: str) -> str:
    """Hash SHA-256 (hex) do token, formato persistido em ``access_token_hash``."""
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def verify_token(token: str | None, expected_hash: str | None) -> bool:
    """Confere ``token`` contra o hash esperado em tempo constante (fail-closed)."""
    if not token or not expected_hash:
        return False
    return hmac.compare_digest(hash_token(token), expected_hash)

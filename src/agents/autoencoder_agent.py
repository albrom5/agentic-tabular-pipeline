"""Agente de Autoencoders.

Treina um autoencoder tabular denso (PyTorch) para uma de três aplicações
(RF10, seção 9 do documento), **sempre comparando contra uma linha de base sem
autoencoder** — o cuidado principal exigido na tabela de agentes:

- ``latent_features`` (representação): usa o vetor latente como atributo para um
  modelo supervisionado e compara com o mesmo modelo sem o latente.
- ``denoising``: reconstrói a entrada corrompida por ruído e compara o erro de
  reconstrução com um preditor-média (baseline trivial).
- ``anomaly_detection``: usa o erro de reconstrução como score de anomalia e
  compara ROC-AUC/PR-AUC contra um Isolation Forest.

Restrição metodológica: o autoencoder é ajustado **apenas com dados de treino
dentro de cada fold** — treinar com toda a base antes da validação causaria
vazamento. Por isso o agente opera sobre os folds do SplitAgent e reaproveita o
pipeline de atributos (ajustado por fold).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.base import clone
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    mean_absolute_error,
    r2_score,
    roc_auc_score,
    root_mean_squared_error,
)
from sklearn.preprocessing import StandardScaler
from torch import nn

from src.agents.base import AgentResult, BaseAgent

# ---------------------------------------------------------------------------
# Convenções
# ---------------------------------------------------------------------------

_DEFAULT_SEED = 42
_USE_CASES = {"latent_features", "denoising", "anomaly_detection"}
_DEFAULTS = {
    "latent_dim": 8,
    "epochs": 30,
    "batch_size": 64,
    "hidden_dims": [64, 32],
    "learning_rate": 1e-3,
    "noise_std": 0.1,
    "latent_mode": "augment",  # augment | replace (apenas latent_features)
}
#: Métricas em que valores menores são melhores.
_LOWER_IS_BETTER = {"rmse", "mae", "mse", "mape", "reconstruction_mse", "log_loss"}


# ---------------------------------------------------------------------------
# Autoencoder denso (PyTorch)
# ---------------------------------------------------------------------------

class _DenseAutoencoder(nn.Module):
    """Autoencoder totalmente conectado para dados tabulares contínuos."""

    def __init__(self, input_dim: int, latent_dim: int, hidden_dims: list[int]) -> None:
        super().__init__()
        dims = [input_dim, *hidden_dims]
        encoder: list[nn.Module] = []
        for a, b in zip(dims[:-1], dims[1:]):
            encoder += [nn.Linear(a, b), nn.ReLU()]
        encoder.append(nn.Linear(dims[-1], latent_dim))
        self.encoder = nn.Sequential(*encoder)

        rev = [latent_dim, *hidden_dims[::-1]]
        decoder: list[nn.Module] = []
        for a, b in zip(rev[:-1], rev[1:]):
            decoder += [nn.Linear(a, b), nn.ReLU()]
        decoder.append(nn.Linear(rev[-1], input_dim))
        self.decoder = nn.Sequential(*decoder)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class AutoencoderAgent(BaseAgent):
    """Agente 7 — Autoencoders tabulares.

    Entradas esperadas em ``context``:
        - ``dataframe`` (pandas.DataFrame): base já limpa.
        - ``task_type`` / ``target_column`` / ``primary_metric``: da Formulação
          (lidos de ``context['problem']`` se ausentes no topo).
        - ``folds``: pares ``(train_idx, test_idx)`` (SplitAgent) — ou
          ``context['split']['folds']``.
        - ``pipeline`` (sklearn, opcional): pipeline de atributos NÃO ajustado;
          clonado e ajustado por fold (anti-leakage).
        - ``feature_columns`` (list, opcional).

    Configuração do autoencoder (``context['autoencoder']`` ou chaves no topo):
        - ``use_case``: "latent_features" (padrão) | "denoising" | "anomaly_detection".
        - ``latent_dim`` (8), ``epochs`` (30), ``batch_size`` (64),
          ``hidden_dims`` ([64, 32]), ``learning_rate`` (1e-3),
          ``noise_std`` (0.1, só denoising),
          ``latent_mode`` ("augment" | "replace", só latent_features).
        - ``random_seed`` (42).

    Saída em ``AgentResult.output``:
        - ``use_case``, ``config``, ``random_seed``, ``n_folds``;
        - ``folds``: métricas do autoencoder e do baseline por fold;
        - ``ae_metrics_mean`` / ``baseline_metrics_mean``;
        - ``comparison`` (métrica primária: AE vs baseline, ``ae_better``, ``delta``);
        - ``verdict`` (texto), ``warnings``.
    """

    name = "Agente de Autoencoders"
    event_type = "autoencoder_training"

    def run(self, context: dict[str, Any]) -> AgentResult:
        warnings: list[str] = []

        df = _load_dataframe(context)
        problem = context.get("problem") or {}
        task_type = str(context.get("task_type") or problem.get("task_type") or "").strip().lower()
        target_column = context.get("target_column") or problem.get("target_column") or None
        primary_metric = str(
            context.get("primary_metric") or problem.get("primary_metric") or ""
        ).strip().lower() or None

        cfg = _resolve_config(context)
        use_case = cfg["use_case"]
        if use_case not in _USE_CASES:
            raise ValueError(
                f"'use_case' inválido: '{use_case}'. Opções: {sorted(_USE_CASES)}."
            )
        if target_column and target_column not in df.columns:
            raise ValueError(f"'target_column' '{target_column}' não existe na base.")

        seed = int(context.get("random_seed", _DEFAULT_SEED))
        folds = _resolve_folds(context, len(df))
        feature_columns = context.get("feature_columns") or [
            c for c in df.columns if c != target_column
        ]
        pipeline = context.get("pipeline")

        # Dispatch por aplicação do autoencoder
        if use_case == "latent_features":
            if task_type not in ("classification", "regression"):
                raise ValueError(
                    "'latent_features' requer task_type 'classification' ou 'regression'."
                )
            if not target_column:
                raise ValueError("'latent_features' requer 'target_column'.")
            primary = _resolve_primary(primary_metric, task_type, warnings)
            fold_records = _run_latent_features(
                df, folds, feature_columns, target_column, pipeline,
                task_type, primary, cfg, seed,
            )
        elif use_case == "denoising":
            primary = "reconstruction_mse"
            fold_records = _run_denoising(
                df, folds, feature_columns, target_column, pipeline, cfg, seed,
            )
        else:  # anomaly_detection
            primary = "roc_auc"
            if not target_column:
                warnings.append(
                    "Detecção de anomalias sem 'target_column': sem rótulos não há "
                    "métricas supervisionadas; reporta-se apenas estatísticas dos scores."
                )
            fold_records = _run_anomaly(
                df, folds, feature_columns, target_column, pipeline, cfg, seed,
            )

        # Agregação e comparação AE × baseline
        ae_mean = _aggregate([r["ae_metrics"] for r in fold_records])
        base_mean = _aggregate([r["baseline_metrics"] for r in fold_records])
        comparison = _comparison(primary, ae_mean, base_mean)
        verdict = _verdict(use_case, comparison)

        output: dict[str, Any] = {
            "use_case": use_case,
            "task_type": task_type or None,
            "target_column": target_column,
            "config": cfg,
            "random_seed": seed,
            "n_folds": len(folds),
            "folds": fold_records,
            "ae_metrics_mean": ae_mean,
            "baseline_metrics_mean": base_mean,
            "comparison": comparison,
            "verdict": verdict,
            "warnings": warnings,
        }
        output = _to_native(output)

        rationale = _build_rationale(output)
        return AgentResult(output=output, rationale=rationale, warnings=warnings)


# ---------------------------------------------------------------------------
# Aplicação 1 — representação (latent features)
# ---------------------------------------------------------------------------

def _run_latent_features(
    df, folds, feature_columns, target_column, pipeline,
    task_type, primary, cfg, seed,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for i, (train_idx, test_idx) in enumerate(folds):
        Xtr, Xte = _transform_fold(pipeline, df, train_idx, test_idx, feature_columns, target_column)
        y_train = df.iloc[train_idx][target_column]
        y_test = df.iloc[test_idx][target_column]

        ae, scaler, loss = _train_autoencoder(Xtr, cfg, seed, denoising=False)
        Ztr, Zte = _latent(ae, scaler, Xtr), _latent(ae, scaler, Xte)

        if cfg["latent_mode"] == "replace":
            Ftr, Fte = Ztr, Zte
        else:  # augment: atributos originais + latentes
            Ftr, Fte = np.hstack([Xtr, Ztr]), np.hstack([Xte, Zte])

        ae_metrics = _supervised_metrics(task_type, Ftr, y_train, Fte, y_test, seed)
        baseline_metrics = _supervised_metrics(task_type, Xtr, y_train, Xte, y_test, seed)
        records.append({
            "fold": i,
            "ae_metrics": ae_metrics,
            "baseline_metrics": baseline_metrics,
            "train_loss_final": round(loss, 6),
        })
    return records


# ---------------------------------------------------------------------------
# Aplicação 2 — denoising
# ---------------------------------------------------------------------------

def _run_denoising(
    df, folds, feature_columns, target_column, pipeline, cfg, seed,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    noise_std = float(cfg["noise_std"])
    for i, (train_idx, test_idx) in enumerate(folds):
        Xtr, Xte = _transform_fold(pipeline, df, train_idx, test_idx, feature_columns, target_column)
        ae, scaler, loss = _train_autoencoder(Xtr, cfg, seed, denoising=True)

        # Corrompe o teste com ruído controlado (seed) e mede a reconstrução do limpo.
        Xte_s = scaler.transform(Xte).astype(np.float32)
        rng = np.random.default_rng(seed + i)
        noisy = Xte_s + rng.normal(scale=noise_std, size=Xte_s.shape).astype(np.float32)
        with torch.no_grad():
            recon = ae(torch.from_numpy(noisy)).numpy()
        ae_mse = float(np.mean((recon - Xte_s) ** 2))
        # Baseline: preditor-média (no espaço padronizado, a média de treino é 0).
        baseline_mse = float(np.mean(Xte_s ** 2))
        noisy_mse = float(np.mean((noisy - Xte_s) ** 2))
        records.append({
            "fold": i,
            "ae_metrics": {"reconstruction_mse": round(ae_mse, 6)},
            "baseline_metrics": {"reconstruction_mse": round(baseline_mse, 6)},
            "noisy_input_mse": round(noisy_mse, 6),
            "train_loss_final": round(loss, 6),
        })
    return records


# ---------------------------------------------------------------------------
# Aplicação 3 — detecção de anomalias
# ---------------------------------------------------------------------------

def _run_anomaly(
    df, folds, feature_columns, target_column, pipeline, cfg, seed,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for i, (train_idx, test_idx) in enumerate(folds):
        Xtr, Xte = _transform_fold(pipeline, df, train_idx, test_idx, feature_columns, target_column)
        ae, scaler, loss = _train_autoencoder(Xtr, cfg, seed, denoising=False)
        ae_score = _reconstruction_error(ae, scaler, Xte)  # maior = mais anômalo

        iso = IsolationForest(random_state=seed).fit(Xtr)
        base_score = -iso.score_samples(Xte)

        y_test = df.iloc[test_idx][target_column] if target_column else None
        ae_metrics = _anomaly_metrics(ae_score, y_test)
        baseline_metrics = _anomaly_metrics(base_score, y_test)
        record = {
            "fold": i,
            "ae_metrics": ae_metrics,
            "baseline_metrics": baseline_metrics,
            "train_loss_final": round(loss, 6),
        }
        if y_test is None:
            record["ae_score_stats"] = _score_stats(ae_score)
        records.append(record)
    return records


# ---------------------------------------------------------------------------
# Treino do autoencoder e extração de representações
# ---------------------------------------------------------------------------

def _train_autoencoder(
    X: np.ndarray, cfg: dict[str, Any], seed: int, *, denoising: bool
) -> tuple[_DenseAutoencoder, StandardScaler, float]:
    torch.manual_seed(seed)
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X).astype(np.float32)
    n, input_dim = Xs.shape

    hidden = [h for h in cfg["hidden_dims"] if h > 0] or [32]
    model = _DenseAutoencoder(input_dim, int(cfg["latent_dim"]), [int(h) for h in hidden])
    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg["learning_rate"]))
    loss_fn = nn.MSELoss()
    batch_size = max(1, int(cfg["batch_size"]))
    rng = np.random.default_rng(seed)

    model.train()
    final_loss = float("nan")
    for _ in range(int(cfg["epochs"])):
        perm = rng.permutation(n)
        epoch_losses: list[float] = []
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            clean = torch.from_numpy(Xs[idx])
            inp = clean
            if denoising:
                noise = torch.from_numpy(
                    rng.normal(scale=float(cfg["noise_std"]), size=clean.shape).astype(np.float32)
                )
                inp = clean + noise
            optimizer.zero_grad()
            loss = loss_fn(model(inp), clean)
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.item()))
        final_loss = float(np.mean(epoch_losses)) if epoch_losses else final_loss
    model.eval()
    return model, scaler, final_loss


def _latent(model: _DenseAutoencoder, scaler: StandardScaler, X: np.ndarray) -> np.ndarray:
    Xs = scaler.transform(X).astype(np.float32)
    with torch.no_grad():
        return model.encode(torch.from_numpy(Xs)).numpy()


def _reconstruction_error(
    model: _DenseAutoencoder, scaler: StandardScaler, X: np.ndarray
) -> np.ndarray:
    Xs = scaler.transform(X).astype(np.float32)
    with torch.no_grad():
        recon = model(torch.from_numpy(Xs)).numpy()
    return np.mean((recon - Xs) ** 2, axis=1)


# ---------------------------------------------------------------------------
# Métricas
# ---------------------------------------------------------------------------

def _supervised_metrics(
    task_type: str, X_train, y_train, X_test, y_test, seed: int
) -> dict[str, float]:
    """Ajusta um modelo-sonda simples e devolve métricas no teste."""
    if task_type == "classification":
        model = LogisticRegression(max_iter=1000, random_state=seed)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        metrics = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "macro_f1": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
        }
        classes = list(model.classes_)
        if len(classes) == 2:
            try:
                proba = model.predict_proba(X_test)[:, 1]
                y_bin = (np.asarray(y_test) == classes[1]).astype(int)
                metrics["roc_auc"] = float(roc_auc_score(y_bin, proba))
            except Exception:  # noqa: BLE001
                pass
        return metrics

    model = Ridge(random_state=seed)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_true = np.asarray(y_test, dtype=float)
    rmse = float(root_mean_squared_error(y_true, y_pred))
    return {
        "rmse": rmse,
        "mse": float(rmse**2),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def _anomaly_metrics(score: np.ndarray, y_true: pd.Series | None) -> dict[str, float]:
    if y_true is None:
        return {}
    arr = np.asarray(y_true)
    positive = _anomaly_positive_label(arr)
    y_bin = (arr == positive).astype(int)
    metrics: dict[str, float] = {}
    try:
        metrics["roc_auc"] = float(roc_auc_score(y_bin, score))
        metrics["average_precision"] = float(average_precision_score(y_bin, score))
        k = int(y_bin.sum())
        if k:
            top = np.argsort(score)[::-1][:k]
            metrics["precision_at_k"] = float(y_bin[top].sum() / k)
    except Exception:  # noqa: BLE001
        pass
    return metrics


def _anomaly_positive_label(y: np.ndarray) -> Any:
    values, counts = np.unique(y, return_counts=True)
    value_set = set(values.tolist())
    if value_set <= {0, 1}:
        return 1
    if value_set <= {True, False}:
        return True
    return values[int(np.argmin(counts))]


def _score_stats(score: np.ndarray) -> dict[str, float]:
    return {
        "mean": round(float(np.mean(score)), 6),
        "std": round(float(np.std(score)), 6),
        "p95": round(float(np.quantile(score, 0.95)), 6),
        "max": round(float(np.max(score)), 6),
    }


# ---------------------------------------------------------------------------
# Agregação e comparação
# ---------------------------------------------------------------------------

def _aggregate(metric_dicts: list[dict[str, float]]) -> dict[str, float]:
    keys: set[str] = set()
    for d in metric_dicts:
        keys.update(d.keys())
    mean: dict[str, float] = {}
    for key in sorted(keys):
        vals = [d[key] for d in metric_dicts if d.get(key) is not None]
        if vals:
            mean[key] = round(float(np.mean(vals)), 6)
    return mean


def _comparison(
    primary: str | None, ae_mean: dict[str, float], base_mean: dict[str, float]
) -> dict[str, Any]:
    a = ae_mean.get(primary) if primary else None
    b = base_mean.get(primary) if primary else None
    if primary is None or a is None or b is None:
        return {"primary_metric": primary, "conclusive": False}
    lower = primary in _LOWER_IS_BETTER
    ae_better = a < b if lower else a > b
    return {
        "primary_metric": primary,
        "autoencoder": a,
        "baseline": b,
        "delta": round(a - b, 6),
        "lower_is_better": lower,
        "ae_better": bool(ae_better),
        "conclusive": True,
    }


def _verdict(use_case: str, comparison: dict[str, Any]) -> str:
    if not comparison.get("conclusive"):
        return (
            f"Comparação inconclusiva para '{use_case}': métrica primária indisponível "
            "(ex.: anomalia sem rótulos)."
        )
    metric = comparison["primary_metric"]
    a, b = comparison["autoencoder"], comparison["baseline"]
    if comparison["ae_better"]:
        return (
            f"O autoencoder ({use_case}) SUPERA o baseline em '{metric}': "
            f"{a} vs {b}."
        )
    return (
        f"O autoencoder ({use_case}) NÃO supera o baseline em '{metric}': "
        f"{a} vs {b}. Use a linha de base."
    )


# ---------------------------------------------------------------------------
# Ingestão, configuração e helpers
# ---------------------------------------------------------------------------

def _transform_fold(
    pipeline, df, train_idx, test_idx, feature_columns, target_column
) -> tuple[np.ndarray, np.ndarray]:
    X_train = df.iloc[train_idx][feature_columns]
    X_test = df.iloc[test_idx][feature_columns]
    y_train = df.iloc[train_idx][target_column] if target_column else None
    if pipeline is None:
        return X_train.to_numpy(dtype=float), X_test.to_numpy(dtype=float)
    pipe = clone(pipeline)
    Xtr = np.asarray(pipe.fit_transform(X_train, y_train), dtype=float)
    Xte = np.asarray(pipe.transform(X_test), dtype=float)
    return Xtr, Xte


def _resolve_config(context: dict[str, Any]) -> dict[str, Any]:
    nested = dict(context.get("autoencoder") or {})

    def pick(key: str, default: Any) -> Any:
        if key in context and context[key] is not None:
            return context[key]
        if key in nested and nested[key] is not None:
            return nested[key]
        return default

    return {
        "use_case": str(pick("use_case", "latent_features")).strip().lower(),
        "latent_dim": int(pick("latent_dim", _DEFAULTS["latent_dim"])),
        "epochs": int(pick("epochs", _DEFAULTS["epochs"])),
        "batch_size": int(pick("batch_size", _DEFAULTS["batch_size"])),
        "hidden_dims": list(pick("hidden_dims", _DEFAULTS["hidden_dims"])),
        "learning_rate": float(pick("learning_rate", _DEFAULTS["learning_rate"])),
        "noise_std": float(pick("noise_std", _DEFAULTS["noise_std"])),
        "latent_mode": str(pick("latent_mode", _DEFAULTS["latent_mode"])).strip().lower(),
    }


def _resolve_primary(primary_metric: str | None, task_type: str, warnings: list[str]) -> str:
    """Garante uma métrica primária computada pelo modelo-sonda."""
    computable = {
        "classification": {"accuracy", "macro_f1", "roc_auc"},
        "regression": {"rmse", "mse", "mae", "r2"},
    }[task_type]
    if primary_metric in computable:
        return primary_metric
    fallback = "macro_f1" if task_type == "classification" else "rmse"
    if primary_metric:
        warnings.append(
            f"Métrica primária '{primary_metric}' não é avaliada pelo modelo-sonda; "
            f"comparando por '{fallback}'."
        )
    return fallback


def _load_dataframe(context: dict[str, Any]) -> pd.DataFrame:
    df = context.get("dataframe")
    if df is not None:
        if not isinstance(df, pd.DataFrame):
            raise TypeError("'dataframe' deve ser um pandas.DataFrame.")
        return df.copy()

    data = context.get("data") or {}
    source_type = str(data.get("source_type", "")).strip().lower()
    source_uri = data.get("source_uri")
    if not source_uri:
        raise ValueError(
            "Forneça 'dataframe' no contexto ou 'data.source_uri' para carregar a base."
        )
    if source_type == "csv":
        return pd.read_csv(source_uri)
    if source_type == "parquet":
        return pd.read_parquet(source_uri)
    raise ValueError(
        f"source_type '{source_type}' não suportado pelo autoencoder "
        "(use 'csv' ou 'parquet', ou passe 'dataframe' diretamente)."
    )


def _resolve_folds(context: dict[str, Any], n_rows: int) -> list[tuple[np.ndarray, np.ndarray]]:
    raw = context.get("folds")
    if raw is None:
        split = context.get("split") or {}
        records = split.get("folds")
        if records:
            raw = [(r["train_idx"], r["test_idx"]) for r in records]
    if not raw:
        raise ValueError(
            "Forneça 'folds' (saída do SplitAgent) ou 'split' com os índices dos folds."
        )
    folds = [(np.asarray(tr, dtype=int), np.asarray(te, dtype=int)) for tr, te in raw]
    for i, (tr, te) in enumerate(folds):
        if tr.max(initial=-1) >= n_rows or te.max(initial=-1) >= n_rows:
            raise ValueError(f"Índices do fold {i} excedem o número de linhas da base.")
    return folds


def _to_native(obj: Any) -> Any:
    """Sanitiza recursivamente para garantir serialização em JSONB."""
    if isinstance(obj, dict):
        return {str(k): _to_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_native(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        f = float(obj)
        return f if math.isfinite(f) else None
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    return obj


def _build_rationale(output: dict[str, Any]) -> str:
    lines = [
        f"Autoencoder '{output['use_case']}' treinado por fold "
        f"({output['n_folds']} fold(s), latent_dim={output['config']['latent_dim']}, "
        f"{output['config']['epochs']} épocas, seed {output['random_seed']}).",
        output["verdict"],
    ]
    lines.append("Ajuste restrito ao treino de cada fold (sem vazamento).")
    if output["warnings"]:
        lines.append("Avisos: " + "; ".join(output["warnings"]))
    return " ".join(lines)

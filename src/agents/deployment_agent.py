"""Agente de Implantação e Monitoramento.

Empacota o modelo recomendado num pipeline serializável (atributos + estimador),
prepara a inferência (endpoint ou batch), monitora drift de dados por PSI e propõe
um ciclo de retreinamento (etapa 10 do fluxo). Conforme o documento, esta etapa
pode ser simplificada no MVP.

Decisão metodológica: o modelo de produção é ajustado na **base completa** (já
limpa) — diferente das etapas de avaliação, que ajustam por fold. O artefato é
serializado com joblib e identificado por hash, e o banco guarda apenas
caminho/hash/metadados (seção 11).
"""

from __future__ import annotations

import datetime
import hashlib
import math
import os
from importlib.metadata import PackageNotFoundError, version
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.pipeline import Pipeline

from src.agents.base import AgentResult, BaseAgent
from src.agents.trainer_agent import _instantiate

_DEFAULT_ARTIFACT_DIR = "artifacts"
_DEFAULT_PSI_THRESHOLD = 0.2  # PSI ≥ 0.2 indica drift significativo (convenção usual)
_DEFAULT_CADENCE_DAYS = 30
_PSI_BINS = 10
_EPS = 1e-6


class DeploymentAgent(BaseAgent):
    """Agente 11 — Implantação e Monitoramento.

    Entradas esperadas em ``context``:
        - ``dataframe`` (pandas.DataFrame): base já limpa (treino de produção).
        - ``pipeline`` (sklearn, opcional): pipeline de atributos NÃO ajustado
          (FeatureAgent); é combinado com o estimador num único pipeline final.
        - Identificação do modelo recomendado (uma das formas):
          ``selected_model`` = {"estimator", "hyperparameters", "model_name"}; ou
          ``evaluation`` (EvaluatorAgent) + ``training``/``model_results`` (Model Zoo).
        - ``task_type`` / ``target_column`` (da Formulação ou ``context['problem']``).
        - ``feature_columns`` (list, opcional), ``random_seed`` (int).

    Sobreposições (opcionais):
        - ``artifact_dir`` (str, padrão "artifacts"): destino do artefato.
        - ``inference_mode``: "batch" (padrão) | "endpoint".
        - ``monitor_data`` (DataFrame): novo lote para um relatório de drift imediato.
        - ``drift_threshold`` (float, padrão 0.2), ``retraining_cadence_days`` (30).

    Saída em ``AgentResult.output``:
        - ``deployment``: artifact_uri, artifact_hash, modelo, params, versões;
        - ``inference``: modo, colunas de entrada, alvo, tipo de predição;
        - ``monitoring``: método (psi), limiar, perfil de referência, drift_report;
        - ``retraining``: política, cadência, gatilhos, próxima revisão; ``warnings``.

    Artefatos vivos em ``AgentResult``: ``model`` (pipeline ajustado),
    ``predict`` (callable) e ``detect_drift`` (callable).
    """

    name = "Agente de Implantação e Monitoramento"
    event_type = "deployment"

    def run(self, context: dict[str, Any]) -> AgentResult:
        warnings: list[str] = []

        df = _load_dataframe(context)
        problem = context.get("problem") or {}
        task_type = str(context.get("task_type") or problem.get("task_type") or "").strip().lower()
        target_column = context.get("target_column") or problem.get("target_column") or None
        if target_column and target_column not in df.columns:
            raise ValueError(f"'target_column' '{target_column}' não existe na base.")
        if task_type in ("classification", "regression") and not target_column:
            raise ValueError(f"'target_column' é obrigatório para task_type='{task_type}'.")

        seed = int(context.get("random_seed", 42))
        feature_columns = context.get("feature_columns") or [
            c for c in df.columns if c != target_column
        ]
        model_name, estimator_path, params = _resolve_selected_model(context)

        # ------------------------------------------------------------------
        # 1. Empacotamento: pipeline final ajustado na base completa
        # ------------------------------------------------------------------
        feature_pipeline = context.get("pipeline")
        estimator, params = _instantiate(estimator_path, {"default_params": params}, seed)
        steps: list[tuple[str, Any]] = []
        if feature_pipeline is not None:
            steps.extend(clone(feature_pipeline).steps)
        steps.append(("model", estimator))
        model = Pipeline(steps)

        X = df[feature_columns]
        y = df[target_column] if target_column else None
        model.fit(X, y)

        artifact_dir = context.get("artifact_dir", _DEFAULT_ARTIFACT_DIR)
        artifact_uri, artifact_hash = _package(model, model_name, artifact_dir, warnings)

        deployment = {
            "model_name": model_name,
            "estimator": estimator_path,
            "hyperparameters": params,
            "feature_columns": feature_columns,
            "n_train_rows": int(len(df)),
            "train_data_hash": _content_hash(df),
            "random_seed": seed,
            "artifact_uri": artifact_uri,
            "artifact_hash": artifact_hash,
            "packaged_at": datetime.datetime.now().isoformat(timespec="seconds"),
            "library_versions": _versions(),
        }

        # ------------------------------------------------------------------
        # 2. Inferência (endpoint / batch)
        # ------------------------------------------------------------------
        inference = {
            "mode": str(context.get("inference_mode", "batch")).strip().lower(),
            "input_columns": feature_columns,
            "target": target_column,
            "prediction_type": _prediction_type(task_type),
            "supports_proba": hasattr(model, "predict_proba"),
        }

        # ------------------------------------------------------------------
        # 3. Monitoramento de drift (PSI sobre os atributos)
        # ------------------------------------------------------------------
        threshold = float(context.get("drift_threshold", _DEFAULT_PSI_THRESHOLD))
        reference = _DriftReference.fit(df, feature_columns, threshold)
        monitoring: dict[str, Any] = {
            "method": "psi",
            "drift_threshold": threshold,
            "reference_profile": reference.to_dict(),
            "drift_report": None,
        }
        monitor_data = context.get("monitor_data")
        if monitor_data is not None:
            monitoring["drift_report"] = reference.evaluate(monitor_data)

        # ------------------------------------------------------------------
        # 4. Plano de retreinamento
        # ------------------------------------------------------------------
        cadence = int(context.get("retraining_cadence_days", _DEFAULT_CADENCE_DAYS))
        next_review = (datetime.date.today() + datetime.timedelta(days=cadence)).isoformat()
        retraining = {
            "policy": "Retreinar por cadência fixa ou quando um gatilho disparar.",
            "cadence_days": cadence,
            "next_review_date": next_review,
            "triggers": [
                f"Drift de dados: PSI ≥ {threshold} em qualquer atributo monitorado.",
                "Queda da métrica primária abaixo do critério de sucesso em produção.",
                f"Janela temporal: a cada {cadence} dias, no mínimo.",
            ],
        }

        output: dict[str, Any] = {
            "deployment": deployment,
            "inference": inference,
            "monitoring": monitoring,
            "retraining": retraining,
            "warnings": warnings,
        }
        output = _to_native(output)

        rationale = _build_rationale(output)
        result = AgentResult(output=output, rationale=rationale, warnings=warnings)
        result.model = model
        result.predict = lambda new_df: model.predict(new_df[feature_columns])
        result.detect_drift = reference.evaluate
        return result


# ---------------------------------------------------------------------------
# Monitor de drift (PSI) — serializável
# ---------------------------------------------------------------------------

class _DriftReference:
    """Perfil de referência (treino) para detectar drift via PSI por atributo."""

    def __init__(self, specs: dict[str, dict[str, Any]], threshold: float) -> None:
        self.specs = specs
        self.threshold = threshold

    @classmethod
    def fit(cls, df: pd.DataFrame, feature_columns: list[str], threshold: float) -> "_DriftReference":
        specs: dict[str, dict[str, Any]] = {}
        for col in feature_columns:
            series = df[col].dropna()
            if series.empty:
                continue
            if pd.api.types.is_bool_dtype(series) or not pd.api.types.is_numeric_dtype(series):
                counts = series.astype(str).value_counts(normalize=True)
                specs[col] = {"type": "categorical", "expected": counts.to_dict()}
            elif pd.api.types.is_numeric_dtype(series):
                edges = np.unique(np.quantile(series, np.linspace(0, 1, _PSI_BINS + 1)))
                if len(edges) < 2:
                    continue  # coluna ~constante: não monitora drift
                # Bins externos abertos (±∞): valores fora do intervalo de treino
                # não são descartados (evita PSI inflado nas caudas).
                edges[0], edges[-1] = -np.inf, np.inf
                expected = _hist_props(series.to_numpy(), edges)
                specs[col] = {
                    "type": "numeric",
                    "edges": edges.tolist(),
                    "expected": expected.tolist(),
                }
        return cls(specs, threshold)

    def evaluate(self, new_df: pd.DataFrame) -> dict[str, Any]:
        per_column: dict[str, float] = {}
        for col, spec in self.specs.items():
            if col not in new_df.columns:
                continue
            series = new_df[col].dropna()
            if series.empty:
                continue
            if spec["type"] == "numeric":
                edges = np.asarray(spec["edges"], dtype=float)
                actual = _hist_props(series.to_numpy(), edges)
                psi = _psi(np.asarray(spec["expected"], dtype=float), actual)
            else:
                psi = _psi_categorical(spec["expected"], series.astype(str))
            per_column[col] = round(float(psi), 6)

        drifted = sorted(c for c, v in per_column.items() if v >= self.threshold)
        return {
            "columns": per_column,
            "drifted_columns": drifted,
            "n_drifted": len(drifted),
            "max_psi": round(max(per_column.values()), 6) if per_column else 0.0,
            "drift_detected": bool(drifted),
        }

    def to_dict(self) -> dict[str, Any]:
        return {"threshold": self.threshold, "columns": list(self.specs), "specs": self.specs}


def _hist_props(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    counts, _ = np.histogram(values, bins=edges)
    props = counts / counts.sum() if counts.sum() else counts.astype(float)
    return np.clip(props, _EPS, None)


def _psi(expected: np.ndarray, actual: np.ndarray) -> float:
    expected = np.clip(expected, _EPS, None)
    actual = np.clip(actual, _EPS, None)
    return float(np.sum((actual - expected) * np.log(actual / expected)))


def _psi_categorical(expected: dict[str, float], actual_series: pd.Series) -> float:
    actual = actual_series.value_counts(normalize=True).to_dict()
    categories = set(expected) | set(actual)
    e = np.array([expected.get(c, _EPS) for c in categories], dtype=float)
    a = np.array([actual.get(c, _EPS) for c in categories], dtype=float)
    return _psi(e, a)


# ---------------------------------------------------------------------------
# Empacotamento e seleção do modelo
# ---------------------------------------------------------------------------

def _resolve_selected_model(context: dict[str, Any]) -> tuple[str, str, dict[str, Any]]:
    """Identifica o modelo a implantar via 'selected_model' ou evaluation+training."""
    selected = context.get("selected_model")
    if selected:
        if not selected.get("estimator"):
            raise ValueError("'selected_model' precisa de 'estimator' (ex.: 'modulo:Classe').")
        return (
            selected.get("model_name", selected["estimator"]),
            selected["estimator"],
            dict(selected.get("hyperparameters") or {}),
        )

    evaluation = context.get("evaluation") or {}
    best_name = evaluation.get("best_model")
    if not best_name:
        raise ValueError(
            "Forneça 'selected_model' ou 'evaluation' (com 'best_model') para escolher o modelo."
        )
    training = context.get("training") or {}
    results = context.get("model_results") or training.get("results") or []
    res = next((r for r in results if r.get("model_name") == best_name), None)
    if not res or not res.get("estimator"):
        raise ValueError(
            f"Modelo '{best_name}' não encontrado em 'training'/'model_results' com 'estimator'."
        )
    return best_name, res["estimator"], dict(res.get("hyperparameters") or {})


def _package(
    model: Pipeline, model_name: str, artifact_dir: str, warnings: list[str]
) -> tuple[str | None, str | None]:
    """Serializa o modelo com joblib e devolve (caminho, hash sha256)."""
    try:
        os.makedirs(artifact_dir, exist_ok=True)
        path = os.path.join(artifact_dir, f"{model_name}.joblib")
        joblib.dump(model, path)
        with open(path, "rb") as fh:
            digest = hashlib.sha256(fh.read()).hexdigest()
        return path, digest
    except OSError as exc:
        warnings.append(f"Falha ao serializar o artefato em '{artifact_dir}': {exc}.")
        return None, None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _prediction_type(task_type: str) -> str:
    return {
        "classification": "class_label",
        "regression": "continuous_value",
        "anomaly": "anomaly_score",
    }.get(task_type, "unknown")


def _content_hash(df: pd.DataFrame) -> str:
    try:
        values = pd.util.hash_pandas_object(df, index=False).values
        return hashlib.sha256(values.tobytes()).hexdigest()
    except Exception:  # pragma: no cover - fallback robusto p/ dtypes exóticos
        return hashlib.sha256(
            pd.util.hash_pandas_object(df.astype(str), index=False).values.tobytes()
        ).hexdigest()


def _versions() -> dict[str, str]:
    out: dict[str, str] = {}
    for pkg in ("scikit-learn", "pandas", "numpy", "torch", "joblib"):
        try:
            out[pkg] = version(pkg)
        except PackageNotFoundError:
            continue
    return out


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
        f"source_type '{source_type}' não suportado pela implantação "
        "(use 'csv' ou 'parquet', ou passe 'dataframe' diretamente)."
    )


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
    dep = output["deployment"]
    mon = output["monitoring"]
    lines = [
        f"Modelo '{dep['model_name']}' empacotado ({dep['n_train_rows']} linhas de treino, "
        f"seed {dep['random_seed']}).",
    ]
    if dep["artifact_uri"]:
        lines.append(f"Artefato: {dep['artifact_uri']} (hash {str(dep['artifact_hash'])[:12]}…).")
    lines.append(
        f"Inferência em modo '{output['inference']['mode']}'; "
        f"monitoramento de drift por PSI (limiar {mon['drift_threshold']})."
    )
    report = mon.get("drift_report")
    if report:
        status = "DETECTADO" if report["drift_detected"] else "não detectado"
        lines.append(f"Drift {status} ({report['n_drifted']} atributo(s), PSI máx {report['max_psi']}).")
    lines.append(f"Retreinamento sugerido até {output['retraining']['next_review_date']}.")
    if output["warnings"]:
        lines.append("Avisos: " + "; ".join(output["warnings"]))
    return " ".join(lines)

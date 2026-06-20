"""Pipeline de treinamento — ponto de entrada de um experimento.

Carrega a configuração YAML, instancia os agentes na ordem do fluxo agentivo
(seção 5), propaga o contexto entre etapas e persiste eventos, métricas e
artefatos no PostgreSQL.

Reprodutibilidade (RNF01/RF15): a mesma ``seed`` e a mesma configuração produzem
os mesmos folds e as mesmas métricas. Anti-leakage: o pipeline de atributos é
ajustado apenas no treino de cada fold pelos agentes a jusante.

Uso:
    python -m src.pipelines.training --config configs/experiment_example.yaml
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from src.agents.autoencoder_agent import AutoencoderAgent
from src.agents.cleaning_agent import CleaningAgent
from src.agents.data_profile_agent import DataProfileAgent
from src.agents.deployment_agent import DeploymentAgent
from src.agents.evaluator_agent import EvaluatorAgent
from src.agents.feature_agent import FeatureAgent
from src.agents.optimization_agent import OptimizationAgent
from src.agents.problem_agent import ProblemAgent
from src.agents.report_agent import ReportAgent
from src.agents.split_agent import SplitAgent
from src.agents.trainer_agent import TrainerAgent

# Mapeia nomes de codificação da config para os aceitos pelo FeatureAgent.
_ENCODING_ALIASES = {"one_hot": "onehot", "onehot": "onehot", "ordinal": "ordinal"}


def load_config(path: str | Path) -> dict[str, Any]:
    """Lê o arquivo de configuração do experimento."""
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def run_experiment(
    config: dict[str, Any],
    dataframe: pd.DataFrame | None = None,
    *,
    event_sink: Any | None = None,
    recorder: Any | None = None,
) -> dict[str, Any]:
    """Executa o experimento de ponta a ponta a partir da configuração.

    Args:
        config: configuração no formato de ``configs/experiment_example.yaml``.
        dataframe: base já carregada; se ausente, é lida de ``config['data']``.
        event_sink: destino dos eventos agentivos (RNF03). Se ``None``, os agentes
            não emitem eventos (modo em memória, útil em testes).
        recorder: :class:`~src.pipelines.persistence.RunRecorder` (ou compatível)
            para gravar experimento/run/dataset/model_results/report. Opcional.

    Returns:
        Dicionário com as saídas de cada etapa, o modelo recomendado, o ranking e
        o relatório em Markdown, além de ``experiment_id``/``run_id`` quando há
        persistência.
    """
    exp = config["experiment"]
    name = exp.get("name", "experimento")
    task_type = str(exp["task_type"]).strip().lower()
    target_column = exp.get("target_column")
    primary_metric = exp.get("primary_metric")
    seed = int(exp.get("random_seed", 42))

    data_cfg = config.get("data", {}) or {}
    val_cfg = config.get("validation", {}) or {}
    prep_cfg = config.get("preprocessing", {}) or {}
    models_cfg = config.get("models", {}) or {}
    ae_cfg = config.get("autoencoder", {}) or {}
    opt_cfg = config.get("optimization", {}) or {}
    deploy_cfg = config.get("deployment", {}) or {}
    storage_cfg = config.get("storage", {}) or {}

    df = dataframe.copy() if dataframe is not None else _load_dataframe(data_cfg)

    # ------------------------------------------------------------------
    # Persistência: cria experimento e run (se houver recorder)
    # ------------------------------------------------------------------
    experiment_id = None
    run_id = None
    if recorder is not None:
        experiment_id = recorder.create_experiment(
            name=name, task_type=task_type, target_column=target_column,
            primary_metric=primary_metric, config=config,
        )

    def context(stage: str, **kwargs: Any) -> dict[str, Any]:
        """Contexto base com identificação para o evento agentivo.

        Chaves com valor ``None`` são descartadas: para os agentes, ausência e
        ``None`` são equivalentes, mas alguns aplicam ``float(get(k, default))`` e
        receberiam ``None`` se a chave estivesse presente.
        """
        ctx = {
            "experiment_id": experiment_id,
            "random_seed": seed,
            "agent_input": {"stage": stage, "experiment": name},
        }
        ctx.update({k: v for k, v in kwargs.items() if v is not None})
        return ctx

    # ------------------------------------------------------------------
    # 1. Formulação do problema
    # ------------------------------------------------------------------
    problem = ProblemAgent(event_sink)(context(
        "problem",
        task_type=task_type,
        target_column=target_column,
        primary_metric=primary_metric,
        split_strategy=val_cfg.get("split_strategy"),
        success_threshold=exp.get("success_threshold"),
        constraints=exp.get("constraints"),
        time_column=data_cfg.get("time_column"),
        group_column=data_cfg.get("group_column"),
    )).output
    primary_metric = problem["primary_metric"]  # pode ter sido preenchido por padrão

    # ------------------------------------------------------------------
    # 2. Ingestão e perfilamento
    # ------------------------------------------------------------------
    profile = DataProfileAgent(event_sink)(context(
        "profile",
        dataframe=df,
        target_column=target_column,
        task_type=task_type,
        id_column=data_cfg.get("id_column"),
        time_column=data_cfg.get("time_column"),
        group_column=data_cfg.get("group_column"),
        rare_category_threshold=prep_cfg.get("rare_category_threshold"),
    )).output

    # ------------------------------------------------------------------
    # 3. Validação e limpeza
    # ------------------------------------------------------------------
    cleaning_res = CleaningAgent(event_sink)(context(
        "cleaning",
        dataframe=df,
        target_column=target_column,
        data_profile=profile,
        numeric_imputation=prep_cfg.get("numeric_imputation", "median"),
        categorical_imputation=prep_cfg.get("categorical_imputation", "most_frequent"),
        rare_category_threshold=prep_cfg.get("rare_category_threshold", 0.01),
    ))
    clean_df = cleaning_res.cleaned_dataframe
    cleaning = cleaning_res.output

    if recorder is not None:
        recorder.save_dataset(
            experiment_id=experiment_id,
            name=data_cfg.get("name", name),
            source_type=data_cfg.get("source_type", "dataframe"),
            source_uri=data_cfg.get("source_uri"),
            content_hash=profile.get("content_hash"),
            schema_json=profile.get("schema"),
            profile_json=profile,
            quality_report_json=cleaning.get("quality_report"),
        )
        run_id = recorder.create_run(
            experiment_id=experiment_id, seed=seed,
            code_version=_git_version(), data_version=profile.get("content_hash"),
        )

    # ------------------------------------------------------------------
    # 4. Engenharia de atributos (pipeline NÃO ajustado)
    # ------------------------------------------------------------------
    feature_res = FeatureAgent(event_sink)(context(
        "features",
        dataframe=clean_df,
        target_column=target_column,
        data_profile=profile,
        numeric_imputation=prep_cfg.get("numeric_imputation", "median"),
        categorical_imputation=prep_cfg.get("categorical_imputation", "most_frequent"),
        scaling=prep_cfg.get("scaling", "standard"),
        categorical_encoding=_ENCODING_ALIASES.get(
            str(prep_cfg.get("categorical_encoding", "onehot")).lower(), "onehot"
        ),
    ))
    pipeline = feature_res.pipeline
    features = feature_res.output

    # ------------------------------------------------------------------
    # 5. Split e validação cruzada
    # ------------------------------------------------------------------
    split_res = SplitAgent(event_sink)(context(
        "split",
        dataframe=clean_df,
        split_strategy=problem.get("split_strategy"),
        task_type=task_type,
        target_column=target_column,
        time_column=problem.get("time_column"),
        group_column=problem.get("group_column"),
        n_splits=val_cfg.get("n_splits", 5),
        test_size=val_cfg.get("test_size", 0.2),
    ))
    folds = split_res.folds
    split = split_res.output

    # ------------------------------------------------------------------
    # 6. Treinamento cruzado (model zoo)
    # ------------------------------------------------------------------
    include = models_cfg.get("include")
    training = TrainerAgent(event_sink)(context(
        "training",
        dataframe=clean_df,
        task_type=task_type,
        target_column=target_column,
        primary_metric=primary_metric,
        folds=folds,
        pipeline=pipeline,
        include=include,
        tiers=models_cfg.get("tiers"),
        time_budget_seconds=models_cfg.get("time_budget_seconds"),
    )).output

    # ------------------------------------------------------------------
    # 7. Autoencoder (opcional)
    # ------------------------------------------------------------------
    autoencoder = None
    if ae_cfg.get("enabled"):
        autoencoder = AutoencoderAgent(event_sink)(context(
            "autoencoder",
            dataframe=clean_df,
            task_type=task_type,
            target_column=target_column,
            primary_metric=primary_metric,
            folds=folds,
            pipeline=pipeline,
            autoencoder=ae_cfg,
        )).output

    # ------------------------------------------------------------------
    # 8. Otimização de hiperparâmetros (opcional)
    # ------------------------------------------------------------------
    optimization = None
    if opt_cfg.get("enabled"):
        optimization = OptimizationAgent(event_sink)(context(
            "optimization",
            dataframe=clean_df,
            task_type=task_type,
            target_column=target_column,
            primary_metric=primary_metric,
            folds=folds,
            pipeline=pipeline,
            include=include,
            n_trials=opt_cfg.get("n_trials", 20),
            timeout_seconds=opt_cfg.get("timeout_seconds"),
        )).output

    # ------------------------------------------------------------------
    # 9. Avaliação e seleção
    # ------------------------------------------------------------------
    evaluation = EvaluatorAgent(event_sink)(context(
        "evaluation",
        model_results=training["results"],
        task_type=task_type,
        primary_metric=primary_metric,
        success_threshold=problem.get("success_threshold"),
        selection_rule=config.get("selection_rule", "best_mean"),
        dataframe=clean_df,
        folds=folds,
        pipeline=pipeline,
        target_column=target_column,
    )).output

    # ------------------------------------------------------------------
    # 10. Implantação (opcional)
    # ------------------------------------------------------------------
    deployment = None
    if deploy_cfg.get("enabled"):
        deployment = DeploymentAgent(event_sink)(context(
            "deployment",
            dataframe=clean_df,
            task_type=task_type,
            target_column=target_column,
            pipeline=pipeline,
            evaluation=evaluation,
            training=training,
            artifact_dir=storage_cfg.get("artifact_dir", "artifacts"),
            inference_mode=deploy_cfg.get("inference_mode", "batch"),
        )).output

    # ------------------------------------------------------------------
    # 11. Relatório técnico e auditoria
    # ------------------------------------------------------------------
    report_res = ReportAgent(event_sink)(context(
        "report",
        experiment_name=name,
        problem=problem,
        profile=profile,
        cleaning=cleaning,
        features=features,
        split=split,
        training=training,
        optimization=optimization,
        autoencoder=autoencoder,
        evaluation=evaluation,
        dataframe=clean_df,
        folds=folds,
        pipeline=pipeline,
        target_column=target_column,
    ))
    report = report_res.output

    # ------------------------------------------------------------------
    # Persistência final
    # ------------------------------------------------------------------
    run_metrics = {
        "best_model": evaluation.get("best_model"),
        "primary_metric": primary_metric,
        "best_primary_mean": evaluation.get("best_primary_mean"),
        "meets_success_threshold": evaluation.get("meets_success_threshold"),
    }
    if recorder is not None:
        artifacts = (
            {deployment["deployment"]["model_name"]: deployment["deployment"]}
            if deployment else None
        )
        recorder.save_model_results(
            run_id=run_id, results=training["results"], artifacts=artifacts
        )
        recorder.save_report(
            experiment_id=experiment_id, run_id=run_id,
            content=report["report_markdown"], summary_json=report["report_json"],
        )
        recorder.finish_run(run_id=run_id, status="completed", metrics_json=run_metrics)
        recorder.set_experiment_status(experiment_id=experiment_id, status="completed")

    return {
        "experiment_id": experiment_id,
        "run_id": run_id,
        "seed": seed,
        "problem": problem,
        "profile": profile,
        "cleaning": cleaning,
        "features": features,
        "split": split,
        "training": training,
        "autoencoder": autoencoder,
        "optimization": optimization,
        "evaluation": evaluation,
        "deployment": deployment,
        "report": report,
        "best_model": evaluation.get("best_model"),
        "ranking": evaluation.get("ranking"),
        "report_markdown": report["report_markdown"],
        "metrics": run_metrics,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_dataframe(data_cfg: dict[str, Any]) -> pd.DataFrame:
    source_type = str(data_cfg.get("source_type", "")).strip().lower()
    source_uri = data_cfg.get("source_uri")
    if not source_uri:
        raise ValueError("Sem 'dataframe' e sem 'data.source_uri' na configuração.")
    if source_type == "csv":
        return pd.read_csv(source_uri)
    if source_type == "parquet":
        return pd.read_parquet(source_uri)
    raise ValueError(
        f"source_type '{source_type}' não suportado (use 'csv' ou 'parquet')."
    )


def _git_version() -> str | None:
    """Hash curto do commit atual, para rastreabilidade (RNF01). Best-effort."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5, check=False,
        )
        return out.stdout.strip() or None
    except Exception:  # noqa: BLE001 - ambiente sem git não deve quebrar o run
        return None


def _build_recorder(config: dict[str, Any]):
    """Constrói um RunRecorder + EventSink a partir de DATABASE_URL (path real)."""
    import os

    from src.db.models import get_session_factory
    from src.pipelines.persistence import RunRecorder

    storage = config.get("storage", {}) or {}
    env_var = storage.get("postgres_uri_env", "DATABASE_URL")
    database_url = os.environ.get(env_var)
    if not database_url:
        return None
    return RunRecorder(get_session_factory(database_url))


def main() -> None:
    parser = argparse.ArgumentParser(description="Executa um experimento do pipeline.")
    parser.add_argument("--config", required=True, help="Caminho do YAML de configuração.")
    parser.add_argument(
        "--no-persist", action="store_true", help="Não grava no PostgreSQL."
    )
    args = parser.parse_args()
    config = load_config(args.config)

    recorder = None if args.no_persist else _build_recorder(config)
    event_sink = recorder.event_sink if recorder is not None else None
    result = run_experiment(config, event_sink=event_sink, recorder=recorder)

    best = result["best_model"]
    print(f"Experimento '{config['experiment'].get('name')}' concluído.")
    print(f"Modelo recomendado: {best}")
    if result["experiment_id"]:
        print(f"experiment_id={result['experiment_id']} run_id={result['run_id']}")


if __name__ == "__main__":
    main()

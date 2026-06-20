"""Agentes do pipeline.

Cada agente é um papel lógico separado, com estado intermediário persistido e
decisões justificadas (seção 6 do documento de apoio).
"""

from src.agents.autoencoder_agent import AutoencoderAgent
from src.agents.base import AgentResult, BaseAgent
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

__all__ = [
    "BaseAgent",
    "AgentResult",
    "ProblemAgent",
    "DataProfileAgent",
    "CleaningAgent",
    "FeatureAgent",
    "SplitAgent",
    "TrainerAgent",
    "AutoencoderAgent",
    "OptimizationAgent",
    "EvaluatorAgent",
    "ReportAgent",
    "DeploymentAgent",
]

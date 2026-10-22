"""
FTC-XAI: Formalized Trust-Calibrated Explainable AI
A Unified, Clinically Validated Framework Linking Explanation Quality to Clinician Trust and Decision Performance
"""

__version__ = "1.0.0"
__author__ = "Aaqib Rashid Mir, Ridwan Uz Zaman Pandit"

from .core import ExplanationPayload, ProvenanceTrace
from .metrics import ExplanationQualityMetrics
from .trust_engine import BayesianTrustEngine, TrustCalibrationMetrics
from .federated_audit import FederatedEquityAuditor
from .explainers import ClinicallyGuidedLIME, MonotonicConstrainedExplainer

__all__ = [
    "ExplanationPayload",
    "ProvenanceTrace",
    "ExplanationQualityMetrics",
    "BayesianTrustEngine",
    "TrustCalibrationMetrics",
    "FederatedEquityAuditor",
    "ClinicallyGuidedLIME",
    "MonotonicConstrainedExplainer",
]

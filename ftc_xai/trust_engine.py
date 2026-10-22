"""
Dynamic Trust-Calibration Engine for FTC-XAI (Extended TAXAI).
Maintains per-clinician state-space Bayesian latent trust parameters,
computes TECE, ORI, URI, and manages real-time guardrail alerts.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Any
import numpy as np


@dataclass
class ClinicianTrustState:
    """State-space latent trust representation for a single clinician."""
    clinician_id: str
    seniority: str  # 'junior' or 'senior'
    alpha: float = 5.0
    beta_param: float = 5.0
    history: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def expected_trust(self) -> float:
        """E[theta] = alpha / (alpha + beta)"""
        return self.alpha / (self.alpha + self.beta_param)

    @property
    def variance(self) -> float:
        """Var[theta] = (alpha * beta) / ((alpha + beta)^2 * (alpha + beta + 1))"""
        a, b = self.alpha, self.beta_param
        return (a * b) / ((a + b) ** 2 * (a + b + 1))


class BayesianTrustEngine:
    """
    Implements the quality-weighted Bayesian state-space trust update rule:
    Updates alpha and beta based on (Model Correctness, Clinician Action, Explanation Quality).
    """

    def __init__(
        self,
        eta: float = 1.0,     # Weight for Appropriate Reliance
        lambda_param: float = 0.8, # Weight for Appropriate Override
        kappa: float = 2.5,   # Heavy penalty for Over-Reliance (Automation Trap)
        psi: float = 1.2,     # Penalty for Under-Reliance (Algorithmic Cynicism)
        warning_quality_threshold: float = 0.55,
        warning_lipschitz_threshold: float = 2.5,
    ):
        self.eta = eta
        self.lambda_param = lambda_param
        self.kappa = kappa
        self.psi = psi
        self.warning_q_thresh = warning_quality_threshold
        self.warning_lip_thresh = warning_lipschitz_threshold
        self.clinicians: Dict[str, ClinicianTrustState] = {}

    def get_or_create_clinician(
        self,
        clinician_id: str,
        seniority: str = "senior",
        init_alpha: float = 5.0,
        init_beta: float = 5.0,
    ) -> ClinicianTrustState:
        if clinician_id not in self.clinicians:
            self.clinicians[clinician_id] = ClinicianTrustState(
                clinician_id=clinician_id,
                seniority=seniority,
                alpha=init_alpha,
                beta_param=init_beta,
            )
        return self.clinicians[clinician_id]

    def check_guardrail_warning(
        self,
        quality_score: float,
        lipschitz_val: float,
        plausibility_dice: float,
    ) -> Tuple[bool, str]:
        """
        Determines whether an explanation fails quality thresholds,
        triggering a safety alert before clinician presentation.
        """
        reasons = []
        if quality_score < self.warning_q_thresh:
            reasons.append(f"Composite quality Q={quality_score:.2f} below threshold {self.warning_q_thresh}")
        if lipschitz_val > self.warning_lip_thresh:
            reasons.append(f"Perturbation instability Lip={lipschitz_val:.2f} exceeds ceiling {self.warning_lip_thresh}")
        if plausibility_dice < 0.40:
            reasons.append(f"Anatomical Dice={plausibility_dice:.2f} indicates off-target attribution")

        if reasons:
            msg = "WARNING: Explanation unverified by FTC-XAI guardrails (" + "; ".join(reasons) + "). Review raw diagnostic data independently."
            return True, msg
        return False, "Explanation verified by FTC-XAI guardrails."

    def update_encounter(
        self,
        clinician_id: str,
        case_id: str,
        model_pred: int,
        ground_truth: int,
        clinician_action: int,  # 1 = accepted model suggestion, 0 = overridden
        explanation_quality: float,
        confidence: float,
        latency_sec: float,
    ) -> Dict[str, Any]:
        """
        Executes a single encounter Bayesian update for a clinician.
        """
        state = self.get_or_create_clinician(clinician_id)
        model_correct = (model_pred == ground_truth)
        q = float(np.clip(explanation_quality, 0.0, 1.0))

        # Determine outcome state
        if model_correct and clinician_action == 1:
            state_label = "Appropriate_Reliance"
            state.alpha += self.eta * q
        elif (not model_correct) and clinician_action == 1:
            state_label = "Over_Reliance"  # Automation bias!
            state.beta_param += self.kappa
        elif model_correct and clinician_action == 0:
            state_label = "Under_Reliance"  # Algorithmic cynicism!
            state.beta_param += self.psi * q
        else:
            state_label = "Appropriate_Override"  # Doctor caught model error!
            state.alpha += self.lambda_param * (1.0 - q)

        encounter_log = {
            "case_id": case_id,
            "model_pred": model_pred,
            "ground_truth": ground_truth,
            "clinician_action": clinician_action,
            "outcome_state": state_label,
            "explanation_quality": q,
            "model_confidence": confidence,
            "latency_sec": latency_sec,
            "posterior_expected_trust": state.expected_trust,
            "posterior_alpha": state.alpha,
            "posterior_beta": state.beta_param,
        }
        state.history.append(encounter_log)
        return encounter_log


class TrustCalibrationMetrics:
    """
    Computes population and per-cohort trust calibration indices:
    - TECE: Trust Expected Calibration Error
    - ORI: Over-Reliance Index
    - URI: Under-Reliance Index
    """

    @staticmethod
    def compute_tece(
        confidences: np.ndarray,
        actions: np.ndarray,
        model_correct: np.ndarray,
        n_bins: int = 5,
    ) -> float:
        """
        Calculates TECE = sum (|B_m| / N) * | AccRate(B_m) - TrueAcc(B_m) |
        """
        confidences = np.asarray(confidences)
        actions = np.asarray(actions)
        model_correct = np.asarray(model_correct)

        bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
        tece = 0.0
        n_total = len(confidences)
        if n_total == 0:
            return 0.0

        for i in range(n_bins):
            in_bin = (confidences >= bin_edges[i]) & (confidences <= bin_edges[i + 1])
            bin_size = np.sum(in_bin)
            if bin_size > 0:
                acc_rate = np.mean(actions[in_bin])
                true_acc = np.mean(model_correct[in_bin])
                tece += (bin_size / n_total) * abs(acc_rate - true_acc)

        return float(tece)

    @staticmethod
    def compute_reliance_indices(
        actions: np.ndarray,
        model_correct: np.ndarray,
    ) -> Tuple[float, float]:
        """
        Returns (ORI, URI):
        ORI = P(action=1 | model_correct=False)
        URI = P(action=0 | model_correct=True)
        """
        actions = np.asarray(actions)
        correct = np.asarray(model_correct)

        err_mask = (~correct)
        if np.sum(err_mask) > 0:
            ori = float(np.mean(actions[err_mask] == 1))
        else:
            ori = 0.0

        corr_mask = correct
        if np.sum(corr_mask) > 0:
            uri = float(np.mean(actions[corr_mask] == 0))
        else:
            uri = 0.0

        return ori, uri

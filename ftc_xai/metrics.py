"""
Explanation Quality Metric Suite for FTC-XAI.
Computes mathematical axes of Fidelity, Local Stability, and Clinical Plausibility.
"""

from typing import Callable, Optional, Tuple, Dict, Any
import numpy as np
from scipy.spatial.distance import cdist
try:
    from scipy.integrate import trapezoid
except ImportError:
    trapezoid = getattr(np, 'trapezoid', getattr(np, 'trapz', None))



class ExplanationQualityMetrics:
    """
    Evaluates an explanation against the formal 3-axis quality suite:
    Axis 1: Fidelity (AUC_del, AUC_ins, Infidelity)
    Axis 2: Numerical Stability (Lipschitz constant Lip_delta)
    Axis 3: Clinical Plausibility (Dice alignment with clinical masks)
    """

    def __init__(
        self,
        weights: Tuple[float, float, float, float] = (0.30, 0.25, 0.25, 0.20),
        plausibility_threshold_pct: float = 80.0,
        lipschitz_radius: float = 0.05,
        n_stability_samples: int = 4,
        random_seed: int = 42,
    ):
        self.weights = weights
        assert abs(sum(weights) - 1.0) < 1e-4, "Quality metric weights must sum to 1.0"
        self.plaus_thresh_pct = plausibility_threshold_pct
        self.lipschitz_radius = lipschitz_radius
        self.n_stability_samples = n_stability_samples
        self.rng = np.random.default_rng(random_seed)

    def deletion_auc(
        self,
        model_predict_fn: Callable[[np.ndarray], np.ndarray],
        x: np.ndarray,
        attributions: np.ndarray,
        reference_baseline: Optional[np.ndarray] = None,
        steps: int = 10,
    ) -> float:
        """
        Calculates Deletion AUC (AUC_del). Features are removed in descending
        order of attribution importance; faster drop = lower AUC = higher fidelity.
        """
        x_flat = x.flatten()
        attr_flat = np.abs(attributions.flatten())
        if reference_baseline is None:
            ref_flat = np.zeros_like(x_flat)
        else:
            ref_flat = reference_baseline.flatten()

        sorted_indices = np.argsort(-attr_flat)
        n_features = len(x_flat)
        step_size = max(1, n_features // steps)

        x_perturbed = x_flat.copy()
        pred_history = []

        # Initial prediction (full input)
        base_pred = model_predict_fn(x_perturbed.reshape(x.shape))[0]
        pred_history.append(base_pred)

        for i in range(0, n_features, step_size):
            end_idx = min(i + step_size, n_features)
            remove_idx = sorted_indices[i:end_idx]
            x_perturbed[remove_idx] = ref_flat[remove_idx]
            curr_pred = model_predict_fn(x_perturbed.reshape(x.shape))[0]
            pred_history.append(curr_pred)

        # Trapz integration over fraction of removed features
        fractions = np.linspace(0.0, 1.0, len(pred_history))
        auc_del = float(trapezoid(pred_history, fractions))
        return float(np.clip(auc_del, 0.0, 1.0))

    def insertion_auc(
        self,
        model_predict_fn: Callable[[np.ndarray], np.ndarray],
        x: np.ndarray,
        attributions: np.ndarray,
        reference_baseline: Optional[np.ndarray] = None,
        steps: int = 10,
    ) -> float:
        """
        Calculates Insertion AUC (AUC_ins). Features are re-introduced
        into a blank baseline in descending order of attribution importance.
        Faster rise = higher AUC = higher fidelity.
        """
        x_flat = x.flatten()
        attr_flat = np.abs(attributions.flatten())
        if reference_baseline is None:
            ref_flat = np.zeros_like(x_flat)
        else:
            ref_flat = reference_baseline.flatten()

        sorted_indices = np.argsort(-attr_flat)
        n_features = len(x_flat)
        step_size = max(1, n_features // steps)

        x_current = ref_flat.copy()
        pred_history = []

        # Baseline prediction
        init_pred = model_predict_fn(x_current.reshape(x.shape))[0]
        pred_history.append(init_pred)

        for i in range(0, n_features, step_size):
            end_idx = min(i + step_size, n_features)
            insert_idx = sorted_indices[i:end_idx]
            x_current[insert_idx] = x_flat[insert_idx]
            curr_pred = model_predict_fn(x_current.reshape(x.shape))[0]
            pred_history.append(curr_pred)

        fractions = np.linspace(0.0, 1.0, len(pred_history))
        auc_ins = float(trapezoid(pred_history, fractions))
        return float(np.clip(auc_ins, 0.0, 1.0))

    def algorithmic_infidelity(
        self,
        model_predict_fn: Callable[[np.ndarray], np.ndarray],
        x: np.ndarray,
        attributions: np.ndarray,
        n_perturbations: int = 25,
        noise_scale: float = 0.05,
    ) -> float:
        """
        Calculates Infidelity: MSE between linear attribution projection and true model output delta.
        """
        x_flat = x.flatten()
        phi_flat = attributions.flatten()
        base_pred = model_predict_fn(x)[0]

        infid_errors = []
        for _ in range(n_perturbations):
            noise = self.rng.normal(0, noise_scale, size=x_flat.shape)
            x_noisy = (x_flat - noise).reshape(x.shape)
            noisy_pred = model_predict_fn(x_noisy)[0]
            true_delta = base_pred - noisy_pred
            linear_pred = float(np.dot(phi_flat, noise))
            infid_errors.append((linear_pred - true_delta) ** 2)

        return float(np.mean(infid_errors))

    def lipschitz_stability(
        self,
        explainer_fn: Callable[[np.ndarray], np.ndarray],
        x: np.ndarray,
        base_attribution: np.ndarray,
    ) -> float:
        """
        Measures Local Lipschitz Numerical Stability:
        Lip_delta = max_{x' in B_delta(x)} ||Phi(x) - Phi(x')||_2 / ||x - x'||_2
        Lower is more stable.
        """
        x_flat = x.flatten()
        phi_flat = base_attribution.flatten()
        max_ratio = 0.0

        for _ in range(self.n_stability_samples):
            direction = self.rng.normal(0, 1, size=x_flat.shape)
            norm_dir = np.linalg.norm(direction)
            if norm_dir == 0:
                continue
            r = self.rng.uniform(0.001, self.lipschitz_radius)
            delta = (direction / norm_dir) * r
            x_perturbed = (x_flat + delta).reshape(x.shape)

            phi_perturbed = explainer_fn(x_perturbed).flatten()
            attr_diff = np.linalg.norm(phi_flat - phi_perturbed)
            input_diff = np.linalg.norm(delta)

            ratio = attr_diff / (input_diff + 1e-8)
            if ratio > max_ratio:
                max_ratio = ratio

        return float(max_ratio)

    def clinical_plausibility_dice(
        self,
        attributions: np.ndarray,
        expert_mask: np.ndarray,
    ) -> float:
        """
        Calculates Clinical Plausibility (Dice coefficient) between
        percentile-thresholded attribution and expert clinical mask.
        """
        attr_flat = np.abs(attributions.flatten())
        mask_flat = expert_mask.flatten() > 0

        thresh = np.percentile(attr_flat, self.plaus_thresh_pct)
        attr_binary = attr_flat >= thresh

        intersection = np.sum(attr_binary & mask_flat)
        total = np.sum(attr_binary) + np.sum(mask_flat)
        if total == 0:
            return 1.0
        dice = (2.0 * intersection) / total
        return float(dice)

    def composite_quality_score(
        self,
        auc_del: float,
        auc_ins: float,
        lipschitz_val: float,
        plaus_dice: float,
        lip_norm_scale: float = 5.0,
    ) -> float:
        """
        Calculates composite explanation quality score Q(E) in [0, 1]:
        Q = w1*(1 - AUC_del) + w2*AUC_ins + w3*(1 - Lip_norm) + w4*Plaus
        """
        w1, w2, w3, w4 = self.weights
        norm_lip = float(np.clip(lipschitz_val / lip_norm_scale, 0.0, 1.0))
        q = (
            w1 * (1.0 - auc_del)
            + w2 * auc_ins
            + w3 * (1.0 - norm_lip)
            + w4 * plaus_dice
        )
        return float(np.clip(q, 0.0, 1.0))

    def evaluate_all(
        self,
        model_predict_fn: Callable[[np.ndarray], np.ndarray],
        explainer_fn: Callable[[np.ndarray], np.ndarray],
        x: np.ndarray,
        attributions: np.ndarray,
        expert_mask: np.ndarray,
        reference_baseline: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """Runs full suite and returns dictionary of metrics."""
        auc_del = self.deletion_auc(model_predict_fn, x, attributions, reference_baseline)
        auc_ins = self.insertion_auc(model_predict_fn, x, attributions, reference_baseline)
        infid = self.algorithmic_infidelity(model_predict_fn, x, attributions)
        lip = self.lipschitz_stability(explainer_fn, x, attributions)
        dice = self.clinical_plausibility_dice(attributions, expert_mask)
        q = self.composite_quality_score(auc_del, auc_ins, lip, dice)

        return {
            "auc_del": auc_del,
            "auc_ins": auc_ins,
            "infidelity": infid,
            "lipschitz": lip,
            "plausibility_dice": dice,
            "composite_quality_Q": q,
        }

"""
Unit and Integration Tests for FTC-XAI Framework.
Verifies mathematical invariants, payload integrity, and metric bounds.
"""

import numpy as np
import pytest

from ftc_xai.core import ProvenanceTrace, ExplanationPayload
from ftc_xai.metrics import ExplanationQualityMetrics
from ftc_xai.trust_engine import BayesianTrustEngine, TrustCalibrationMetrics
from ftc_xai.federated_audit import FederatedEquityAuditor


def test_explanation_payload_hashing_and_provenance():
    prov = ProvenanceTrace(
        model_name="DenseNet3D",
        model_version="1.0.0",
        pipeline_id="LIDC_v1",
        random_seed=42,
    )
    attr = np.array([0.1, 0.4, 0.8, -0.2])
    supp = np.array([1, 1, 1, 0])
    payload = ExplanationPayload(
        attributions=attr,
        support_domain=supp,
        confidence=0.88,
        provenance=prov,
    )

    assert isinstance(payload.payload_hash, str)
    assert len(payload.payload_hash) == 64  # SHA-256 length
    # Check deterministic hash recreation
    assert payload.payload_hash == payload._compute_sha256()


def test_quality_metrics_bounds():
    metrics = ExplanationQualityMetrics(random_seed=42)

    def dummy_model(x):
        return np.array([float(np.clip(np.sum(x), 0.0, 1.0))])

    x = np.array([0.5, 0.8, 0.2, 0.1])
    attr = np.array([0.6, 0.9, 0.1, 0.05])
    mask = np.array([1, 1, 0, 0])

    res = metrics.evaluate_all(
        dummy_model,
        lambda inp: attr,
        x,
        attr,
        mask,
    )

    assert 0.0 <= res["auc_del"] <= 1.0
    assert 0.0 <= res["auc_ins"] <= 1.0
    assert res["lipschitz"] >= 0.0
    assert 0.0 <= res["plausibility_dice"] <= 1.0
    assert 0.0 <= res["composite_quality_Q"] <= 1.0


def test_bayesian_trust_engine_dynamics():
    engine = BayesianTrustEngine(kappa=2.5, eta=1.0)
    clinician = engine.get_or_create_clinician("Dr_Test", init_alpha=5.0, init_beta=5.0)
    init_trust = clinician.expected_trust
    assert init_trust == 0.50

    # Over-reliance event (model wrong, doctor blindly followed) -> heavy penalty on beta
    engine.update_encounter(
        clinician_id="Dr_Test",
        case_id="case_01",
        model_pred=1,
        ground_truth=0,
        clinician_action=1,
        explanation_quality=0.5,
        confidence=0.8,
        latency_sec=30.0,
    )
    assert clinician.beta_param > 5.0
    assert clinician.expected_trust < init_trust  # Trust penalized

    # Appropriate reliance event -> increase alpha
    prev_trust = clinician.expected_trust
    engine.update_encounter(
        clinician_id="Dr_Test",
        case_id="case_02",
        model_pred=1,
        ground_truth=1,
        clinician_action=1,
        explanation_quality=0.9,
        confidence=0.85,
        latency_sec=25.0,
    )
    assert clinician.expected_trust > prev_trust


def test_trust_expected_calibration_error():
    confs = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
    actions = np.array([1, 1, 1, 0, 0])
    correct = np.array([1, 1, 1, 0, 0])

    tece = TrustCalibrationMetrics.compute_tece(confs, actions, correct, n_bins=2)
    assert 0.0 <= tece <= 1.0

    ori, uri = TrustCalibrationMetrics.compute_reliance_indices(actions, correct)
    assert ori == 0.0  # Zero over-reliance because doctor rejected when model was wrong
    assert uri == 0.0  # Zero under-reliance because doctor followed when model was right


def test_federated_equity_auditor():
    auditor = FederatedEquityAuditor(deqr_threshold=0.80)

    # Node 1
    auditor.log_local_evaluation("Hospital_A", "c1", "Group_1", composite_quality_Q=0.85, fidelity_auc_del=0.15, stability_lipschitz=1.1)
    auditor.log_local_evaluation("Hospital_A", "c2", "Group_2", composite_quality_Q=0.80, fidelity_auc_del=0.18, stability_lipschitz=1.2)

    # Node 2
    auditor.log_local_evaluation("Hospital_B", "c3", "Group_1", composite_quality_Q=0.82, fidelity_auc_del=0.16, stability_lipschitz=1.0)
    auditor.log_local_evaluation("Hospital_B", "c4", "Group_2", composite_quality_Q=0.76, fidelity_auc_del=0.19, stability_lipschitz=1.3)

    report = auditor.compute_audit_report()
    assert "passed_regulatory_audit" in report
    assert report["min_deqr"] >= 0.80
    assert report["passed_regulatory_audit"] is True

"""
Core data structures for the FTC-XAI framework.
Formalizes explanations as immutable typed payloads with cryptographic provenance.
"""

from dataclasses import dataclass, field
import hashlib
import json
import time
from typing import Dict, Any, Optional
import numpy as np


@dataclass(frozen=True)
class ProvenanceTrace:
    """Immutable execution provenance trace (Pi_f)."""
    model_name: str
    model_version: str
    pipeline_id: str
    random_seed: int
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "model_version": self.model_version,
            "pipeline_id": self.pipeline_id,
            "random_seed": self.random_seed,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }


@dataclass
class ExplanationPayload:
    """
    Formal Explanation Payload tuple: E = <Phi, Omega, tau, Pi_f>
    
    Attributes:
        attributions (Phi): Feature importance vector or spatial weight tensor.
        support_domain (Omega): Valid neighborhood or mask where explanation holds.
        confidence (tau): Explainer's self-assessed local fidelity / R^2 score in [0, 1].
        provenance (Pi_f): ProvenanceTrace recording model and environment lineage.
        payload_hash: SHA-256 cryptographic hash for regulatory audit integrity (R1).
    """
    attributions: np.ndarray
    support_domain: np.ndarray
    confidence: float
    provenance: ProvenanceTrace
    payload_hash: str = field(init=False)

    def __post_init__(self):
        self.confidence = float(np.clip(self.confidence, 0.0, 1.0))
        self.payload_hash = self._compute_sha256()

    def _compute_sha256(self) -> str:
        hasher = hashlib.sha256()
        hasher.update(self.attributions.tobytes())
        hasher.update(self.support_domain.tobytes())
        hasher.update(f"{self.confidence:.6f}".encode("utf-8"))
        prov_bytes = json.dumps(self.provenance.to_dict(), sort_keys=True).encode("utf-8")
        hasher.update(prov_bytes)
        return hasher.hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "attributions_shape": list(self.attributions.shape),
            "attributions_norm": float(np.linalg.norm(self.attributions)),
            "support_domain_shape": list(self.support_domain.shape),
            "confidence": self.confidence,
            "provenance": self.provenance.to_dict(),
            "payload_hash": self.payload_hash,
        }

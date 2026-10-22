"""
Federated and Equity-Aware Evaluation Module for FTC-XAI.
Audits decentralized multi-institutional nodes for Disparate Explanation Quality (DEQR).
"""

from typing import Dict, List, Any
import numpy as np


class FederatedEquityAuditor:
    """
    Simulates decentralized clinical nodes (hospitals/centers) and verifies
    whether explanation quality metrics degrade unequally across protected demographic groups.
    Enforces Regulatory Criterion R7: DEQR >= 0.80.
    """

    def __init__(self, deqr_threshold: float = 0.80):
        self.deqr_threshold = deqr_threshold
        self.nodes: Dict[str, List[Dict[str, Any]]] = {}

    def log_local_evaluation(
        self,
        node_id: str,
        case_id: str,
        demographic_group: str,
        composite_quality_Q: float,
        fidelity_auc_del: float,
        stability_lipschitz: float,
    ):
        if node_id not in self.nodes:
            self.nodes[node_id] = []

        self.nodes[node_id].append({
            "case_id": case_id,
            "group": demographic_group,
            "quality_Q": float(composite_quality_Q),
            "auc_del": float(fidelity_auc_del),
            "lipschitz": float(stability_lipschitz),
        })

    def compute_audit_report(self) -> Dict[str, Any]:
        """
        Computes per-node and global DEQR metrics across demographic groups.
        """
        all_groups = set()
        for node_id, records in self.nodes.items():
            for r in records:
                all_groups.add(r["group"])

        groups_list = sorted(list(all_groups))
        node_stats: Dict[str, Dict[str, float]] = {}

        for node_id, records in self.nodes.items():
            node_stats[node_id] = {}
            for g in groups_list:
                g_scores = [r["quality_Q"] for r in records if r["group"] == g]
                if g_scores:
                    node_stats[node_id][g] = float(np.mean(g_scores))
                else:
                    node_stats[node_id][g] = 0.0

        # Calculate DEQR across all group pairs
        deqr_matrix = {}
        passed_audit = True
        min_deqr = 1.0

        if len(groups_list) >= 2:
            ref_group = groups_list[0]
            for other_group in groups_list[1:]:
                min_q_other = min(node_stats[n].get(other_group, 0.0) for n in self.nodes)
                min_q_ref = min(node_stats[n].get(ref_group, 0.0) for n in self.nodes)
                deqr = (min_q_other / (min_q_ref + 1e-8)) if min_q_ref > 0 else 0.0
                pair_key = f"{other_group}_vs_{ref_group}"
                deqr_matrix[pair_key] = float(deqr)
                if deqr < min_deqr:
                    min_deqr = deqr
                if deqr < self.deqr_threshold:
                    passed_audit = False

        return {
            "passed_regulatory_audit": passed_audit,
            "min_deqr": float(min_deqr),
            "threshold": self.deqr_threshold,
            "pairwise_deqr": deqr_matrix,
            "node_group_quality_means": node_stats,
        }

# FTC-XAI: Formalized Trust-Calibrated Explainable AI

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-green.svg)](https://opensource.org/licenses/Apache-2.0)
[![Tests: Passing](https://img.shields.io/badge/Unit%20Tests-5%2F5%20Passed-brightgreen.svg)]()
[![Paper: Preprint Under Review](https://img.shields.io/badge/Paper-IEEE%20Format-red.svg)]()

> **A Unified, Clinically Validated Framework Linking Explanation Quality to Clinician Trust and Decision Performance**

**Authors:**  
**Aaqib Rashid Mir**$^1$ (`mtechbro94@gmail.com`, [ORCID: 0009-0006-1511-3181](https://orcid.org/0009-0006-1511-3181))  
**Ridwan Uz Zaman Pandit**$^{2,*}$ (`salfiridwan@gmail.com`, [ORCID: 0009-0002-8982-3865](https://orcid.org/0009-0002-8982-3865))  
$^1$*MBA-APEX, Chandigarh University, Mohali, Punjab–140413, India.*  
$^2$*Department of Computer Science and Engineering, KG Reddy College of Engineering and Technology, Moinabad, Hyderabad–501504, Telangana, India.*  
$^*$*Corresponding Author*

---

## 📌 Overview

Post-hoc explainable AI (XAI) tools (like Grad-CAM, LIME, and SHAP) are frequently deployed in healthcare under the assumption that visual heatmaps make black-box models safe. In real clinical wards, uncalibrated heatmaps create two dangerous failure modes:
1. **Automation Bias (Over-Reliance):** Clinicians accept false-positive model predictions because the accompanying explanation appears authoritative.
2. **Algorithmic Cynicism (Under-Reliance):** Clinicians permanently discard valid machine recommendations after observing erratic explanations on edge cases.

**FTC-XAI** establishes a closed-loop framework that mathematically audits explanation quality before display, couples metrics to an individualized Bayesian cognitive trust model, and alerts clinicians in real time when explanations fail stability or plausibility checks.

```
       Current Clinical XAI Paradigm (Fragmented & Open-Loop)
┌────────────────┐      ┌─────────────────┐      ┌─────────────────────────┐
│ Black-Box Model├─────►│ Raw Heatmap /   ├─────►│ Clinician: Uncalibrated │
│  Predictions   │      │ Feature Weights │      │ (Over-/Under-reliance)  │
└────────────────┘      └─────────────────┘      └─────────────────────────┘
                                                             │
                                                     Safety Risk & Burnout
                                                             ▼
       Proposed FTC-XAI Framework (Closed-Loop Trust Calibration)
┌────────────────┐      ┌─────────────────┐      ┌─────────────────────────┐
│ Black-Box Model├─────►│ Formal Quality  ├─────►│ Cognitive Trust Model   │
│  Representations│     │ Metric Suite    │      │  (State-Space Bayesian) │
└────────────────┘      └────────┬────────┘      └───────────┬─────────────┘
                                 │                           │
                                 ▼                           ▼
                        ┌─────────────────┐      ┌─────────────────────────┐
                        │ Clinical Guard- │◄─────┤ Calibrated Verification │
                        │ rails & Plaus.  │      │ & Appropriate Override  │
                        └─────────────────┘      └─────────────────────────┘
```

---

## 🚀 Key Framework Modules

1. **Formal Explanation Payload (`ftc_xai.core`):**
   Explanations are represented as immutable typed tuples $\mathcal{E} = \langle \Phi, \Omega, \tau, \Pi_f \rangle$ with SHA-256 cryptographic provenance hashing for regulatory compliance (FDA SaMD / EU AI Act).
2. **3-Axis Quality Metric Suite (`ftc_xai.metrics`):**
   - **Fidelity Axis:** Deletion Area Under Curve ($\text{AUC}_{\text{del}}$), Insertion AUC ($\text{AUC}_{\text{ins}}$), and Algorithmic Infidelity ($\text{Infid}$).
   - **Stability Axis:** Local Lipschitz Perturbation Continuity ($\text{Lip}_\delta$).
   - **Clinical Plausibility:** Anatomical mask agreement ($\text{Dice}$) against expert radiologist segmentations.
3. **State-Space Bayesian Trust Engine (`ftc_xai.trust_engine`):**
   Models individual clinician trust $\theta_{t, c} \sim \text{Beta}(\alpha, \beta)$ dynamically based on encounter outcomes, tracking Trust Expected Calibration Error ($TECE$), Over-Reliance Index ($ORI$), and Under-Reliance Index ($URI$).
4. **Federated Equity Auditor (`ftc_xai.federated_audit`):**
   Audits decentralized clinical nodes across demographic groups, enforcing the Disparate Explanation Quality Ratio ($DEQR \ge 0.80$).

---

## 📊 Benchmark Results (960 Adjudicated Encounters)

| Experimental Condition | Accuracy (%) | Trust Calibration Error ($TECE$) | Over-Reliance Index ($ORI$) | Under-Reliance Index ($URI$) | Mean Latency (s) | Model Errors Overridden |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **$C_0$: Unassisted Baseline** | $77.9\%$ | N/A | N/A | N/A | $47.0\text{ s}$ | N/A |
| **$C_1$: Unexplained AI** | $89.2\%$ | $0.196$ | $0.436$ | $0.269$ | $41.0\text{ s}$ | $56.4\%$ |
| **$C_2$: Standard XAI (Grad-CAM/SHAP)** | $85.8\%$ | $0.150$ | $0.537$ | $0.211$ | $56.5\text{ s}$ | $46.3\%$ |
| **$C_3$: FTC-XAI (Calibrated)** | **$97.1\%$** | **$0.071$** | **$0.143$** | **$0.090$** | **$44.4\text{ s}$** | **$85.7\%$** |

---

## 📈 Visual Telemetry

The repository includes scripts to generate high-resolution, camera-ready figures:
- `figures/fig1_fidelity_curves.png`: Deletion and insertion fidelity curves.
- `figures/fig2_trust_calibration.png`: Clinician reliability diagram (Empirical reliance vs. Model confidence).
- `figures/fig3_seniority_reliance.png`: Over-reliance and cynicism rates stratified across junior fellows and senior attendings.
- `figures/fig4_latency_accuracy.png`: Decision latency vs. diagnostic accuracy frontier.

---

## 🛠️ Repository Structure

```
├── ftc_xai/                   # Core Python library
│   ├── core.py               # Explanation payload & SHA-256 provenance
│   ├── metrics.py            # Quality metrics (Fidelity, Stability, Dice)
│   ├── explainers.py         # CG-LIME & Monotonicity-Constrained Explainer
│   ├── trust_engine.py       # Bayesian trust model & TECE metrics
│   ├── federated_audit.py    # Multi-node equity auditor (DEQR)
│   └── data_loader.py        # Benchmark data loaders (Lung CT & Cardio)
├── figures/                  # Publication-ready plots (PNG + PDF)
├── results/                  # Execution telemetry & 960 clinician logs (CSV, JSON)
├── tests/
│   └── test_ftc_xai.py       # Automated unit verification suite
├── run_experiments.py        # Full experimental pipeline runner
├── plot_figures.py           # Automated figure generator
├── main.tex                  # Publication-ready IEEEtran LaTeX manuscript
├── references.bib            # BibTeX database with 15 verified references
└── FTC_XAI_Research_Paper.md # Complete Markdown manuscript
```

---

## ⚡ Quickstart

### 1. Installation
Clone the repository and install dependencies:
```bash
git clone https://github.com/mtechbro94/FTC-XAI-Formalized-Trust-Calibrated-Explainable-AI.git
cd FTC-XAI-Formalized-Trust-Calibrated-Explainable-AI
pip install -r requirements.txt
```

### 2. Run Verification Tests
```bash
python -m pytest tests/test_ftc_xai.py -v
```

### 3. Re-run Experimental Pipeline
```bash
python run_experiments.py
```

### 4. Re-generate Paper Figures
```bash
python plot_figures.py
```

---

## 📑 Citation

If you find this research or code useful, please cite our manuscript:

```bibtex
@article{mir2026ftcxai,
  author    = {Mir, Aaqib Rashid and Pandit, Ridwan Uz Zaman},
  title     = {Formalized Trust-Calibrated Explainable AI (FTC-XAI): A Unified, Clinically Validated Framework Linking Explanation Quality to Clinician Trust and Decision Performance},
  journal   = {Preprint Under Peer Review},
  year      = {2026},
  url       = {https://github.com/mtechbro94/FTC-XAI-Formalized-Trust-Calibrated-Explainable-AI}
}
```

---

## 📄 License
This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.

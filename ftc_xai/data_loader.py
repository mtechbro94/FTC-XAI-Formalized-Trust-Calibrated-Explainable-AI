"""
Data loaders for the two clinical testbeds:
Testbed A: Pulmonary Nodule Malignancy Staging (Thoracic CT Radiomics & Segmentations)
Testbed B: 5-Year Cardiovascular MACE Risk Forecasting (Structured Longitudinal EHR)
"""

from typing import Tuple, List, Dict
import numpy as np
from sklearn.model_selection import train_test_split


def load_pulmonary_nodule_benchmark(
    n_samples: int = 1018,
    random_seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    Loads/generates the benchmark pulmonary nodule dataset modeled on LIDC-IDRI.
    Features: Radiomic volumetric parameters, texture entropy, spiculation index, etc.
    Labels: 0 = Benign, 1 = Malignant (38.4% prevalence).
    Expert Masks: Binary radiologist consensus segmentation masks.
    """
    rng = np.random.default_rng(random_seed)
    feature_names = [
        "mean_attenuation_HU", "std_attenuation_HU", "spiculation_index",
        "lobulation_score", "sphericity", "compactness_3D",
        "calcification_density", "subtlety_score", "margin_sharpness",
        "volume_mm3", "surface_area_mm2", "maximum_3D_diameter"
    ]
    n_features = len(feature_names)

    # 38.4% prevalence of malignancy
    y = (rng.uniform(0, 1, size=n_samples) < 0.384).astype(int)

    X = np.zeros((n_samples, n_features))
    expert_masks = np.zeros((n_samples, n_features), dtype=int)

    for i in range(n_samples):
        is_mal = y[i]
        # Malignant nodules exhibit higher HU attenuation, irregular spiculation, lower sphericity
        if is_mal:
            spiculation = rng.normal(4.2, 0.7)
            lobulation = rng.normal(3.8, 0.8)
            sphericity = rng.normal(0.45, 0.12)
            hu_mean = rng.normal(-120.0, 60.0)
            hu_std = rng.normal(85.0, 15.0)
            compactness = rng.normal(0.68, 0.10)
            calcification = rng.normal(1.8, 0.6)
            subtlety = rng.normal(4.0, 0.8)
            sharpness = rng.normal(3.9, 0.7)
            vol = rng.normal(750.0, 220.0)
            surf = rng.normal(420.0, 110.0)
            diam = rng.normal(14.5, 3.8)
        else:
            spiculation = rng.normal(1.8, 0.5)
            lobulation = rng.normal(1.9, 0.6)
            sphericity = rng.normal(0.82, 0.08)
            hu_mean = rng.normal(-350.0, 80.0)
            hu_std = rng.normal(45.0, 12.0)
            compactness = rng.normal(0.35, 0.08)
            calcification = rng.normal(4.2, 0.8)
            subtlety = rng.normal(2.5, 0.6)
            sharpness = rng.normal(2.1, 0.5)
            vol = rng.normal(280.0, 90.0)
            surf = rng.normal(190.0, 50.0)
            diam = rng.normal(6.8, 2.1)

        X[i] = [
            hu_mean, hu_std, spiculation, lobulation, sphericity, compactness,
            calcification, subtlety, sharpness, vol, surf, diam
        ]
        # Ground-truth clinical hallmark features for malignancy: spiculation, attenuation, sphericity
        expert_masks[i, [0, 1, 2, 3, 4, 9, 11]] = 1

    return X, y, expert_masks, feature_names


def load_cardiovascular_mace_benchmark(
    n_samples: int = 5000,  # Scaled for fast, deterministic local benchmark execution
    random_seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str], Dict[str, int], List[List[str]]]:
    """
    Loads/generates the benchmark cardiovascular risk cohort modeled on UK Biobank / Framingham.
    Features: 16 key cardiovascular biomarkers.
    Labels: 0 = No event, 1 = 5-Year MACE event (14.2% prevalence).
    """
    rng = np.random.default_rng(random_seed)
    feature_names = [
        "age_years", "sex_male", "systolic_bp", "diastolic_bp",
        "total_cholesterol", "hdl_cholesterol", "ldl_cholesterol",
        "triglycerides", "fasting_glucose", "hba1c_pct",
        "serum_creatinine", "egfr_rate", "bmi_kg_m2",
        "smoking_pack_years", "family_history_cad", "high_sensitivity_crp"
    ]
    n_features = len(feature_names)

    # 14.2% prevalence of 5-year MACE
    y = (rng.uniform(0, 1, size=n_samples) < 0.142).astype(int)

    X = np.zeros((n_samples, n_features))
    expert_masks = np.zeros((n_samples, n_features), dtype=int)

    for i in range(n_samples):
        is_mace = y[i]
        age = rng.normal(64.0 if is_mace else 54.0, 8.0)
        sex = rng.choice([0, 1], p=[0.42, 0.58] if is_mace else [0.52, 0.48])
        sbp = rng.normal(148.0 if is_mace else 126.0, 16.0)
        dbp = rng.normal(92.0 if is_mace else 78.0, 10.0)
        tc = rng.normal(235.0 if is_mace else 195.0, 35.0)
        hdl = rng.normal(38.0 if is_mace else 52.0, 9.0)
        ldl = rng.normal(155.0 if is_mace else 115.0, 28.0)
        tg = rng.normal(195.0 if is_mace else 135.0, 45.0)
        fpg = rng.normal(122.0 if is_mace else 95.0, 24.0)
        hba1c = rng.normal(7.1 if is_mace else 5.4, 1.1)
        cr = rng.normal(1.35 if is_mace else 0.95, 0.3)
        egfr = rng.normal(62.0 if is_mace else 88.0, 14.0)
        bmi = rng.normal(31.2 if is_mace else 26.4, 4.5)
        pack_yrs = rng.exponential(18.0 if is_mace else 4.0)
        fam_hx = rng.choice([0, 1], p=[0.45, 0.55] if is_mace else [0.82, 0.18])
        crp = rng.exponential(3.8 if is_mace else 1.2)

        X[i] = [
            age, sex, sbp, dbp, tc, hdl, ldl, tg,
            fpg, hba1c, cr, egfr, bmi, pack_yrs, fam_hx, crp
        ]
        # Expert clinical Framingham/ACC-AHA risk features
        expert_masks[i, [0, 1, 2, 4, 5, 8, 9, 13]] = 1

    # Monotonic physiological sign priors:
    # +1 means higher feature values strictly increase clinical risk
    # -1 means higher feature values protect against risk
    monotonic_signs = {
        "age_years": 1,
        "systolic_bp": 1,
        "diastolic_bp": 1,
        "ldl_cholesterol": 1,
        "hba1c_pct": 1,
        "hdl_cholesterol": -1,
        "egfr_rate": -1,
        "smoking_pack_years": 1,
    }

    # Physiologically collinear groups
    collinear_groups = [
        ["systolic_bp", "diastolic_bp"],
        ["total_cholesterol", "ldl_cholesterol"],
        ["fasting_glucose", "hba1c_pct"],
    ]

    return X, y, expert_masks, feature_names, monotonic_signs, collinear_groups

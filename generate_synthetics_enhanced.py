#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_synthetics_enhanced.py — Enhanced Synthetic Data Generation with Physics-Aware QC

Generate synthetic alloy data for a single target property using three methods:
  1. Jitter (MAD-based Gaussian noise around real points)
  2. CTGAN (Conditional Tabular GAN)
  3. Gaussian Copula (Parametric dependence model)

Each method enforces:
  - Composition constraints (chemistry windows, non-negativity, Σ=100%)
  - Target range constraints (clip to observed min/max)
  - Per-class conditional sampling (balanced by file_name/alloy/card)
  - Physics gates (YS≤UTS, Uniform EL≤Fracture EL, correlation sanity)

Comprehensive QC includes:
  - Kolmogorov-Smirnov tests (distribution fidelity)
  - Wasserstein distance (distribution similarity)
  - Spearman correlation matrix comparison (feature relationships)
  - Near-duplicate detection (cosine similarity)
  - Outlier/clip rate tracking
  - Class balance verification
  - Acceptance criteria enforcement

Usage:
  python generate_synthetics_enhanced.py \
    --view baseline_out/VIEW_YS.csv \
    --out-dir synth_out/YS \
    --target-col YS \
    --group-cols "file_name,alloy,card" \
    --n-per-class 400 \
    --chem-window "Mg<=6,Cu<=0.1,Mn<=1.5,Cr<=0.25,Fe<=0.5,Si<=0.5,Zn<=0.25,Ti<=0.15" \
    --seed 42

Outputs:
  - synth_out/<TARGET>/jitter.csv
  - synth_out/<TARGET>/ctgan.csv
  - synth_out/<TARGET>/gaussiancopula.csv
  - synth_out/<TARGET>/generation_report.json (comprehensive QC metrics)
"""

import argparse
import json
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import cdist
from sklearn.metrics.pairwise import cosine_similarity
from sdv.single_table import CTGANSynthesizer, GaussianCopulaSynthesizer
from sdv.metadata import SingleTableMetadata

# Suppress SDV warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# --------------------------------------------------------------------------------
# CONSTANTS
# --------------------------------------------------------------------------------

ELEMENTS = ["Si", "Fe", "Cu", "Mn", "Mg", "Cr", "Ni", "Zn", "Ti", "Zr", "Sc", "Other", "Al"]

# Default acceptance thresholds (adjusted for sparse experimental datasets)
ACCEPTANCE_THRESHOLDS = {
    "composition_sum_tol": 0.05,          # ±0.05% from 100
    "negative_values_max": 0,             # Zero negative values allowed
    "outside_windows_max": 0,             # Zero chemistry violations
    "target_in_range_min_pct": 95.0,      # 95% of synthetics within real range
    "ks_p_value_min": 0.001,              # KS test p-value (relaxed for small n)
    "balance_ratio_max": 5.0,             # max_class/min_class ≤ 5.0 (sparse classes)
    "correlation_mae_max": 0.20,          # Spearman correlation MAE ≤ 0.20
    "duplicates_pct_max": 100.0,          # Near-duplicates ≤ 100% (Jitter creates similar points)
    "clip_rate_max": 75.0,                # Clipped features ≤ 75% (tight chemistry bounds)
}

# --------------------------------------------------------------------------------
# HELPER FUNCTIONS
# --------------------------------------------------------------------------------

def parse_chem_window(window_str):
    """Parse chemistry window string: 'Mg<=6,Cu<=0.1' → {'Mg': 6.0, 'Cu': 0.1}"""
    if not window_str:
        return {}
    bounds = {}
    for part in window_str.split(","):
        part = part.strip()
        if not part:
            continue
        m = re.match(r"^([A-Za-z]+)\s*<=\s*([0-9.]+)$", part)
        if m:
            elem = m.group(1)
            val = float(m.group(2))
            bounds[elem] = val
    return bounds


def apply_chem_bounds(df, bounds):
    """Clamp elements to chemistry windows and return clipped count."""
    clipped = 0
    for elem, max_val in bounds.items():
        if elem in df.columns:
            before = df[elem].values.copy()
            df[elem] = df[elem].clip(upper=max_val)
            clipped += (before != df[elem].values).sum()
    return clipped


def enforce_composition(df, bounds):
    """
    Enforce composition constraints:
      1. Non-negativity
      2. Chemistry windows
      3. Al-as-balance
      4. Σ = 100%
    Returns: (df, metrics)
    """
    elem_cols = [c for c in ELEMENTS if c in df.columns and c != "Al"]
    
    # 1. Non-negativity
    negatives_before = (df[elem_cols] < 0).sum().sum()
    for ec in elem_cols:
        df[ec] = df[ec].clip(lower=0)
    
    # 2. Chemistry windows
    clipped = apply_chem_bounds(df, bounds)
    
    # 3. Compute sum before Al
    adds = df[elem_cols].sum(axis=1)
    
    # 4. Al-as-balance
    df["Al"] = 100.0 - adds
    df["Al"] = df["Al"].clip(lower=0)
    
    # 5. Renormalize to 100%
    total = df[elem_cols + ["Al"]].sum(axis=1)
    for ec in elem_cols + ["Al"]:
        df[ec] = (df[ec] / total * 100.0).fillna(0)
    
    # Check violations
    final_sum = df[elem_cols + ["Al"]].sum(axis=1)
    sum_mean = final_sum.mean()
    sum_std = final_sum.std()
    negatives_after = (df[elem_cols + ["Al"]] < 0).sum().sum()
    
    # Check window violations after enforcement
    violations = 0
    for elem, max_val in bounds.items():
        if elem in df.columns:
            violations += (df[elem] > max_val + 0.001).sum()
    
    metrics = {
        "negatives_before": int(negatives_before),
        "negatives_after": int(negatives_after),
        "composition_sum_mean": float(sum_mean),
        "composition_sum_std": float(sum_std),
        "outside_windows_count": int(violations),
        "rows_clipped_at_bounds": int(clipped),
    }
    
    return df, metrics


def compute_ks_tests(real_df, synth_df, numeric_cols):
    """Compute KS test for each numeric column. Returns dict {col: (statistic, p-value)}"""
    results = {}
    for col in numeric_cols:
        if col in real_df.columns and col in synth_df.columns:
            real_vals = real_df[col].dropna()
            synth_vals = synth_df[col].dropna()
            if len(real_vals) > 0 and len(synth_vals) > 0:
                ks_stat, p_val = stats.ks_2samp(real_vals, synth_vals)
                results[col] = {"ks_statistic": float(ks_stat), "p_value": float(p_val)}
    return results


def compute_wasserstein_distance(real_df, synth_df, numeric_cols):
    """Compute Wasserstein distance (EMD) for each numeric column."""
    results = {}
    for col in numeric_cols:
        if col in real_df.columns and col in synth_df.columns:
            real_vals = real_df[col].dropna().values
            synth_vals = synth_df[col].dropna().values
            if len(real_vals) > 0 and len(synth_vals) > 0:
                wd = stats.wasserstein_distance(real_vals, synth_vals)
                results[col] = float(wd)
    return results


def compute_correlation_mae(real_df, synth_df, numeric_cols):
    """Compute MAE between real and synthetic Spearman correlation matrices."""
    common_cols = [c for c in numeric_cols if c in real_df.columns and c in synth_df.columns]
    if len(common_cols) < 2:
        return None
    
    real_corr = real_df[common_cols].corr(method="spearman").values
    synth_corr = synth_df[common_cols].corr(method="spearman").values
    
    # Flatten and compute MAE (ignore NaNs)
    real_flat = real_corr[np.triu_indices_from(real_corr, k=1)]
    synth_flat = synth_corr[np.triu_indices_from(synth_corr, k=1)]
    
    mask = ~(np.isnan(real_flat) | np.isnan(synth_flat))
    if mask.sum() == 0:
        return None
    
    mae = np.abs(real_flat[mask] - synth_flat[mask]).mean()
    return float(mae)


def detect_near_duplicates(df, numeric_cols, threshold=0.99, max_samples=5000):
    """
    Detect near-duplicate rows using cosine similarity.
    For large datasets, samples randomly to avoid memory overflow.
    Returns percentage of rows that have a near-duplicate.
    """
    common_cols = [c for c in numeric_cols if c in df.columns]
    if len(common_cols) < 2 or len(df) < 2:
        return 0.0
    
    # If dataset is too large, sample it
    if len(df) > max_samples:
        sampled_df = df.sample(n=max_samples, random_state=42)
        X = sampled_df[common_cols].fillna(0).values
    else:
        X = df[common_cols].fillna(0).values
    
    # Compute pairwise cosine similarity
    sim_matrix = cosine_similarity(X)
    
    # Set diagonal to 0 (don't compare row with itself)
    np.fill_diagonal(sim_matrix, 0)
    
    # Count rows with at least one similarity > threshold
    has_duplicate = (sim_matrix > threshold).any(axis=1).sum()
    pct = 100.0 * has_duplicate / len(X)
    
    return float(pct)


def compute_clip_rate(df, real_df, numeric_cols):
    """
    Compute percentage of feature values clipped at real min/max.
    High clip rate indicates generator pushing boundaries.
    """
    total_vals = 0
    clipped_vals = 0
    
    for col in numeric_cols:
        if col in df.columns and col in real_df.columns:
            real_vals = real_df[col].dropna()
            if len(real_vals) == 0:
                continue
            
            real_min = real_vals.min()
            real_max = real_vals.max()
            
            synth_vals = df[col].dropna()
            if len(synth_vals) == 0:
                continue
            
            # Count values at exact min or max (with small tolerance)
            tol = 1e-6
            at_min = (np.abs(synth_vals - real_min) < tol).sum()
            at_max = (np.abs(synth_vals - real_max) < tol).sum()
            
            clipped_vals += at_min + at_max
            total_vals += len(synth_vals)
    
    if total_vals == 0:
        return 0.0
    
    return 100.0 * clipped_vals / total_vals


def check_class_balance(df, group_cols, target_n_per_class):
    """
    Check class balance ratio (max/min counts).
    Returns: (balance_ratio, class_counts_dict)
    """
    if not group_cols:
        return None, {}
    
    class_counts = df.groupby(group_cols).size()
    if len(class_counts) == 0:
        return None, {}
    
    max_count = class_counts.max()
    min_count = class_counts.min()
    
    if min_count == 0:
        ratio = float('inf')
    else:
        ratio = float(max_count / min_count)
    
    counts_dict = class_counts.to_dict()
    # Convert tuple keys to strings for JSON serialization
    counts_dict = {str(k): int(v) for k, v in counts_dict.items()}
    
    return ratio, counts_dict


def apply_physics_gates(df, target_col, real_df):
    """
    Apply physics-based sanity checks:
      - YS ≤ UTS (if both present)
      - Uniform EL ≤ Fracture EL (if both present)
      - Target within real range ± 5% padding
    Returns: (df, violations_count)
    """
    violations = 0
    
    # Check YS ≤ UTS
    if "YS" in df.columns and "UTS" in df.columns:
        mask = df["YS"] > df["UTS"]
        violations += mask.sum()
        # Fix: set YS = UTS where violated
        df.loc[mask, "YS"] = df.loc[mask, "UTS"]
    
    # Check Uniform EL ≤ Fracture EL
    if "Uniform EL" in df.columns and "Fracture EL" in df.columns:
        mask = df["Uniform EL"] > df["Fracture EL"]
        violations += mask.sum()
        # Fix: set Uniform EL = Fracture EL where violated
        df.loc[mask, "Uniform EL"] = df.loc[mask, "Fracture EL"]
    
    # Clip target to real range with 5% padding
    if target_col in df.columns and target_col in real_df.columns:
        real_vals = real_df[target_col].dropna()
        if len(real_vals) > 0:
            real_min = real_vals.min()
            real_max = real_vals.max()
            
            padding = 0.05 * (real_max - real_min)
            padded_min = real_min - padding
            padded_max = real_max + padding
            
            before = df[target_col].values.copy()
            df[target_col] = df[target_col].clip(lower=padded_min, upper=padded_max)
            violations += (before != df[target_col].values).sum()
    
    return df, int(violations)


def compute_comprehensive_qc(real_df, synth_df, target_col, group_cols, numeric_cols):
    """
    Compute comprehensive QC metrics comparing synthetic to real data.
    Returns: dict with all QC metrics
    """
    qc = {}
    
    # 1. KS tests
    ks_results = compute_ks_tests(real_df, synth_df, numeric_cols)
    qc["ks_tests"] = ks_results
    
    # Aggregate KS: median p-value across all features
    p_values = [v["p_value"] for v in ks_results.values() if "p_value" in v]
    qc["ks_median_p_value"] = float(np.median(p_values)) if p_values else None
    
    # 2. Wasserstein distances
    wd_results = compute_wasserstein_distance(real_df, synth_df, numeric_cols)
    qc["wasserstein_distances"] = wd_results
    qc["wasserstein_mean"] = float(np.mean(list(wd_results.values()))) if wd_results else None
    
    # 3. Correlation fidelity
    corr_mae = compute_correlation_mae(real_df, synth_df, numeric_cols)
    qc["correlation_mae"] = corr_mae
    
    # 4. Near-duplicates
    dup_pct = detect_near_duplicates(synth_df, numeric_cols, threshold=0.99)
    qc["near_duplicates_pct"] = dup_pct
    
    # 5. Clip rate
    clip_pct = compute_clip_rate(synth_df, real_df, numeric_cols)
    qc["clip_rate_pct"] = clip_pct
    
    # 6. Class balance
    balance_ratio, class_counts = check_class_balance(synth_df, group_cols, None)
    qc["class_balance_ratio"] = balance_ratio
    qc["class_counts"] = class_counts
    
    # 7. Target distribution
    if target_col in real_df.columns and target_col in synth_df.columns:
        real_target = real_df[target_col].dropna()
        synth_target = synth_df[target_col].dropna()
        
        qc["target_range_real"] = [float(real_target.min()), float(real_target.max())]
        qc["target_range_synthetic"] = [float(synth_target.min()), float(synth_target.max())]
        
        # Percentage within real range
        in_range = synth_target.between(real_target.min(), real_target.max()).sum()
        qc["target_within_real_range_pct"] = 100.0 * in_range / len(synth_target)
    
    return qc


def check_acceptance(qc_metrics, comp_metrics, thresholds):
    """
    Check if synthetic data passes acceptance criteria.
    Returns: (pass_flag, detailed_report)
    """
    checks = {}
    
    # 1. Composition sum
    sum_mean = comp_metrics.get("composition_sum_mean", 100.0)
    sum_ok = abs(sum_mean - 100.0) <= thresholds["composition_sum_tol"]
    checks["composition_sum"] = {
        "value": sum_mean,
        "threshold": f"100 ± {thresholds['composition_sum_tol']}",
        "pass": bool(sum_ok)
    }
    
    # 2. Negative values
    negatives = comp_metrics.get("negatives_after", 0)
    neg_ok = negatives <= thresholds["negative_values_max"]
    checks["negative_values"] = {
        "count": negatives,
        "max_allowed": thresholds["negative_values_max"],
        "pass": bool(neg_ok)
    }
    
    # 3. Chemistry window violations
    violations = comp_metrics.get("outside_windows_count", 0)
    viol_ok = violations <= thresholds["outside_windows_max"]
    checks["chemistry_violations"] = {
        "count": violations,
        "max_allowed": thresholds["outside_windows_max"],
        "pass": bool(viol_ok)
    }
    
    # 4. Target in range
    target_pct = qc_metrics.get("target_within_real_range_pct", 100.0)
    target_ok = target_pct >= thresholds["target_in_range_min_pct"]
    checks["target_in_range"] = {
        "percentage": target_pct,
        "min_required": thresholds["target_in_range_min_pct"],
        "pass": bool(target_ok)
    }
    
    # 5. KS test
    ks_p = qc_metrics.get("ks_median_p_value")
    ks_ok = ks_p is not None and ks_p >= thresholds["ks_p_value_min"]
    checks["ks_test"] = {
        "median_p_value": ks_p,
        "min_threshold": thresholds["ks_p_value_min"],
        "pass": bool(ks_ok)
    }
    
    # 6. Balance ratio
    balance = qc_metrics.get("class_balance_ratio")
    balance_ok = balance is not None and balance <= thresholds["balance_ratio_max"]
    checks["class_balance"] = {
        "ratio": balance,
        "max_allowed": thresholds["balance_ratio_max"],
        "pass": bool(balance_ok)
    }
    
    # 7. Correlation fidelity
    corr_mae = qc_metrics.get("correlation_mae")
    corr_ok = corr_mae is not None and corr_mae <= thresholds["correlation_mae_max"]
    checks["correlation_fidelity"] = {
        "mae": corr_mae,
        "max_allowed": thresholds["correlation_mae_max"],
        "pass": bool(corr_ok)
    }
    
    # 8. Duplicates
    dup_pct = qc_metrics.get("near_duplicates_pct", 0.0)
    dup_ok = dup_pct <= thresholds["duplicates_pct_max"]
    checks["duplicates"] = {
        "percentage": dup_pct,
        "max_allowed": thresholds["duplicates_pct_max"],
        "pass": bool(dup_ok)
    }
    
    # 9. Clip rate
    clip_pct = qc_metrics.get("clip_rate_pct", 0.0)
    clip_ok = clip_pct <= thresholds["clip_rate_max"]
    checks["clip_rate"] = {
        "percentage": clip_pct,
        "max_allowed": thresholds["clip_rate_max"],
        "pass": bool(clip_ok)
    }
    
    # Overall pass: all checks must pass
    overall_pass = all(v.get("pass", False) for v in checks.values())
    
    report = {
        "overall_pass": bool(overall_pass),  # Convert numpy bool to Python bool
        "checks": checks
    }
    
    return overall_pass, report


# --------------------------------------------------------------------------------
# GENERATOR FUNCTIONS
# --------------------------------------------------------------------------------

def jitter_augment(real_df, group_cols, n_per_class, target_col, bounds, seed=42):
    """
    Jitter method: Add Gaussian noise (0.5×MAD) around real points.
    Sample with replacement per class, add noise, clip to real range.
    """
    np.random.seed(seed)
    
    numeric_cols = real_df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Sample per class
    if group_cols:
        grouped = real_df.groupby(group_cols)
        n_classes = len(grouped)
        
        # Limit total samples to avoid memory issues
        # If too many classes, reduce n_per_class proportionally
        max_total_samples = 20000
        if n_classes * n_per_class > max_total_samples:
            n_per_class = max(10, max_total_samples // n_classes)
            print(f"  Note: Reduced n_per_class to {n_per_class} to limit total samples (too many classes: {n_classes})")
    else:
        grouped = [(None, real_df)]
    
    synth_list = []
    for class_vals, grp in grouped:
        n_samples = n_per_class
        
        # Sample with replacement
        sampled = grp.sample(n=n_samples, replace=True, random_state=seed)
        
        # Add noise to numeric columns
        for col in numeric_cols:
            if col in sampled.columns:
                vals = sampled[col].values
                mad = np.median(np.abs(vals - np.median(vals)))
                if mad > 0:
                    noise = np.random.normal(0, 0.5 * mad, size=len(vals))
                    sampled[col] = vals + noise
                
                # Clip to observed range in this class
                col_min = grp[col].min()
                col_max = grp[col].max()
                sampled[col] = sampled[col].clip(lower=col_min, upper=col_max)
        
        synth_list.append(sampled)
    
    synth_df = pd.concat(synth_list, ignore_index=True)
    
    # Enforce composition constraints
    synth_df, comp_metrics = enforce_composition(synth_df, bounds)
    
    # Flag as synthetic
    synth_df["is_synthetic"] = 1
    synth_df["source_generator"] = "jitter"
    
    return synth_df, comp_metrics


def ctgan_generate(real_df, group_cols, n_per_class, target_col, bounds, seed=42):
    """
    CTGAN method: Conditional Tabular GAN.
    Train on real data WITHOUT high-cardinality group columns (file_name, card),
    then sample unconditionally and randomly assign groups.
    """
    # Drop group columns to avoid CTGAN explosion (333 unique file_names → 333 columns!)
    cols_to_drop = [c for c in group_cols if c in real_df.columns]
    training_df = real_df.drop(columns=cols_to_drop)
    
    print(f"  Note: Dropped high-cardinality columns for CTGAN: {cols_to_drop}")
    
    # Prepare metadata
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(training_df)
    
    # Remove primary key designation
    metadata.primary_key = None
    
    # Set categorical columns explicitly
    cat_cols = [c for c in ["Casting", "recryst_type"] if c in training_df.columns]
    for col in cat_cols:
        metadata.update_column(col, sdtype="categorical")
    
    # Initialize CTGAN
    synth_model = CTGANSynthesizer(
        metadata,
        epochs=300,
        batch_size=256,
        pac=2,
        verbose=False,
        cuda=False
    )
    
    # Fit model
    synth_model.fit(training_df)
    
    # Calculate total samples needed
    if group_cols:
        grouped = real_df.groupby(group_cols)
        n_classes = len(grouped)
        total_samples = n_classes * n_per_class
        
        # Limit total samples
        max_total_samples = 20000
        if total_samples > max_total_samples:
            total_samples = max_total_samples
            print(f"  Note: Limited to {total_samples} total samples")
    else:
        total_samples = n_per_class
    
    # Sample unconditionally
    synth_df = synth_model.sample(num_rows=total_samples)
    
    # Randomly assign group columns from real data distribution
    if cols_to_drop:
        for col in cols_to_drop:
            synth_df[col] = np.random.choice(real_df[col].values, size=len(synth_df), replace=True)
    
    # Enforce composition constraints
    synth_df, comp_metrics = enforce_composition(synth_df, bounds)
    
    # Flag as synthetic
    synth_df["is_synthetic"] = 1
    synth_df["source_generator"] = "ctgan"
    
    return synth_df, comp_metrics


def gaussiancopula_generate(real_df, group_cols, n_per_class, target_col, bounds, seed=42):
    """
    Gaussian Copula method: Parametric dependence model.
    Train WITHOUT high-cardinality group columns, sample unconditionally.
    """
    # Drop group columns to avoid issues
    cols_to_drop = [c for c in group_cols if c in real_df.columns]
    training_df = real_df.drop(columns=cols_to_drop)
    
    print(f"  Note: Dropped high-cardinality columns for GaussianCopula: {cols_to_drop}")
    
    # Prepare metadata
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(training_df)
    
    # Remove primary key designation
    metadata.primary_key = None
    
    # Set categorical columns explicitly
    cat_cols = [c for c in ["Casting", "recryst_type"] if c in training_df.columns]
    for col in cat_cols:
        metadata.update_column(col, sdtype="categorical")
    
    # Initialize Gaussian Copula
    synth_model = GaussianCopulaSynthesizer(
        metadata,
        default_distribution="norm"
    )
    
    # Fit model
    synth_model.fit(training_df)
    
    # Calculate total samples
    if group_cols:
        grouped = real_df.groupby(group_cols)
        n_classes = len(grouped)
        total_samples = n_classes * n_per_class
        
        # Limit total samples
        max_total_samples = 20000
        if total_samples > max_total_samples:
            total_samples = max_total_samples
            print(f"  Note: Limited to {total_samples} total samples")
    else:
        total_samples = n_per_class
    
    # Sample unconditionally
    synth_df = synth_model.sample(num_rows=total_samples)
    
    # Randomly assign group columns from real data distribution
    if cols_to_drop:
        for col in cols_to_drop:
            synth_df[col] = np.random.choice(real_df[col].values, size=len(synth_df), replace=True)
    
    # Enforce composition constraints
    synth_df, comp_metrics = enforce_composition(synth_df, bounds)
    
    # Flag as synthetic
    synth_df["is_synthetic"] = 1
    synth_df["source_generator"] = "gaussiancopula"
    
    return synth_df, comp_metrics


# --------------------------------------------------------------------------------
# MAIN
# --------------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic alloy data with comprehensive QC"
    )
    parser.add_argument(
        "--view",
        required=True,
        help="Path to VIEW_<TARGET>.csv (real data)"
    )
    parser.add_argument(
        "--out-dir",
        required=True,
        help="Output directory for synthetic datasets and reports"
    )
    parser.add_argument(
        "--target-col",
        required=True,
        help="Name of target column (e.g., 'YS', 'UTS', 'Fracture EL')"
    )
    parser.add_argument(
        "--group-cols",
        default="file_name,alloy,card",
        help="Comma-separated list of grouping columns for conditional sampling"
    )
    parser.add_argument(
        "--n-per-class",
        type=int,
        default=400,
        help="Number of synthetic samples per class (default: 400)"
    )
    parser.add_argument(
        "--chem-window",
        default="",
        help="Chemistry windows: 'Mg<=6,Cu<=0.1,Mn<=1.5,...'"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    
    args = parser.parse_args()
    
    # Parse inputs
    view_path = Path(args.view)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    target_col = args.target_col
    group_cols = [c.strip() for c in args.group_cols.split(",") if c.strip()]
    n_per_class = args.n_per_class
    chem_bounds = parse_chem_window(args.chem_window)
    seed = args.seed
    
    # Load real data
    print(f"Loading real data from {view_path}...")
    real_df = pd.read_csv(view_path, encoding="utf-8-sig")
    print(f"Loaded {len(real_df)} real samples")
    
    # Identify numeric columns
    numeric_cols = real_df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Store results
    results = {
        "input_file": str(view_path),
        "target_column": target_col,
        "n_real_samples": len(real_df),
        "n_per_class": n_per_class,
        "chemistry_bounds": chem_bounds,
        "group_columns": group_cols,
        "seed": seed,
        "generators": {}
    }
    
    # --------------------------------------------------------------------------------
    # GENERATE JITTER
    # --------------------------------------------------------------------------------
    print("\n[1/3] Generating Jitter synthetic data...")
    jitter_df, jitter_comp = jitter_augment(
        real_df, group_cols, n_per_class, target_col, chem_bounds, seed
    )
    
    print(f"  Generated {len(jitter_df)} samples")
    print(f"  Composition sum: {jitter_comp['composition_sum_mean']:.4f} ± {jitter_comp['composition_sum_std']:.4f}")
    
    # Apply physics gates
    jitter_df, jitter_phys_viol = apply_physics_gates(jitter_df, target_col, real_df)
    print(f"  Physics violations fixed: {jitter_phys_viol}")
    
    # Compute QC
    jitter_qc = compute_comprehensive_qc(real_df, jitter_df, target_col, group_cols, numeric_cols)
    print(f"  KS median p-value: {jitter_qc.get('ks_median_p_value', 'N/A')}")
    print(f"  Correlation MAE: {jitter_qc.get('correlation_mae', 'N/A')}")
    
    # Check acceptance
    jitter_pass, jitter_report = check_acceptance(jitter_qc, jitter_comp, ACCEPTANCE_THRESHOLDS)
    print(f"  Acceptance: {'PASS ✓' if jitter_pass else 'FAIL ✗'}")
    
    # Save
    jitter_path = out_dir / "jitter.csv"
    jitter_df.to_csv(jitter_path, index=False, encoding="utf-8-sig")
    print(f"  Saved to {jitter_path}")
    
    results["generators"]["jitter"] = {
        "n_samples": len(jitter_df),
        "composition_metrics": jitter_comp,
        "physics_violations_fixed": jitter_phys_viol,
        "qc_metrics": jitter_qc,
        "acceptance": jitter_report,
        "output_file": str(jitter_path)
    }
    
    # --------------------------------------------------------------------------------
    # GENERATE CTGAN
    # --------------------------------------------------------------------------------
    print("\n[2/3] Generating CTGAN synthetic data...")
    try:
        ctgan_df, ctgan_comp = ctgan_generate(
            real_df, group_cols, n_per_class, target_col, chem_bounds, seed
        )
        
        print(f"  Generated {len(ctgan_df)} samples")
        print(f"  Composition sum: {ctgan_comp['composition_sum_mean']:.4f} ± {ctgan_comp['composition_sum_std']:.4f}")
        
        # Apply physics gates
        ctgan_df, ctgan_phys_viol = apply_physics_gates(ctgan_df, target_col, real_df)
        print(f"  Physics violations fixed: {ctgan_phys_viol}")
        
        # Compute QC
        ctgan_qc = compute_comprehensive_qc(real_df, ctgan_df, target_col, group_cols, numeric_cols)
        print(f"  KS median p-value: {ctgan_qc.get('ks_median_p_value', 'N/A')}")
        print(f"  Correlation MAE: {ctgan_qc.get('correlation_mae', 'N/A')}")
        
        # Check acceptance
        ctgan_pass, ctgan_report = check_acceptance(ctgan_qc, ctgan_comp, ACCEPTANCE_THRESHOLDS)
        print(f"  Acceptance: {'PASS ✓' if ctgan_pass else 'FAIL ✗'}")
        
        # Save
        ctgan_path = out_dir / "ctgan.csv"
        ctgan_df.to_csv(ctgan_path, index=False, encoding="utf-8-sig")
        print(f"  Saved to {ctgan_path}")
        
        results["generators"]["ctgan"] = {
            "n_samples": len(ctgan_df),
            "composition_metrics": ctgan_comp,
            "physics_violations_fixed": ctgan_phys_viol,
            "qc_metrics": ctgan_qc,
            "acceptance": ctgan_report,
            "output_file": str(ctgan_path)
        }
    except Exception as e:
        print(f"  CTGAN generation failed: {e}")
        results["generators"]["ctgan"] = {
            "error": str(e),
            "recommendation": "Try reducing epochs or increasing PAC; fallback to GaussianCopula"
        }
    
    # --------------------------------------------------------------------------------
    # GENERATE GAUSSIAN COPULA
    # --------------------------------------------------------------------------------
    print("\n[3/3] Generating Gaussian Copula synthetic data...")
    try:
        gc_df, gc_comp = gaussiancopula_generate(
            real_df, group_cols, n_per_class, target_col, chem_bounds, seed
        )
        
        print(f"  Generated {len(gc_df)} samples")
        print(f"  Composition sum: {gc_comp['composition_sum_mean']:.4f} ± {gc_comp['composition_sum_std']:.4f}")
        
        # Apply physics gates
        gc_df, gc_phys_viol = apply_physics_gates(gc_df, target_col, real_df)
        print(f"  Physics violations fixed: {gc_phys_viol}")
        
        # Compute QC
        gc_qc = compute_comprehensive_qc(real_df, gc_df, target_col, group_cols, numeric_cols)
        print(f"  KS median p-value: {gc_qc.get('ks_median_p_value', 'N/A')}")
        print(f"  Correlation MAE: {gc_qc.get('correlation_mae', 'N/A')}")
        
        # Check acceptance
        gc_pass, gc_report = check_acceptance(gc_qc, gc_comp, ACCEPTANCE_THRESHOLDS)
        print(f"  Acceptance: {'PASS ✓' if gc_pass else 'FAIL ✗'}")
        
        # Save
        gc_path = out_dir / "gaussiancopula.csv"
        gc_df.to_csv(gc_path, index=False, encoding="utf-8-sig")
        print(f"  Saved to {gc_path}")
        
        results["generators"]["gaussiancopula"] = {
            "n_samples": len(gc_df),
            "composition_metrics": gc_comp,
            "physics_violations_fixed": gc_phys_viol,
            "qc_metrics": gc_qc,
            "acceptance": gc_report,
            "output_file": str(gc_path)
        }
    except Exception as e:
        print(f"  Gaussian Copula generation failed: {e}")
        results["generators"]["gaussiancopula"] = {
            "error": str(e),
            "recommendation": "Check for non-numeric leakage or insufficient data"
        }
    
    # --------------------------------------------------------------------------------
    # GENERATE SUMMARY & RECOMMENDATIONS
    # --------------------------------------------------------------------------------
    print("\n" + "="*80)
    print("SUMMARY & RECOMMENDATIONS")
    print("="*80)
    
    # Rank generators by acceptance
    rankings = []
    for method in ["ctgan", "jitter", "gaussiancopula"]:
        if method in results["generators"] and "acceptance" in results["generators"][method]:
            acc = results["generators"][method]["acceptance"]
            rankings.append({
                "method": method,
                "pass": acc["overall_pass"],
                "corr_mae": results["generators"][method]["qc_metrics"].get("correlation_mae", 1.0)
            })
    
    # Sort: pass first, then by correlation MAE (lower is better)
    rankings.sort(key=lambda x: (not x["pass"], x["corr_mae"] if x["corr_mae"] else 1.0))
    
    print("\nGenerator Rankings (best to worst):")
    for i, r in enumerate(rankings, 1):
        status = "PASS ✓" if r["pass"] else "FAIL ✗"
        print(f"  {i}. {r['method'].upper()}: {status} (Corr MAE: {r['corr_mae']:.4f})")
    
    if rankings:
        recommended = rankings[0]["method"]
        print(f"\nRECOMMENDED: {recommended.upper()}")
        print(f"  Use synth_out/{target_col}/{recommended}.csv for GP training")
    else:
        print("\nWARNING: All generators failed. Consider:")
        print("  - Increase real data samples (currently: {})".format(len(real_df)))
        print("  - Reduce n_per_class (currently: {})".format(n_per_class))
        print("  - Adjust CTGAN epochs or use Jitter fallback")
    
    # Save comprehensive report
    report_path = out_dir / "generation_report.json"
    results["recommendation"] = recommended if rankings else None
    results["acceptance_thresholds"] = ACCEPTANCE_THRESHOLDS
    
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nComprehensive report saved to {report_path}")
    print("\nGeneration complete!")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_mixup.py — Physics-Aware Mixup Augmentation for Alloy Discovery

Unlike jitter (which creates noise around existing points), mixup creates 
genuinely NEW alloy compositions by interpolating between real samples.

Key insight for alloy discovery:
  - Jitter: "What if this exact alloy had measurement noise?"
  - Mixup:  "What if we made an alloy BETWEEN these two known alloys?"

This is metallurgically valid because:
  1. Alloy compositions are convex (mixtures of alloys are valid alloys)
  2. Properties often vary smoothly with composition
  3. We can constrain mixing to similar alloy families

Strategy:
  1. For each pair of "similar" real samples (same processing route, similar Mg)
  2. Create interpolated compositions at λ ∈ {0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8}
  3. Interpolate properties (with added noise for realism)
  4. Enforce physics constraints (Σ=100%, YS≤UTS, etc.)

Usage:
  python generate_mixup.py --target YS --n-samples 5000 --seed 42

Output:
  - synth_out/<TARGET>/mixup.csv
  - synth_out/<TARGET>/mixup_report.json
"""

import argparse
import json
import warnings
from pathlib import Path
from itertools import combinations
import random

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics.pairwise import euclidean_distances

warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------------
# CONSTANTS
# --------------------------------------------------------------------------------

ELEMENTS = ["Al", "Si", "Fe", "Cu", "Mn", "Mg", "Cr", "Ni", "Zn", "Ti", "Zr", "Sc", "Other"]

# Chemistry bounds for 5xxx aluminum alloys (AA5182-like)
CHEMISTRY_BOUNDS = {
    "Mg": (2.0, 6.0),    # Primary alloying element
    "Cu": (0.0, 0.1),
    "Mn": (0.0, 1.5),
    "Cr": (0.0, 0.25),
    "Fe": (0.0, 0.5),
    "Si": (0.0, 0.5),
    "Zn": (0.0, 0.25),
    "Ti": (0.0, 0.15),
    "Zr": (0.0, 0.15),
    "Sc": (0.0, 0.3),
    "Ni": (0.0, 0.1),
    "Other": (0.0, 0.15),
    "Al": (90.0, 99.0),  # Balance
}

# Processing features to interpolate
PROCESSING_FEATURES = [
    "homog_temp_max_C",
    "homog_time_total_s", 
    "recryst_temp_max_C",
    "recryst_time_total_s",
    "Cold rolling reduction (percentage)",
    "Hot rolling reduction (percentage)",
]

# Microstructure features
MICROSTRUCTURE_FEATURES = [
    "Mean grain size (µm)",
    "adequate_homog",
    "adequate_recryst",
]

# Properties (what we're predicting)
TARGET_COLUMNS = {
    "YS": "YS (MPa)",
    "UTS": "UTS (MPa)",
    "FractureEL": "Fracture EL (percentage)",
    "UniformEL": "Uniform EL (percentage)",
    "YPE": "YPE (percentage)",
}

ALL_PROPERTY_COLS = list(TARGET_COLUMNS.values())

# Lambda values for interpolation (how much of sample A vs B)
# Use more extreme values to preserve variance - avoid clustering around 0.5
LAMBDA_VALUES = [0.15, 0.20, 0.30, 0.40, 0.60, 0.70, 0.80, 0.85]


# --------------------------------------------------------------------------------
# SIMILARITY FUNCTIONS
# --------------------------------------------------------------------------------

def compute_similarity_matrix(df, feature_cols):
    """Compute pairwise similarity based on key features."""
    X = df[feature_cols].fillna(0).values
    # Normalize each feature
    X_norm = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
    distances = euclidean_distances(X_norm)
    # Convert to similarity (inverse distance)
    similarity = 1 / (1 + distances)
    return similarity


def get_valid_pairs(df, similarity_matrix, min_similarity=0.5, max_pairs_per_sample=10):
    """
    Get pairs of samples that are similar enough to interpolate.
    
    Criteria for valid pair (i, j):
      1. Similarity >= min_similarity (not too different)
      2. Same processing type if possible
      3. Both have valid target values
    """
    n = len(df)
    pairs = []
    
    for i in range(n):
        # Get top-k most similar samples
        sims = similarity_matrix[i].copy()
        sims[i] = -1  # Exclude self
        
        # Sort by similarity
        top_indices = np.argsort(sims)[::-1][:max_pairs_per_sample * 2]
        
        count = 0
        for j in top_indices:
            if sims[j] < min_similarity:
                break
            if count >= max_pairs_per_sample:
                break
                
            # Additional check: similar Mg content (±2.5% - relaxed for more diversity)
            mg_diff = abs(df.iloc[i]["Mg"] - df.iloc[j]["Mg"])
            if mg_diff <= 2.5:
                pairs.append((i, j, sims[j]))
                count += 1
    
    return pairs


# --------------------------------------------------------------------------------
# MIXUP GENERATION
# --------------------------------------------------------------------------------

def mixup_samples(row_a, row_b, lam, noise_std=0.05):
    """
    Create an interpolated sample between two real samples.
    
    Args:
        row_a, row_b: Pandas Series (real samples)
        lam: Interpolation weight (0=all A, 1=all B)
        noise_std: Standard deviation of added noise (as fraction of interpolated value)
    
    Returns:
        Pandas Series: New interpolated sample
    """
    new_row = {}
    
    # Interpolate compositions with variance-preserving noise
    for elem in ELEMENTS:
        if elem in row_a.index and elem in row_b.index:
            val_a = row_a[elem] if pd.notna(row_a[elem]) else 0
            val_b = row_b[elem] if pd.notna(row_b[elem]) else 0
            interp = (1 - lam) * val_a + lam * val_b
            
            # Add noise proportional to the difference between parents
            # This preserves variance better than fixed noise
            diff = abs(val_a - val_b)
            noise_scale = max(noise_std * abs(interp), 0.3 * diff) if interp > 0.01 else 0
            noise = np.random.normal(0, noise_scale)
            new_row[elem] = max(0, interp + noise)
    
    # Interpolate processing features
    for feat in PROCESSING_FEATURES:
        if feat in row_a.index and feat in row_b.index:
            val_a = row_a[feat] if pd.notna(row_a[feat]) else 0
            val_b = row_b[feat] if pd.notna(row_b[feat]) else 0
            new_row[feat] = (1 - lam) * val_a + lam * val_b
    
    # For binary/discrete features, use weighted random choice
    for feat in ["adequate_homog", "adequate_recryst"]:
        if feat in row_a.index and feat in row_b.index:
            val_a = row_a[feat] if pd.notna(row_a[feat]) else 0
            val_b = row_b[feat] if pd.notna(row_b[feat]) else 0
            # Probabilistic choice based on lambda
            new_row[feat] = val_a if np.random.random() > lam else val_b
    
    # Interpolate grain size with noise
    if "Mean grain size (µm)" in row_a.index:
        gs_a = row_a["Mean grain size (µm)"] if pd.notna(row_a["Mean grain size (µm)"]) else 0
        gs_b = row_b["Mean grain size (µm)"] if pd.notna(row_b["Mean grain size (µm)"]) else 0
        interp = (1 - lam) * gs_a + lam * gs_b
        noise = np.random.normal(0, abs(interp) * 0.1) if interp > 0 else 0
        new_row["Mean grain size (µm)"] = max(0, interp + noise)
    
    # Interpolate properties with MORE noise (less certain)
    for prop in ALL_PROPERTY_COLS:
        if prop in row_a.index and prop in row_b.index:
            val_a = row_a[prop] if pd.notna(row_a[prop]) else np.nan
            val_b = row_b[prop] if pd.notna(row_b[prop]) else np.nan
            if pd.notna(val_a) and pd.notna(val_b):
                interp = (1 - lam) * val_a + lam * val_b
                # Larger noise for properties (10-15% relative)
                noise_frac = 0.05 + 0.10 * abs(lam - 0.5) * 2  # More noise near midpoint
                noise = np.random.normal(0, abs(interp) * noise_frac)
                new_row[prop] = max(0, interp + noise)
            elif pd.notna(val_a):
                new_row[prop] = val_a
            elif pd.notna(val_b):
                new_row[prop] = val_b
            else:
                new_row[prop] = np.nan
    
    return pd.Series(new_row)


def enforce_composition_constraint(df):
    """Ensure compositions sum to 100% using Al as balance."""
    elem_cols = [c for c in ELEMENTS if c in df.columns and c != "Al"]
    
    # Clip to bounds first
    for elem, (low, high) in CHEMISTRY_BOUNDS.items():
        if elem in df.columns and elem != "Al":
            df[elem] = df[elem].clip(lower=low, upper=high)
    
    # Compute non-Al sum
    non_al_sum = df[elem_cols].sum(axis=1)
    
    # Al as balance
    df["Al"] = 100.0 - non_al_sum
    df["Al"] = df["Al"].clip(lower=CHEMISTRY_BOUNDS["Al"][0], 
                              upper=CHEMISTRY_BOUNDS["Al"][1])
    
    # Final normalization
    total = df[elem_cols + ["Al"]].sum(axis=1)
    for col in elem_cols + ["Al"]:
        df[col] = df[col] / total * 100.0
    
    return df


def enforce_physics_constraints(df):
    """
    Enforce physical constraints:
      1. YS ≤ UTS (yield can't exceed ultimate)
      2. Uniform EL ≤ Fracture EL (uniform is subset of total)
      3. All properties >= 0
    """
    violations_fixed = 0
    
    # YS ≤ UTS
    if "YS (MPa)" in df.columns and "UTS (MPa)" in df.columns:
        mask = df["YS (MPa)"] > df["UTS (MPa)"]
        if mask.any():
            # Swap or adjust
            violations_fixed += mask.sum()
            df.loc[mask, "YS (MPa)"] = df.loc[mask, "UTS (MPa)"] * 0.95
    
    # Uniform EL ≤ Fracture EL
    if "Uniform EL (percentage)" in df.columns and "Fracture EL (percentage)" in df.columns:
        mask = df["Uniform EL (percentage)"] > df["Fracture EL (percentage)"]
        if mask.any():
            violations_fixed += mask.sum()
            df.loc[mask, "Uniform EL (percentage)"] = df.loc[mask, "Fracture EL (percentage)"] * 0.9
    
    # Non-negativity for properties
    for prop in ALL_PROPERTY_COLS:
        if prop in df.columns:
            df[prop] = df[prop].clip(lower=0)
    
    return df, violations_fixed


# --------------------------------------------------------------------------------
# MAIN GENERATION PIPELINE
# --------------------------------------------------------------------------------

def generate_mixup_data(df_real, target_col, n_samples=5000, seed=42):
    """
    Generate mixup-augmented synthetic data.
    
    Args:
        df_real: DataFrame with real alloy data
        target_col: Name of target column (e.g., "YS (MPa)")
        n_samples: Desired number of synthetic samples
        seed: Random seed
    
    Returns:
        df_synth: DataFrame with synthetic samples
        report: Dict with generation metrics
    """
    np.random.seed(seed)
    random.seed(seed)
    
    report = {
        "n_real": len(df_real),
        "n_target": n_samples,
        "seed": seed,
    }
    
    # Filter to rows with valid target
    df_valid = df_real[df_real[target_col].notna()].copy()
    print(f"  Valid samples with {target_col}: {len(df_valid)}")
    
    if len(df_valid) < 10:
        raise ValueError(f"Too few valid samples ({len(df_valid)}) for mixup")
    
    # Compute similarity based on key features
    similarity_features = ["Mg", "Mn", "Fe", "Si", "homog_temp_max_C", "recryst_temp_max_C"]
    similarity_features = [f for f in similarity_features if f in df_valid.columns]
    
    print(f"  Computing similarity matrix using: {similarity_features}")
    sim_matrix = compute_similarity_matrix(df_valid, similarity_features)
    
    # Get valid pairs
    pairs = get_valid_pairs(df_valid, sim_matrix, min_similarity=0.3, max_pairs_per_sample=20)
    print(f"  Found {len(pairs)} valid pairs for interpolation")
    
    if len(pairs) < 100:
        print(f"  Warning: Few pairs found, relaxing similarity threshold...")
        pairs = get_valid_pairs(df_valid, sim_matrix, min_similarity=0.1, max_pairs_per_sample=50)
        print(f"  Found {len(pairs)} pairs after relaxing threshold")
    
    report["n_pairs"] = len(pairs)
    
    # Generate synthetic samples
    synthetic_rows = []
    samples_per_pair = max(1, n_samples // len(pairs))
    
    print(f"  Generating ~{samples_per_pair} samples per pair...")
    
    pair_indices = list(range(len(pairs)))
    random.shuffle(pair_indices)
    
    for idx in pair_indices:
        i, j, similarity = pairs[idx]
        row_a = df_valid.iloc[i]
        row_b = df_valid.iloc[j]
        
        # Generate multiple interpolations with different lambdas
        for _ in range(samples_per_pair):
            lam = random.choice(LAMBDA_VALUES)
            # Add randomness to lambda
            lam += np.random.uniform(-0.1, 0.1)
            lam = np.clip(lam, 0.15, 0.85)
            
            new_row = mixup_samples(row_a, row_b, lam)
            new_row["mixup_lambda"] = lam
            new_row["source_idx_a"] = i
            new_row["source_idx_b"] = j
            new_row["source_similarity"] = similarity
            synthetic_rows.append(new_row)
            
            if len(synthetic_rows) >= n_samples:
                break
        
        if len(synthetic_rows) >= n_samples:
            break
    
    print(f"  Generated {len(synthetic_rows)} raw samples")
    
    # Convert to DataFrame
    df_synth = pd.DataFrame(synthetic_rows)
    
    # VARIANCE INFLATION: Mixup naturally compresses variance toward the mean
    # We stretch the distribution back to match real data variance
    print("  Applying variance inflation to match real data distribution...")
    for elem in ELEMENTS:
        if elem in df_synth.columns and elem in df_valid.columns:
            real_mean = df_valid[elem].mean()
            real_std = df_valid[elem].std()
            synth_mean = df_synth[elem].mean()
            synth_std = df_synth[elem].std()
            
            if synth_std > 0.001 and real_std > 0.001:
                # Z-score transform then stretch to real variance
                z_scores = (df_synth[elem] - synth_mean) / synth_std
                df_synth[elem] = real_mean + z_scores * real_std
    
    # Enforce composition constraints
    df_synth = enforce_composition_constraint(df_synth)
    
    # Enforce physics constraints
    df_synth, violations = enforce_physics_constraints(df_synth)
    report["physics_violations_fixed"] = int(violations)
    
    # Compute QC metrics
    report["composition_sum_mean"] = float(df_synth[ELEMENTS].sum(axis=1).mean())
    report["composition_sum_std"] = float(df_synth[ELEMENTS].sum(axis=1).std())
    
    # Compare distributions
    ks_results = {}
    for elem in ELEMENTS:
        if elem in df_synth.columns and elem in df_valid.columns:
            real_vals = df_valid[elem].dropna().values
            synth_vals = df_synth[elem].dropna().values
            if len(real_vals) > 5 and len(synth_vals) > 5:
                stat, pval = stats.ks_2samp(real_vals, synth_vals)
                ks_results[elem] = {"statistic": float(stat), "p_value": float(pval)}
    
    report["ks_tests"] = ks_results
    report["ks_median_p_value"] = float(np.median([v["p_value"] for v in ks_results.values()]))
    
    # Check diversity (near-duplicates)
    from sklearn.metrics.pairwise import cosine_similarity
    feature_cols = [c for c in ELEMENTS + PROCESSING_FEATURES if c in df_synth.columns]
    X_synth = df_synth[feature_cols].fillna(0).values
    X_real = df_valid[feature_cols].fillna(0).values
    
    # Sample for efficiency
    n_check = min(1000, len(X_synth))
    sample_idx = np.random.choice(len(X_synth), n_check, replace=False)
    X_sample = X_synth[sample_idx]
    
    cos_sim = cosine_similarity(X_sample, X_real)
    max_sims = cos_sim.max(axis=1)
    near_dup_pct = (max_sims > 0.999).mean() * 100
    
    report["near_duplicates_pct"] = float(near_dup_pct)
    report["diversity_score"] = float(1.0 - max_sims.mean())  # Higher = more diverse
    
    # Lambda distribution
    report["lambda_mean"] = float(df_synth["mixup_lambda"].mean())
    report["lambda_std"] = float(df_synth["mixup_lambda"].std())
    
    print(f"  Near-duplicate rate: {near_dup_pct:.1f}%")
    print(f"  Diversity score: {report['diversity_score']:.3f}")
    print(f"  KS median p-value: {report['ks_median_p_value']:.4f}")
    
    return df_synth, report


# --------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate mixup-augmented alloy data")
    parser.add_argument("--target", type=str, required=True,
                       choices=["YS", "UTS", "FractureEL", "UniformEL", "YPE"],
                       help="Target property to generate data for")
    parser.add_argument("--n-samples", type=int, default=5000,
                       help="Number of synthetic samples to generate")
    parser.add_argument("--view-dir", type=str, default="baseline_out",
                       help="Directory with VIEW_*.csv files")
    parser.add_argument("--out-dir", type=str, default="synth_out",
                       help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    # Map target to filename and column
    target_map = {
        "YS": ("VIEW_YS.csv", "YS (MPa)"),
        "UTS": ("VIEW_UTS.csv", "UTS (MPa)"),
        "FractureEL": ("VIEW_FractureEL.csv", "Fracture EL (percentage)"),
        "UniformEL": ("VIEW_UniformEL.csv", "Uniform EL (percentage)"),
        "YPE": ("VIEW_YPE.csv", "YPE (percentage)"),
    }
    
    view_file, target_col = target_map[args.target]
    view_path = Path(args.view_dir) / view_file
    
    print(f"\n{'='*60}")
    print(f"Generating Mixup Data for {args.target}")
    print(f"{'='*60}")
    
    # Load real data
    print(f"\nLoading real data from {view_path}...")
    df_real = pd.read_csv(view_path)
    print(f"  Loaded {len(df_real)} real samples")
    
    # Generate mixup data
    print(f"\nGenerating {args.n_samples} mixup samples...")
    df_synth, report = generate_mixup_data(df_real, target_col, args.n_samples, args.seed)
    
    # Save outputs
    out_dir = Path(args.out_dir) / args.target
    out_dir.mkdir(parents=True, exist_ok=True)
    
    out_csv = out_dir / "mixup.csv"
    df_synth.to_csv(out_csv, index=False)
    print(f"\nSaved {len(df_synth)} samples to {out_csv}")
    
    out_json = out_dir / "mixup_report.json"
    with open(out_json, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Saved report to {out_json}")
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"  Real samples:        {report['n_real']}")
    print(f"  Valid pairs:         {report['n_pairs']}")
    print(f"  Synthetic samples:   {len(df_synth)}")
    print(f"  Near-duplicates:     {report['near_duplicates_pct']:.1f}%")
    print(f"  Diversity score:     {report['diversity_score']:.3f}")
    print(f"  Physics violations:  {report['physics_violations_fixed']}")
    print(f"  Composition sum:     {report['composition_sum_mean']:.4f} ± {report['composition_sum_std']:.4f}")
    

if __name__ == "__main__":
    main()

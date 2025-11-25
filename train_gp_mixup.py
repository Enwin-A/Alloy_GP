#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_gp_mixup.py - Train GP Models with Mixup-Augmented Data for Alloy Discovery

Strategy:
  1. Load MIXUP synthetic data (diverse interpolated compositions)
  2. Train GP with proper k-fold cross-validation
  3. Validate on held-out REAL data
  4. Report honest uncertainty estimates for alloy discovery

Key difference from jitter approach:
  - Mixup data explores BETWEEN known alloys (good for discovery)
  - Jitter data stays AT known alloys (good for measurement uncertainty)

Usage:
  python train_gp_mixup.py --train-type mixup --n-samples 3500

The GP trained on mixup data will:
  - Predict properties for NOVEL alloy compositions
  - Provide uncertainty that reflects exploration vs exploitation
  - Enable inverse design optimization
"""

import argparse
import json
import pickle
import warnings
from pathlib import Path
from datetime import datetime
import time
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel as C
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

try:
    from scipy.optimize import minimize
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

warnings.filterwarnings("ignore", category=UserWarning)

# --------------------------------------------------------------------------------
# FEATURE CONFIGURATION
# --------------------------------------------------------------------------------

INPUT_FEATURES = [
    # Composition (13 elements)
    "Al", "Si", "Fe", "Cu", "Mn", "Mg", "Cr", "Ni", "Zn", "Ti", "Zr", "Sc", "Other",
    # Processing (6 features)
    "homog_temp_max_C", "homog_time_total_s",
    "recryst_temp_max_C", "recryst_time_total_s",
    "Cold rolling reduction (percentage)", "Hot rolling reduction (percentage)",
    # Microstructure (3 features)
    "Mean grain size (µm)", "adequate_homog", "adequate_recryst",
]

TARGETS = {
    "YS": "YS (MPa)",
    "UTS": "UTS (MPa)",
    "FractureEL": "Fracture EL (percentage)",
    "UniformEL": "Uniform EL (percentage)",
    "YPE": "YPE (percentage)"
}

# --------------------------------------------------------------------------------
# GP MODEL
# --------------------------------------------------------------------------------

def create_gp_kernel(n_features):
    """Create Matérn 5/2 kernel with ARD."""
    amplitude = C(1.0, constant_value_bounds=(0.01, 100.0))
    matern = Matern(
        nu=2.5,
        length_scale=np.ones(n_features),
        length_scale_bounds=(0.01, 10.0)
    )
    noise = WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-5, 10.0))
    return amplitude * matern + noise


def train_gp_with_progress(X_train, y_train, target_name, max_iter=150, seed=42):
    """Train GP with manual optimization and progress tracking."""
    print(f"\n  Training GP for {target_name}...")
    print(f"  Training samples: {len(X_train)}")
    
    kernel = create_gp_kernel(X_train.shape[1])
    
    gp = GaussianProcessRegressor(
        kernel=kernel,
        optimizer=None,
        normalize_y=True,
        alpha=1e-10,
        random_state=seed
    )
    
    start_time = time.time()
    gp.fit(X_train, y_train)
    
    if SCIPY_AVAILABLE:
        initial_theta = gp.kernel_.theta
        bounds = gp.kernel_.bounds
        iter_count = [0]
        
        def obj_func(theta):
            lml, grad = gp.log_marginal_likelihood(theta, eval_gradient=True)
            return -lml, -grad
        
        def callback(theta):
            iter_count[0] += 1
            if iter_count[0] % 25 == 0:
                lml = gp.log_marginal_likelihood(theta, eval_gradient=False)
                elapsed = time.time() - start_time
                print(f"    Iter {iter_count[0]}: LML={lml:.2f}, elapsed={elapsed:.1f}s")
        
        res = minimize(
            lambda th: obj_func(th)[0],
            x0=initial_theta,
            jac=lambda th: obj_func(th)[1],
            bounds=bounds,
            method='L-BFGS-B',
            callback=callback,
            options={'maxiter': max_iter}
        )
        
        gp.kernel_.theta = res.x
        gp.fit(X_train, y_train)
        
        elapsed = time.time() - start_time
        print(f"  ✓ Done in {elapsed:.1f}s, LML={gp.log_marginal_likelihood_value_:.2f}")
    else:
        # Fallback to sklearn optimizer
        gp = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=3,
            normalize_y=True,
            alpha=1e-10,
            random_state=seed
        )
        gp.fit(X_train, y_train)
        elapsed = time.time() - start_time
        print(f"  ✓ Done in {elapsed:.1f}s")
    
    return gp


def evaluate_model(gp, X_test, y_test, scaler):
    """Evaluate GP model and return metrics."""
    X_test_scaled = scaler.transform(X_test)
    y_pred, y_std = gp.predict(X_test_scaled, return_std=True)
    
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    
    # Calibration
    errors = np.abs(y_test - y_pred)
    within_1sigma = (errors <= y_std).mean()
    within_2sigma = (errors <= 2 * y_std).mean()
    
    return {
        "r2": r2,
        "rmse": rmse,
        "mae": mae,
        "within_1sigma": within_1sigma,
        "within_2sigma": within_2sigma,
        "mean_std": y_std.mean(),
    }


# --------------------------------------------------------------------------------
# DATA LOADING
# --------------------------------------------------------------------------------

def load_training_data(target_name, train_type="mixup", synth_dir="synth_out", max_samples=3500):
    """Load synthetic training data."""
    if train_type == "mixup":
        csv_path = Path(synth_dir) / target_name / "mixup.csv"
    else:
        csv_path = Path(synth_dir) / target_name / "jitter.csv"
    
    if not csv_path.exists():
        raise FileNotFoundError(f"Training data not found: {csv_path}")
    
    df = pd.read_csv(csv_path)
    print(f"  Loaded {len(df)} samples from {csv_path}")
    
    # Subsample if needed (GP is O(N³))
    if len(df) > max_samples:
        print(f"  Subsampling to {max_samples} for GP tractability")
        df = df.sample(n=max_samples, random_state=42)
    
    return df


def load_validation_data(target_name, val_dir="baseline_out"):
    """Load real validation data."""
    csv_path = Path(val_dir) / f"VIEW_{target_name}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Validation data not found: {csv_path}")
    
    df = pd.read_csv(csv_path)
    print(f"  Loaded {len(df)} real samples for validation")
    return df


def prepare_features(df, target_col, features=None):
    """Extract features and target, handling missing values."""
    if features is None:
        features = INPUT_FEATURES
    
    # Get available features
    available = [f for f in features if f in df.columns]
    missing = [f for f in features if f not in df.columns]
    if missing:
        print(f"  Warning: Missing features: {missing}")
    
    # Filter rows with valid target
    df_valid = df[df[target_col].notna()].copy()
    
    # Fill missing feature values
    X = df_valid[available].fillna(0).values
    y = df_valid[target_col].values
    
    return X, y, available


# --------------------------------------------------------------------------------
# MAIN TRAINING PIPELINE
# --------------------------------------------------------------------------------

def train_and_validate(target_name, train_type="mixup", max_samples=3500, 
                       n_folds=5, synth_dir="synth_out", val_dir="baseline_out",
                       out_dir="gp_models", seed=42):
    """
    Full training and validation pipeline for one target.
    
    Strategy:
      1. Load synthetic data (mixup or jitter)
      2. k-fold CV on synthetic data (assess training)
      3. Final model trained on all synthetic data
      4. Validate on 100% of real data (assess generalization)
    """
    target_col = TARGETS[target_name]
    print(f"\n{'='*60}")
    print(f"Training GP for {target_name} ({target_col})")
    print(f"{'='*60}")
    
    # Load data
    print("\n[1] Loading data...")
    df_train = load_training_data(target_name, train_type, synth_dir, max_samples)
    df_val = load_validation_data(target_name, val_dir)
    
    # Prepare features
    X_train, y_train, feature_names = prepare_features(df_train, target_col)
    X_val, y_val, _ = prepare_features(df_val, target_col, feature_names)
    
    print(f"  Training: {X_train.shape[0]} samples × {X_train.shape[1]} features")
    print(f"  Validation: {X_val.shape[0]} samples × {X_val.shape[1]} features")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # k-fold CV on training data
    print(f"\n[2] {n_folds}-fold Cross-Validation on synthetic data...")
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    cv_results = []
    
    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X_train_scaled)):
        print(f"\n  Fold {fold_idx + 1}/{n_folds}:")
        
        X_fold_train = X_train_scaled[train_idx]
        y_fold_train = y_train[train_idx]
        X_fold_test = X_train[test_idx]  # Unscaled for evaluate_model
        y_fold_test = y_train[test_idx]
        
        # Train on fold (faster settings for CV)
        gp_fold = train_gp_with_progress(X_fold_train, y_fold_train, 
                                          f"{target_name}-fold{fold_idx+1}", 
                                          max_iter=75, seed=seed)
        
        # Evaluate on fold test set
        metrics = evaluate_model(gp_fold, X_fold_test, y_fold_test, scaler)
        cv_results.append(metrics)
        print(f"    R²={metrics['r2']:.3f}, RMSE={metrics['rmse']:.2f}, "
              f"Cal±1σ={metrics['within_1sigma']:.1%}")
    
    # Summarize CV
    cv_summary = {
        "r2_mean": np.mean([r["r2"] for r in cv_results]),
        "r2_std": np.std([r["r2"] for r in cv_results]),
        "rmse_mean": np.mean([r["rmse"] for r in cv_results]),
        "rmse_std": np.std([r["rmse"] for r in cv_results]),
        "calibration_1sigma_mean": np.mean([r["within_1sigma"] for r in cv_results]),
    }
    print(f"\n  CV Summary: R²={cv_summary['r2_mean']:.3f}±{cv_summary['r2_std']:.3f}, "
          f"RMSE={cv_summary['rmse_mean']:.2f}±{cv_summary['rmse_std']:.2f}")
    
    # Train final model on all synthetic data
    print(f"\n[3] Training final model on all {len(X_train_scaled)} synthetic samples...")
    gp_final = train_gp_with_progress(X_train_scaled, y_train, target_name, 
                                       max_iter=200, seed=seed)
    
    # Validate on real data
    print(f"\n[4] Validating on {len(X_val)} REAL samples...")
    val_metrics = evaluate_model(gp_final, X_val, y_val, scaler)
    
    print(f"\n  REAL DATA VALIDATION:")
    print(f"    R² score:      {val_metrics['r2']:.4f}")
    print(f"    RMSE:          {val_metrics['rmse']:.2f}")
    print(f"    MAE:           {val_metrics['mae']:.2f}")
    print(f"    Mean σ:        {val_metrics['mean_std']:.2f}")
    print(f"    Within ±1σ:    {val_metrics['within_1sigma']:.1%} (expect ~68%)")
    print(f"    Within ±2σ:    {val_metrics['within_2sigma']:.1%} (expect ~95%)")
    
    # Warnings
    if val_metrics['r2'] < 0.5:
        print(f"  ⚠️  WARNING: Low R² on real data - model may not generalize well")
    if val_metrics['within_1sigma'] > 0.85:
        print(f"  ⚠️  WARNING: Overconfident uncertainty estimates")
    if val_metrics['within_1sigma'] < 0.50:
        print(f"  ⚠️  WARNING: Underconfident uncertainty estimates")
    
    # Save model
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    model_pkg = {
        "gp_model": gp_final,
        "scaler": scaler,
        "feature_names": feature_names,
        "target_name": target_name,
        "target_col": target_col,
        "train_type": train_type,
        "cv_summary": cv_summary,
        "val_metrics": val_metrics,
        "n_train": len(X_train),
        "n_val": len(X_val),
        "trained_at": datetime.now().isoformat(),
    }
    
    model_file = out_path / f"gp_{target_name}_{train_type}.pkl"
    with open(model_file, "wb") as f:
        pickle.dump(model_pkg, f)
    print(f"\n  Saved model to {model_file}")
    
    # Save metrics as JSON
    metrics_file = out_path / f"metrics_{target_name}_{train_type}.json"
    metrics_json = {
        "target": target_name,
        "train_type": train_type,
        "cv_summary": cv_summary,
        "validation": {k: float(v) for k, v in val_metrics.items()},
        "n_train": len(X_train),
        "n_val": len(X_val),
    }
    with open(metrics_file, "w") as f:
        json.dump(metrics_json, f, indent=2)
    
    return model_pkg


# --------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train GP models with mixup data")
    parser.add_argument("--train-type", type=str, default="mixup",
                       choices=["mixup", "jitter"],
                       help="Type of synthetic data to use")
    parser.add_argument("--targets", type=str, default="all",
                       help="Comma-separated targets or 'all'")
    parser.add_argument("--n-samples", type=int, default=3500,
                       help="Max training samples (GP is O(N³))")
    parser.add_argument("--n-folds", type=int, default=5,
                       help="Number of CV folds")
    parser.add_argument("--synth-dir", type=str, default="synth_out")
    parser.add_argument("--val-dir", type=str, default="baseline_out")
    parser.add_argument("--out-dir", type=str, default="gp_models")
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    
    # Parse targets
    if args.targets == "all":
        targets = list(TARGETS.keys())
    else:
        targets = [t.strip() for t in args.targets.split(",")]
    
    print("="*70)
    print("GP MODEL TRAINING WITH MIXUP AUGMENTATION")
    print("="*70)
    print(f"Train type:     {args.train_type}")
    print(f"Targets:        {targets}")
    print(f"Max samples:    {args.n_samples}")
    print(f"CV folds:       {args.n_folds}")
    print("="*70)
    
    results = {}
    for target in targets:
        try:
            result = train_and_validate(
                target_name=target,
                train_type=args.train_type,
                max_samples=args.n_samples,
                n_folds=args.n_folds,
                synth_dir=args.synth_dir,
                val_dir=args.val_dir,
                out_dir=args.out_dir,
                seed=args.seed
            )
            results[target] = {
                "status": "success",
                "val_r2": result["val_metrics"]["r2"],
                "val_rmse": result["val_metrics"]["rmse"],
                "calibration": result["val_metrics"]["within_1sigma"],
            }
        except Exception as e:
            print(f"\n  ERROR training {target}: {e}")
            results[target] = {"status": "failed", "error": str(e)}
    
    # Final summary
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)
    print(f"{'Target':<15} {'Status':<10} {'Val R²':<10} {'Val RMSE':<10} {'Cal ±1σ':<10}")
    print("-"*55)
    for target, r in results.items():
        if r["status"] == "success":
            print(f"{target:<15} {'✓ OK':<10} {r['val_r2']:.4f}     {r['val_rmse']:.2f}       {r['calibration']:.1%}")
        else:
            print(f"{target:<15} {'✗ FAIL':<10} —          —          —")
    
    print("\n" + "="*70)
    print("Models saved to:", args.out_dir)
    print("="*70)


if __name__ == "__main__":
    main()

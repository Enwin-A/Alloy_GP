#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_gp_models.py - Gaussian Process Model Training for Aluminum Alloy Design

Train GP regression models for 5 mechanical properties:
  1. YS (Yield Strength)
  2. UTS (Ultimate Tensile Strength)
  3. Fracture EL (Fracture Elongation)
  4. Uniform EL (Uniform Elongation)
  5. YPE (Yield Point Elongation)

Training Strategy:
  - Train on synthetic Jitter data (19K-20K samples per target)
  - Validate on real VIEW data (64-333 samples per target)
  - Use Matérn 5/2 kernel (twice-differentiable, good for physical systems)
  - Optimize hyperparameters via log-marginal likelihood
  - Export trained models for inverse design

Usage:
  python train_gp_models.py --train-dir synth_out --val-dir baseline_out --out-dir gp_models
"""

import argparse
import json
import pickle
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel as C
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

warnings.filterwarnings("ignore", category=UserWarning)

# --------------------------------------------------------------------------------
# FEATURE SELECTION
# --------------------------------------------------------------------------------

# Input features for GP models (22 features total)
INPUT_FEATURES = [
    # Composition (13 elements)
    "Al", "Si", "Fe", "Cu", "Mn", "Mg", "Cr", "Ni", "Zn", "Ti", "Zr", "Sc", "Other",
    
    # Processing (6 features)
    "homog_temp_max_C",
    "homog_time_total_s",
    "recryst_temp_max_C",
    "recryst_time_total_s",
    "Cold rolling reduction (percentage)",
    "Hot rolling reduction (percentage)",
    
    # Microstructure (3 features)
    "Mean grain size (µm)",
    "adequate_homog",
    "adequate_recryst",
]

# Target properties (5 outputs)
TARGETS = {
    "YS": "YS (MPa)",
    "UTS": "UTS (MPa)",
    "FractureEL": "Fracture EL (percentage)",
    "UniformEL": "Uniform EL (percentage)",
    "YPE": "YPE (percentage)"
}

# --------------------------------------------------------------------------------
# GP MODEL CONFIGURATION
# --------------------------------------------------------------------------------

def create_gp_kernel():
    """
    Create Matérn 5/2 kernel with automatic relevance determination (ARD).
    
    Kernel structure:
      k(x, x') = σ² * Matérn(x, x' | ν=2.5, l) + σ_noise² * δ(x, x')
    
    Where:
      - σ² (amplitude): Overall output variance
      - l (length_scale): How far correlations extend (one per feature with ARD)
      - ν=2.5: Twice-differentiable (smooth but not too smooth)
      - σ_noise²: Observation noise variance
    """
    # Amplitude (output variance) - bounds: [0.01, 100]
    amplitude = C(1.0, constant_value_bounds=(0.01, 100.0))
    
    # Matérn kernel with ARD (separate length scale per feature)
    # Length scales initialized to 1.0, bounds: [0.01, 10]
    # ARD allows GP to learn which features are most important
    n_features = len(INPUT_FEATURES)
    matern = Matern(
        nu=2.5,
        length_scale=np.ones(n_features),
        length_scale_bounds=(0.01, 10.0)
    )
    
    # White noise kernel (measurement uncertainty)
    # Bounds: [1e-5, 10] - prevents overfitting while allowing flexibility
    noise = WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-5, 10.0))
    
    # Combined kernel
    kernel = amplitude * matern + noise
    
    return kernel


def train_gp_model(X_train, y_train, target_name, seed=42):
    """
    Train Gaussian Process regressor with optimized hyperparameters.
    
    Args:
        X_train: Training features (N × 22)
        y_train: Training targets (N,)
        target_name: Name of target property (for logging)
        seed: Random seed for reproducibility
    
    Returns:
        Trained GP model
    """
    print(f"\nTraining GP for {target_name}...")
    print(f"  Training samples: {len(X_train)}")
    
    # Create kernel
    kernel = create_gp_kernel()
    
    # Initialize GP
    gp = GaussianProcessRegressor(
        kernel=kernel,
        n_restarts_optimizer=10,  # Restart optimization 10 times to avoid local minima
        normalize_y=True,         # Standardize targets (improves numerical stability)
        alpha=1e-10,              # Small regularization for numerical stability
        random_state=seed
    )
    
    # Fit model (optimizes hyperparameters via log-marginal likelihood)
    print(f"  Optimizing hyperparameters (this may take 2-5 minutes)...")
    gp.fit(X_train, y_train)
    
    # Report optimized hyperparameters
    print(f"  ✓ Training complete!")
    print(f"  Log-marginal likelihood: {gp.log_marginal_likelihood_value_:.2f}")
    print(f"  Kernel: {gp.kernel_}")
    
    return gp


def validate_gp_model(gp, X_val, y_val, scaler, target_name, target_range=None):
    """
    Validate GP model on real data and compute metrics.
    
    Args:
        gp: Trained GP model
        X_val: Validation features (M × 22)
        y_val: True validation targets (M,)
        scaler: Feature scaler used during training
        target_name: Name of target property
        target_range: (min, max) of target in training data
    
    Returns:
        Dictionary of validation metrics
    """
    print(f"\nValidating {target_name} on real data...")
    print(f"  Validation samples: {len(X_val)}")
    
    # Scale validation features
    X_val_scaled = scaler.transform(X_val)
    
    # Predict with uncertainty
    y_pred, y_std = gp.predict(X_val_scaled, return_std=True)
    
    # Compute metrics
    r2 = r2_score(y_val, y_pred)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    mae = mean_absolute_error(y_val, y_pred)
    
    # Compute RMSE as percentage of target range
    if target_range:
        range_span = target_range[1] - target_range[0]
        rmse_pct = 100.0 * rmse / range_span
    else:
        rmse_pct = None
    
    # Check uncertainty calibration
    # Ideally: ~68% of samples within ±1σ, ~95% within ±2σ
    errors = np.abs(y_val - y_pred)
    within_1sigma = (errors <= y_std).mean()
    within_2sigma = (errors <= 2 * y_std).mean()
    
    # Check for physics violations (if applicable)
    violations = 0
    if target_name in ["YS", "UTS"]:
        # No specific cross-target violations to check here
        # (Would need paired predictions to check YS≤UTS)
        pass
    
    metrics = {
        "r2_score": float(r2),
        "rmse": float(rmse),
        "rmse_pct": float(rmse_pct) if rmse_pct else None,
        "mae": float(mae),
        "mean_uncertainty": float(y_std.mean()),
        "median_uncertainty": float(np.median(y_std)),
        "uncertainty_calibration": {
            "within_1sigma": float(within_1sigma),
            "within_2sigma": float(within_2sigma)
        },
        "n_validation_samples": len(X_val),
        "target_range_train": target_range if target_range else None
    }
    
    # Report
    print(f"  R² score: {r2:.4f}")
    print(f"  RMSE: {rmse:.2f}" + (f" ({rmse_pct:.1f}% of range)" if rmse_pct else ""))
    print(f"  MAE: {mae:.2f}")
    print(f"  Mean uncertainty (σ): {y_std.mean():.2f}")
    print(f"  Uncertainty calibration:")
    print(f"    Within ±1σ: {within_1sigma:.1%} (expect ~68%)")
    print(f"    Within ±2σ: {within_2sigma:.1%} (expect ~95%)")
    
    # Warnings
    if r2 < 0.7:
        print(f"  ⚠️  WARNING: Low R² ({r2:.3f}) - model may not be learning well")
    if within_1sigma < 0.5 or within_1sigma > 0.85:
        print(f"  ⚠️  WARNING: Poor uncertainty calibration ({within_1sigma:.1%})")
    
    return metrics


def plot_predictions(y_true, y_pred, y_std, target_name, save_path):
    """
    Create prediction plots: scatter, residuals, uncertainty.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # 1. Predicted vs. True
    ax = axes[0]
    ax.scatter(y_true, y_pred, alpha=0.5, s=20)
    ax.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2, label='Perfect')
    ax.set_xlabel(f'True {target_name}')
    ax.set_ylabel(f'Predicted {target_name}')
    ax.set_title(f'{target_name}: Predicted vs. True')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # 2. Residuals
    ax = axes[1]
    residuals = y_true - y_pred
    ax.scatter(y_pred, residuals, alpha=0.5, s=20)
    ax.axhline(0, color='r', linestyle='--', lw=2)
    ax.set_xlabel(f'Predicted {target_name}')
    ax.set_ylabel('Residual (True - Pred)')
    ax.set_title(f'{target_name}: Residuals')
    ax.grid(alpha=0.3)
    
    # 3. Uncertainty
    ax = axes[2]
    errors = np.abs(residuals)
    ax.scatter(y_std, errors, alpha=0.5, s=20)
    ax.plot([0, y_std.max()], [0, y_std.max()], 'r--', lw=2, label='Perfect calibration')
    ax.set_xlabel('Predicted Uncertainty (σ)')
    ax.set_ylabel('Actual Error |True - Pred|')
    ax.set_title(f'{target_name}: Uncertainty Calibration')
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved plot to {save_path}")


# --------------------------------------------------------------------------------
# MAIN TRAINING LOOP
# --------------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train GP models for alloy design")
    parser.add_argument("--train-dir", default="synth_out", help="Directory with synthetic training data")
    parser.add_argument("--val-dir", default="baseline_out", help="Directory with real validation data")
    parser.add_argument("--out-dir", default="gp_models", help="Output directory for trained models")
    parser.add_argument("--max-train-samples", type=int, default=5000, help="Max training samples (GP memory scales O(N³))")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    train_dir = Path(args.train_dir)
    val_dir = Path(args.val_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("GAUSSIAN PROCESS MODEL TRAINING")
    print("="*80)
    print(f"Training data: {train_dir}")
    print(f"Validation data: {val_dir}")
    print(f"Output directory: {out_dir}")
    print(f"Random seed: {args.seed}")
    
    # Store all results
    all_results = {
        "training_date": datetime.now().isoformat(),
        "input_features": INPUT_FEATURES,
        "n_features": len(INPUT_FEATURES),
        "targets": {},
        "seed": args.seed
    }
    
    # Train model for each target
    for target_key, target_col in TARGETS.items():
        print("\n" + "="*80)
        print(f"TARGET: {target_key} ({target_col})")
        print("="*80)
        
        # Load training data (synthetic Jitter)
        train_file = train_dir / target_key / "jitter.csv"
        if not train_file.exists():
            print(f"  ❌ Training file not found: {train_file}")
            continue
        
        df_train = pd.read_csv(train_file, encoding="utf-8-sig")
        print(f"Loaded training data: {len(df_train)} samples")
        
        # Load validation data (real VIEW)
        val_file = val_dir / f"VIEW_{target_key}.csv"
        if not val_file.exists():
            print(f"  ❌ Validation file not found: {val_file}")
            continue
        
        df_val = pd.read_csv(val_file, encoding="utf-8-sig")
        print(f"Loaded validation data: {len(df_val)} samples")
        
        # Extract features and targets
        # Check if all features are present
        missing_train = [f for f in INPUT_FEATURES if f not in df_train.columns]
        missing_val = [f for f in INPUT_FEATURES if f not in df_val.columns]
        
        if missing_train:
            print(f"  ❌ Missing features in training data: {missing_train}")
            continue
        if missing_val:
            print(f"  ❌ Missing features in validation data: {missing_val}")
            continue
        if target_col not in df_train.columns:
            print(f"  ❌ Target column '{target_col}' not in training data")
            continue
        if target_col not in df_val.columns:
            print(f"  ❌ Target column '{target_col}' not in validation data")
            continue
        
        X_train = df_train[INPUT_FEATURES].values
        y_train = df_train[target_col].values
        
        X_val = df_val[INPUT_FEATURES].values
        y_val = df_val[target_col].values
        
        # Check for NaNs
        if np.isnan(X_train).any() or np.isnan(y_train).any():
            print(f"  ⚠️  WARNING: NaNs detected in training data, dropping...")
            mask = ~(np.isnan(X_train).any(axis=1) | np.isnan(y_train))
            X_train = X_train[mask]
            y_train = y_train[mask]
            print(f"  Remaining training samples: {len(X_train)}")
        
        if np.isnan(X_val).any() or np.isnan(y_val).any():
            print(f"  ⚠️  WARNING: NaNs detected in validation data, dropping...")
            mask = ~(np.isnan(X_val).any(axis=1) | np.isnan(y_val))
            X_val = X_val[mask]
            y_val = y_val[mask]
            print(f"  Remaining validation samples: {len(X_val)}")
        
        # Subsample training data if too large (GP memory scales O(N³))
        if len(X_train) > args.max_train_samples:
            print(f"  ⚠️  Subsampling training data: {len(X_train)} → {args.max_train_samples}")
            indices = np.random.RandomState(args.seed).choice(
                len(X_train), size=args.max_train_samples, replace=False
            )
            X_train = X_train[indices]
            y_train = y_train[indices]
        
        # Standardize features (important for GP with ARD)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        
        # Train GP
        gp = train_gp_model(X_train_scaled, y_train, target_key, seed=args.seed)
        
        # Validate
        target_range = (y_train.min(), y_train.max())
        metrics = validate_gp_model(gp, X_val, y_val, scaler, target_key, target_range)
        
        # Plot predictions
        X_val_scaled = scaler.transform(X_val)
        y_pred, y_std = gp.predict(X_val_scaled, return_std=True)
        
        plot_path = out_dir / f"{target_key}_validation_plots.png"
        plot_predictions(y_val, y_pred, y_std, target_key, plot_path)
        
        # Save model
        model_data = {
            "gp_model": gp,
            "scaler": scaler,
            "input_features": INPUT_FEATURES,
            "target_column": target_col,
            "target_key": target_key,
            "training_samples": len(X_train),
            "validation_metrics": metrics,
            "training_date": datetime.now().isoformat(),
            "seed": args.seed
        }
        
        model_path = out_dir / f"gp_model_{target_key}.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(model_data, f)
        print(f"  ✓ Saved model to {model_path}")
        
        # Store results
        all_results["targets"][target_key] = {
            "target_column": target_col,
            "training_samples": len(X_train),
            "validation_samples": len(X_val),
            "validation_metrics": metrics,
            "model_file": str(model_path),
            "plot_file": str(plot_path)
        }
    
    # Save summary report
    report_path = out_dir / "training_report.json"
    with open(report_path, "w") as f:
        json.dump(all_results, f, indent=2)
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE!")
    print("="*80)
    print(f"\nTrained models: {len(all_results['targets'])}/5")
    print(f"Summary report: {report_path}")
    print("\nValidation Performance Summary:")
    print("-" * 80)
    print(f"{'Target':<12} {'R²':<8} {'RMSE':<10} {'MAE':<10} {'1σ Calib':<12}")
    print("-" * 80)
    
    for target_key, results in all_results["targets"].items():
        metrics = results["validation_metrics"]
        r2 = metrics["r2_score"]
        rmse = metrics["rmse"]
        mae = metrics["mae"]
        calib = metrics["uncertainty_calibration"]["within_1sigma"]
        
        print(f"{target_key:<12} {r2:<8.4f} {rmse:<10.2f} {mae:<10.2f} {calib:<12.1%}")
    
    print("-" * 80)
    print("\nNext steps:")
    print("  1. Review validation plots in gp_models/")
    print("  2. Check training_report.json for detailed metrics")
    print("  3. If R² > 0.8 and uncertainty well-calibrated → proceed to inverse design")
    print("  4. If R² < 0.7 → may need feature engineering or more data")


if __name__ == "__main__":
    main()

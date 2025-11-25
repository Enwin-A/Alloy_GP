# ALLOY DISCOVERY WORKFLOW
# ========================
#
# Goal: Train GP models that can predict properties for NOVEL alloy compositions
#       and guide discovery of new alloys with target properties.
#
# The key insight is that we need models that understand the SPACE BETWEEN
# known alloys, not just memorize existing compositions.

# RECOMMENDED APPROACH: Hybrid Strategy
# =====================================
#
# 1. Use MIXUP for training (explores interpolated space)
# 2. Use REAL data for final validation (honest generalization estimate)  
# 3. Train on smaller but more diverse dataset (3000-5000 samples)
#
# This gives you:
#   - Models that generalize to novel compositions
#   - Honest uncertainty estimates for exploration
#   - Faster training (O(N³) scales better with smaller N)

Write-Host "========================================"
Write-Host "ALLOY DISCOVERY TRAINING PIPELINE"
Write-Host "========================================"
Write-Host ""

# Step 1: Generate Mixup Data (creates diverse interpolated compositions)
Write-Host "[Step 1/3] Generating MIXUP synthetic data..."
Write-Host "  This creates NEW compositions by interpolating between known alloys."
Write-Host ""

$targets = @("YS", "UTS", "FractureEL", "UniformEL", "YPE")
$samples = @{
    "YS" = 5000
    "UTS" = 5000
    "FractureEL" = 5000
    "UniformEL" = 2000   # Sparse data
    "YPE" = 4000
}

foreach ($t in $targets) {
    Write-Host "  Generating $t..."
    python generate_mixup.py --target $t --n-samples $samples[$t] --seed 42
    if ($LASTEXITCODE -ne 0) {
        Write-Host "  ERROR: Failed to generate $t" -ForegroundColor Red
        exit 1
    }
}

Write-Host ""
Write-Host "[Step 2/3] Training GP models with mixup data..."
Write-Host "  Training with k-fold cross-validation and real data validation."
Write-Host ""

python train_gp_mixup.py --train-type mixup --n-samples 3500 --n-folds 5

if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Training failed" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "[Step 3/3] Done!"
Write-Host "========================================"
Write-Host ""
Write-Host "Models saved to: gp_models/"
Write-Host ""
Write-Host "Next steps for alloy discovery:"
Write-Host "  1. Load models with pickle.load()"
Write-Host "  2. Define target properties (e.g., YS > 300 MPa)"  
Write-Host "  3. Use optimization (Bayesian, genetic) to search composition space"
Write-Host "  4. GP uncertainty guides exploration vs exploitation"
Write-Host ""

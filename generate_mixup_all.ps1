# generate_mixup_all.ps1 - Generate mixup-augmented data for all targets
# 
# Mixup creates genuinely NEW alloy compositions by interpolating between
# real samples, unlike jitter which just adds noise to existing samples.
#
# This is ideal for ALLOY DISCOVERY because:
#   - Explores composition space BETWEEN known alloys
#   - Creates diverse training data
#   - Maintains physical validity (compositions sum to 100%)
#
# Usage: .\generate_mixup_all.ps1

Write-Host "========================================"
Write-Host "Generating MIXUP Data for All Targets"
Write-Host "========================================"
Write-Host ""
Write-Host "Unlike jitter (which copies+noise existing alloys),"
Write-Host "mixup creates genuinely NEW alloy compositions"
Write-Host "by interpolating between similar real samples."
Write-Host ""

$targets = @(
    @{Name="YS"; Samples=5000; Desc="Yield Strength"},
    @{Name="UTS"; Samples=5000; Desc="Ultimate Tensile Strength"},
    @{Name="FractureEL"; Samples=5000; Desc="Fracture Elongation"},
    @{Name="UniformEL"; Samples=2000; Desc="Uniform Elongation (sparse)"},
    @{Name="YPE"; Samples=4000; Desc="Yield Point Elongation"}
)

$startTime = Get-Date

foreach ($t in $targets) {
    Write-Host ""
    Write-Host "[$($targets.IndexOf($t)+1)/$($targets.Count)] $($t.Name) ($($t.Desc)): GENERATING..."
    Write-Host "-" * 60
    
    python generate_mixup.py --target $t.Name --n-samples $t.Samples --seed 42
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: Failed to generate $($t.Name)" -ForegroundColor Red
    } else {
        Write-Host "SUCCESS: $($t.Name) complete" -ForegroundColor Green
    }
}

$elapsed = (Get-Date) - $startTime

Write-Host ""
Write-Host "========================================"
Write-Host "ALL TARGETS COMPLETE!" -ForegroundColor Green
Write-Host "========================================"
Write-Host ""
Write-Host "Generated mixup datasets:"
foreach ($t in $targets) {
    Write-Host "  - synth_out/$($t.Name)/mixup.csv"
}
Write-Host ""
Write-Host "Time elapsed: $($elapsed.Minutes)m $($elapsed.Seconds)s"
Write-Host ""
Write-Host "Next step: Train GP models with mixup data"
Write-Host "  python train_gp_mixup.py"

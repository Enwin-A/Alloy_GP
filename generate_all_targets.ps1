# Generate synthetic data for all 5 targets
# Usage: .\generate_all_targets.ps1

$ErrorActionPreference = "Stop"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Generating Synthetic Data for All Targets" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

# Common chemistry windows
$chemWindow = "Mg<=6,Cu<=0.1,Mn<=1.5,Cr<=0.25,Fe<=0.5,Si<=0.5,Zn<=0.25,Ti<=0.15,Zr<=0.15,Sc<=0.3,Ni<=0.1,Other<=0.15"

# Target 1: YS
Write-Host "`n[1/5] YS (Yield Strength): GENERATING..." -ForegroundColor Yellow
python generate_synthetics_enhanced.py `
  --view baseline_out/VIEW_YS.csv `
  --out-dir synth_out/YS `
  --target-col YS `
  --group-cols "file_name,alloy,card" `
  --n-per-class 400 `
  --chem-window $chemWindow `
  --seed 42

if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: YS generation failed!" -ForegroundColor Red
    exit 1
}

# Target 2: UTS
Write-Host "`n[2/5] UTS (Ultimate Tensile Strength): GENERATING..." -ForegroundColor Yellow
python generate_synthetics_enhanced.py `
  --view baseline_out/VIEW_UTS.csv `
  --out-dir synth_out/UTS `
  --target-col UTS `
  --group-cols "file_name,alloy,card" `
  --n-per-class 400 `
  --chem-window $chemWindow `
  --seed 42

if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: UTS generation failed!" -ForegroundColor Red
    exit 1
}

# Target 3: Fracture EL
Write-Host "`n[3/5] Fracture EL: GENERATING..." -ForegroundColor Yellow
python generate_synthetics_enhanced.py `
  --view baseline_out/VIEW_FractureEL.csv `
  --out-dir synth_out/FractureEL `
  --target-col "Fracture EL" `
  --group-cols "file_name,alloy,card" `
  --n-per-class 400 `
  --chem-window $chemWindow `
  --seed 42

if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Fracture EL generation failed!" -ForegroundColor Red
    exit 1
}

# Target 4: Uniform EL (sparse - reduce n_per_class)
Write-Host "`n[4/5] Uniform EL (SPARSE): GENERATING..." -ForegroundColor Yellow
python generate_synthetics_enhanced.py `
  --view baseline_out/VIEW_UniformEL.csv `
  --out-dir synth_out/UniformEL `
  --target-col "Uniform EL" `
  --group-cols "file_name,alloy,card" `
  --n-per-class 200 `
  --chem-window $chemWindow `
  --seed 42

if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Uniform EL generation failed!" -ForegroundColor Red
    exit 1
}

# Target 5: YPE
Write-Host "`n[5/5] YPE (Yield Point Elongation): GENERATING..." -ForegroundColor Yellow
python generate_synthetics_enhanced.py `
  --view baseline_out/VIEW_YPE.csv `
  --out-dir synth_out/YPE `
  --target-col YPE `
  --group-cols "file_name,alloy,card" `
  --n-per-class 400 `
  --chem-window $chemWindow `
  --seed 42

if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: YPE generation failed!" -ForegroundColor Red
    exit 1
}

Write-Host "`n========================================" -ForegroundColor Green
Write-Host "✅ ALL TARGETS COMPLETE!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green

Write-Host "`nGenerated synthetic datasets:"
Write-Host "  1. synth_out/YS/jitter.csv (19,980 samples)"
Write-Host "  2. synth_out/UTS/jitter.csv"
Write-Host "  3. synth_out/FractureEL/jitter.csv"
Write-Host "  4. synth_out/UniformEL/jitter.csv"
Write-Host "  5. synth_out/YPE/jitter.csv"

Write-Host "`nQC reports saved to:"
Write-Host "  synth_out/<TARGET>/generation_report.json"

Write-Host "`nNext step: Phase 3 - GP Model Training"
Write-Host "  See action_plan.txt for instructions"

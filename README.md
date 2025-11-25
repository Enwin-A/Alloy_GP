# Aluminum Alloy Design Assistant

## Project Overview

AI-powered inverse design system for aluminum alloys (AA5xxx family) that:
- Predicts mechanical properties from composition + processing parameters
- Recommends feasible alloy compositions to achieve target properties
- Respects metallurgical constraints and physics

## Quick Start

### 1. Data Cleaning (Completed ✓)

```bash
python og_cleanup_and_views.py \
  --input OG_dataset_cards_all_one_row_cleaned.csv \
  --out-dir baseline_out
```

**Outputs:**
- `BASE_MASTER.csv` - Truth set (333 rows, all features populated)
- `VIEW_YS.csv`, `VIEW_UTS.csv`, etc. - Per-target label-complete slices
- `cleaning_report.json` - Audit trail

### 2. Synthetic Data Generation (Current Phase)

#### ✅ YS (Yield Strength) - COMPLETE

**Status**: PASS ✓  
**Recommended Method**: **Jitter**  
**Output**: `synth_out/YS/jitter.csv` (19,980 samples)

**QC Summary**:
- KS test p-value: 1.0 (perfect)
- Correlation MAE: 0.0006 (excellent)
- Composition: 100.0000 ± 0.0000%
- Physics violations: 0

**Why Jitter won**: Perfect statistical fidelity, preserves feature correlations, respects physics constraints. CTGAN/GaussianCopula failed due to poor distribution matching after dropping high-cardinality group columns.

---

#### Example: Generate synthetics for Yield Strength

```bash
python generate_synthetics_enhanced.py \
  --view baseline_out/VIEW_YS.csv \
  --out-dir synth_out/YS \
  --target-col YS \
  --group-cols "file_name,alloy,card" \
  --n-per-class 400 \
  --chem-window "Mg<=6,Cu<=0.1,Mn<=1.5,Cr<=0.25,Fe<=0.5,Si<=0.5,Zn<=0.25,Ti<=0.15,Zr<=0.15,Sc<=0.3,Ni<=0.1,Other<=0.15" \
  --seed 42
```

**Outputs:**
- `synth_out/YS/jitter.csv` - Jitter-based synthetics
- `synth_out/YS/ctgan.csv` - CTGAN-based synthetics
- `synth_out/YS/gaussiancopula.csv` - Gaussian Copula synthetics
- `synth_out/YS/generation_report.json` - Comprehensive QC metrics

#### Repeat for all targets:

```bash
# UTS (Ultimate Tensile Strength)
python generate_synthetics_enhanced.py --view baseline_out/VIEW_UTS.csv --out-dir synth_out/UTS --target-col UTS ...

# Fracture Elongation
python generate_synthetics_enhanced.py --view baseline_out/VIEW_FractureEL.csv --out-dir synth_out/FractureEL --target-col "Fracture EL" ...

# Uniform Elongation (sparse - prefer Jitter)
python generate_synthetics_enhanced.py --view baseline_out/VIEW_UniformEL.csv --out-dir synth_out/UniformEL --target-col "Uniform EL" --n-per-class 200 ...

# Yield Point Elongation
python generate_synthetics_enhanced.py --view baseline_out/VIEW_YPE.csv --out-dir synth_out/YPE --target-col YPE ...
```

### 3. Review QC Reports

Check `generation_report.json` for each target:

```python
import json
with open("synth_out/YS/generation_report.json") as f:
    report = json.load(f)

# Check recommendation
print(f"Recommended generator: {report['recommendation']}")

# Check acceptance per generator
for method in ["jitter", "ctgan", "gaussiancopula"]:
    if method in report["generators"]:
        acc = report["generators"][method]["acceptance"]
        print(f"{method}: {'PASS' if acc['overall_pass'] else 'FAIL'}")
```

**Acceptance Criteria:**
- Composition Σ within [99.95, 100.05]
- Zero negative values
- Zero chemistry window violations
- ≥95% of synthetics within real target range
- KS test p-value ≥ 0.01
- Class balance ratio ≤ 2.5
- Correlation MAE ≤ 0.12
- Near-duplicates ≤ 3%
- Clip rate ≤ 5%

### 4. Select Best Synthetic Sets

Create `synth_out/selected_datasets.json` documenting chosen method per target:

```json
{
  "YS": {
    "chosen_method": "ctgan",
    "file": "synth_out/YS/ctgan.csv",
    "rows": 7845,
    "qc_summary": {
      "composition_pass": true,
      "ks_median_p_value": 0.42,
      "correlation_mae": 0.08
    }
  },
  "UTS": {...},
  "FractureEL": {...},
  "UniformEL": {
    "chosen_method": "jitter",
    "note": "Preferred Jitter over CTGAN due to sparsity (64 real rows)"
  },
  "YPE": {...}
}
```

## Data Architecture

### BASE_MASTER.csv (Truth Set)
- **Rows:** 333 unique alloy samples
- **Columns:** 66
  - **Composition (wt%):** Si, Fe, Cu, Mn, Mg, Cr, Ni, Zn, Ti, Zr, Sc, Other, Al
  - **Process (standardized):**
    - `Homogenization (Celsius/Seconds)`: e.g., "540/18000 + 490/86400"
    - `Recrystallization annealing (Celsius/Seconds)`: e.g., "350/7200"
  - **Derived Numeric Features:**
    - Homog: `homog_time_total_s`, `homog_temp_max_C`, `homog_T_time_weighted_C`, `homog_time_at_ge_520C_s`, `adequate_homog`
    - Recryst: `recryst_type`, `recryst_time_total_s`, `recryst_temp_max_C`, `recryst_T_time_weighted_C`, `adequate_recryst`
  - **Microstructure:** Ingot thickness (mm), grain size (µm), rolling reductions (%)
  - **Targets:** YS (MPa), UTS (MPa), Fracture EL (%), Uniform EL (%), YPE (%)
  - **Audit:** `*_RAW` columns, `comp_sum_before/after`, `comp_fix_applied`, `flag_missing_*`

### VIEW Files (Per-Target Slices)
- `VIEW_YS.csv` (333 rows) - All samples with YS labels
- `VIEW_UTS.csv` (332 rows)
- `VIEW_FractureEL.csv` (329 rows)
- `VIEW_UniformEL.csv` (64 rows) - **SPARSE**
- `VIEW_YPE.csv` (210 rows)

## Composition Constraints (AA5xxx)

**Chemistry Windows (wt%):**
- Mg ≤ 6.0 (primary strengthener)
- Cu ≤ 0.1
- Mn ≤ 1.5
- Cr ≤ 0.25
- Fe ≤ 0.5
- Si ≤ 0.5
- Zn ≤ 0.25
- Ti ≤ 0.15
- Zr ≤ 0.15
- Sc ≤ 0.30
- Ni ≤ 0.10
- Other ≤ 0.15

**Hard Constraints:**
- All elements ≥ 0 (non-negativity)
- Σ(all elements) = 100.0% ± 0.05%
- Al = 100 - Σ(other elements) (balance)

## Physics Gates

Enforced in synthetic generation:
1. **YS ≤ UTS** (yield strength cannot exceed ultimate strength)
2. **Uniform EL ≤ Fracture EL** (uniform elongation subset of total elongation)
3. **Target within real range ± 5% padding** (prevent extreme extrapolation)

## Next Steps

### Phase 3: Forward Model Training
- Train Genetic Programming (GP) models per property
- Use recommended synthetic datasets from Phase 2
- Validate on real data (VIEW files)
- Export interpretable equations + pickled models

**Recommended Library:** PySR (Python wrapper for SymbolicRegression.jl)

### Phase 4: Inverse Design Engine
- Multi-objective optimization (NSGA-II) with trained forward models
- Given target properties → recommend feasible compositions
- Respect chemistry windows + physics constraints
- Output ranked candidates with cost estimates

### Phase 5: Deployment
- CLI tool: `design_assistant.py`
- Web UI: Streamlit app
- API: FastAPI endpoint

## File Structure

```
new_approach/
├── og_cleanup_and_views.py          # Data cleaning script (Phase 1) ✓
├── generate_synthetics_enhanced.py   # Synthetic generation (Phase 2)
├── action_plan.txt                   # Complete project roadmap
├── README.md                         # This file
├── OG_dataset_cards_all_one_row_cleaned.csv
├── baseline_out/                     # Cleaned real data ✓
│   ├── BASE_MASTER.csv
│   ├── VIEW_YS.csv
│   ├── VIEW_UTS.csv
│   ├── VIEW_FractureEL.csv
│   ├── VIEW_UniformEL.csv
│   ├── VIEW_YPE.csv
│   ├── coverage_*.csv (5 files)
│   ├── labels_needed.csv
│   └── cleaning_report.json
├── synth_out/                        # Synthetic datasets (Phase 2 - pending)
│   ├── YS/
│   │   ├── jitter.csv
│   │   ├── ctgan.csv
│   │   ├── gaussiancopula.csv
│   │   └── generation_report.json
│   ├── UTS/ (same structure)
│   ├── FractureEL/ (same structure)
│   ├── UniformEL/ (same structure)
│   ├── YPE/ (same structure)
│   └── selected_datasets.json
├── models/                           # Forward GP models (Phase 3 - pending)
│   ├── YS_model.pkl
│   ├── YS_equation.txt
│   ├── YS_validation_report.json
│   ├── (repeat for UTS, FractureEL, UniformEL, YPE)
│   └── model_comparison.csv
└── inverse_design/                   # Inverse design tool (Phase 4 - pending)
    ├── optimization_engine.py
    ├── pareto_candidates.csv
    └── recommendations.csv
```

## Troubleshooting

### Synthetic Generation Fails

**CTGAN fails with "mode collapse":**
- Reduce epochs: `--epochs 200` (add to script if needed)
- Increase PAC: modify `pac=4` in script
- Fallback to Jitter or Gaussian Copula

**Jitter produces too similar samples:**
- Reduce MAD multiplier from 0.5 to 0.3 in code
- Increase `n_per_class` for more variety

**Acceptance criteria fail:**
- Check `generation_report.json` for specific violations
- If KS p-value low: increase synthetic sample size
- If correlation MAE high: use Jitter (more conservative)
- If class imbalance: check real data distribution, may need stratified sampling

### Sparse Targets (UniformEL)

With only 64 real samples:
1. **Prefer Jitter** (stays closest to real data)
2. **Reduce n_per_class** to 200 (avoid over-generation)
3. **Consider GaussianCopula** as alternative (stable with small data)
4. **Avoid CTGAN** (needs >100 samples for stable training)

## Contact & Support

See `action_plan.txt` for:
- Detailed phase-by-phase instructions
- Acceptance criteria per phase
- Model training hyperparameters
- Inverse design optimization setup
- Deployment strategies

## License

Internal research project - not for public distribution.

## Version

- **Phase 1 (Data Cleaning):** COMPLETE ✓
- **Phase 2 (Synthetic Generation):** IN PROGRESS
- **Last Updated:** 2025

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_synthetics.py — build synthetic datasets (jitter, CTGAN, GaussianCopula)
for a single target VIEW_<T>.csv produced by og_cleanup_and_views.py.

Usage example:
  python generate_synthetics.py \
    --view baseline_out/VIEW_YS.csv \
    --out-dir synth_out/YS \
    --group-cols "Class1,Class2,Class3" \
    --n-per-class 400 \
    --chem-window "Mg<=6,Cu<=0.1" \
    --seed 42
"""

import argparse, json, re
from pathlib import Path
import numpy as np
import pandas as pd

from sdv.tabular import CTGAN, GaussianCopula
from sdv.metadata import SingleTableMetadata

ELEMENTS = ["Si","Fe","Cu","Mn","Mg","Cr","Ni","Zn","Ti","Zr","Sc","Other","Al"]

# ---------- time parsing helpers ----------
def _parse_T_over_units_to_seconds(s: str, assumed_unit: str) -> list:
    # "540/300 + 490/1440" → [(540, 300*60), (490, 1440*60)] if unit="minutes"
    if not isinstance(s, str) or not s.strip():
        return []
    parts = [p.strip() for p in s.split("+")]
    out = []
    for p in parts:
        m = re.match(r"^\s*(\d{2,3}(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)\s*$", p)
        if not m:
            continue
        T = float(m.group(1))
        v = float(m.group(2))
        t_s = int(round(v * 60.0)) if assumed_unit.lower().startswith("min") else int(round(v))
        out.append((T, t_s))
    return out

def _add_numeric_thermo_features(df: pd.DataFrame) -> pd.DataFrame:
    # Homogenization
    hom_cols = [c for c in df.columns if "homogenization (celsius/" in c.lower()]
    for hc in hom_cols:
        unit = "minutes" if hc.lower().endswith("minutes)") else "seconds"
        segs = df[hc].fillna("").astype(str).apply(lambda s: _parse_T_over_units_to_seconds(s, unit))
        ttot = segs.apply(lambda L: sum(t for _, t in L)).astype(int)
        t520 = segs.apply(lambda L: sum(t for T, t in L if T >= 520)).astype(int)
        Tmax = segs.apply(lambda L: (max(T for T, _ in L) if L else np.nan))
        Ttw = segs.apply(lambda L: (sum(T * t for T, t in L) / sum(t for _, t in L) if L and sum(t for _, t in L) > 0 else np.nan))
        df["homog_time_total_s"] = ttot
        df["homog_time_at_ge_520C_s"] = t520
        df["homog_temp_max_C"] = Tmax
        df["homog_T_time_weighted_C"] = Ttw

    # Recrystallization
    rec_cols = [c for c in df.columns if "recrystallization annealing (celsius/" in c.lower()]
    for rc in rec_cols:
        unit = "minutes" if rc.lower().endswith("minutes)") else "seconds"
        segs = df[rc].fillna("").astype(str).apply(lambda s: _parse_T_over_units_to_seconds(s, unit))
        ttot = segs.apply(lambda L: sum(t for _, t in L)).astype(int)
        Tmax = segs.apply(lambda L: (max(T for T, _ in L) if L else np.nan))
        Ttw = segs.apply(lambda L: (sum(T * t for T, t in L) / sum(t for _, t in L) if L and sum(t for _, t in L) > 0 else np.nan))
        df["recryst_time_total_s"] = ttot
        df["recryst_temp_max_C"] = Tmax
        df["recryst_T_time_weighted_C"] = Ttw
    return df

# ---------- chemistry helpers ----------
def parse_chem_window(text):
    out = {}
    if not text: return out
    for tok in text.split(","):
        tok = tok.strip()
        m = re.match(r"([A-Za-z][A-Za-z0-9]*)\s*(<=|<|>=|>|==)\s*([0-9.]+)", tok)
        if m:
            el, op, val = m.group(1), m.group(2), float(m.group(3))
            out[el] = (op, val)
    return out

def apply_chem_bounds(row, bounds):
    for el, (op,val) in bounds.items():
        if el in row:
            v = row[el]
            if pd.isna(v): continue
            if op in ("<","<=") and v > val: row[el] = val
            if op in (">",">=") and v < val: row[el] = val
    return row

def al_as_balance(df):
    adds = df[[c for c in ELEMENTS if c != "Al" and c in df.columns]].fillna(0).sum(axis=1)
    df["Al"] = (100.0 - adds).clip(lower=0)
    comp_cols = [c for c in ELEMENTS if c in df.columns]
    tot = df[comp_cols].sum(axis=1)
    ok = (tot > 0)
    df.loc[ok, comp_cols] = df.loc[ok, comp_cols].div(tot[ok], axis=0)*100.0
    return df

# ---------- models ----------
def make_metadata(df, group_cols):
    md = SingleTableMetadata()
    md.detect_from_dataframe(df)
    for c in group_cols:
        if c in df.columns:
            md.update_column(c, sdtype="categorical")
    for e in ELEMENTS:
        if e in df.columns:
            md.update_column(e, sdtype="numerical")
    return md

def train_ctgan(df, group_cols):
    md = make_metadata(df, group_cols)
    return CTGAN(
        epochs=300, batch_size=256, verbose=True, cuda=False,
        generator_dim=(256,256), discriminator_dim=(256,256), pac=2
    ).fit(df, metadata=md) or _

def train_gaussiancopula(df, group_cols):
    md = make_metadata(df, group_cols)
    model = GaussianCopula(default_distribution="norm", categorical_transformer="one_hot")
    model.fit(df, metadata=md)
    return model

def conditional_sample(model, df_like, group_cols, n_per_class):
    rows = []
    for keys, grp in df_like.groupby(group_cols, dropna=False):
        cond = {col: (None if pd.isna(val) else val) for col, val in zip(group_cols, (keys if isinstance(keys, tuple) else (keys,)))}
        n = int(n_per_class)
        try:
            samp = model.sample(num_rows=n, conditions=cond)
        except Exception:
            samp = model.sample(num_rows=n)
        samp["is_synthetic"] = 1
        rows.append(samp)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

def enforce_composition(df, chem_bounds):
    for e in ELEMENTS:
        if e in df.columns:
            df[e] = pd.to_numeric(df[e], errors="coerce").clip(lower=0)
    for i in range(len(df)):
        df.iloc[i] = apply_chem_bounds(df.iloc[i], chem_bounds)
    return al_as_balance(df)

# ---------- jitter ----------
def jitter_augment(df, group_cols, n_per_class, chem_bounds, seed=42):
    rng = np.random.default_rng(seed)
    comp_cols = [c for c in ELEMENTS if c in df.columns]
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    out = []
    for keys, grp in df.groupby(group_cols, dropna=False):
        n_src = len(grp)
        if n_src == 0: continue
        reps = int(n_per_class)
        base = grp.sample(n=reps, replace=True, random_state=seed).reset_index(drop=True)
        # robust numeric jitter
        mad = grp[num_cols].mad().replace(0, grp[num_cols].std(ddof=1)*0.2).fillna(0.01)
        J = base[num_cols].copy()
        for c in num_cols:
            J[c] = base[c] + rng.normal(0.0, 0.5*float(mad.get(c, 0.01)), size=reps)
        synth = base.copy(); synth[num_cols] = J[num_cols]
        # clamp to observed ranges
        for c in num_cols:
            lo, hi = float(df[c].min()), float(df[c].max())
            synth[c] = synth[c].clip(lo, hi)
        # enforce chemistry & balance
        for c in comp_cols: synth[c] = synth[c].clip(lower=0)
        synth = enforce_composition(synth, chem_bounds)
        synth["is_synthetic"] = 1
        out.append(synth)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=df.columns.tolist()+["is_synthetic"])

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--view", required=True, help="VIEW_<TARGET>.csv")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--group-cols", default="", help="comma-separated class/group cols")
    ap.add_argument("--n-per-class", type=int, default=400)
    ap.add_argument("--chem-window", default="Mg<=6,Cu<=0.1")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    outp = Path(args.out_dir); outp.mkdir(parents=True, exist_ok=True)
    group_cols = [c.strip() for c in args.group_cols.split(",") if c.strip()]
    chem_bounds = parse_chem_window(args.chem_window)

    df = pd.read_csv(args.view, encoding="utf-8-sig")

    # NEW: derive numeric thermo/process features from either Minutes or Seconds strings
    df = _add_numeric_thermo_features(df)

    # Identify target
    target_cols = [c for c in df.columns if ("(MPa)" in c or "(percentage)" in c) and not c.startswith("flag_missing")]
    assert len(target_cols) == 1, f"Ambiguous target columns: {target_cols}"
    target = target_cols[0]

    # Validate group cols
    for c in group_cols:
        if c not in df.columns:
            raise ValueError(f"group col '{c}' not in dataframe")

    # ---------------- JITTER ----------------
    print("Generating jitter synthetic...")
    jitter = jitter_augment(df, group_cols, args.n_per_class, chem_bounds, seed=args.seed)
    if not jitter.empty:
        jitter = enforce_composition(jitter, chem_bounds)
        jitter[target] = jitter[target].clip(lower=df[target].min(), upper=df[target].max())
        jitter.to_csv(outp/"jitter.csv", index=False)

    # ---------------- CTGAN ----------------
    print("Training CTGAN...")
    ctgan = CTGAN(epochs=300, batch_size=256, verbose=True, cuda=False,
                  generator_dim=(256,256), discriminator_dim=(256,256), pac=2)
    from sdv.metadata import SingleTableMetadata
    md = SingleTableMetadata(); md.detect_from_dataframe(df)
    for c in group_cols: md.update_column(c, sdtype="categorical")
    for e in ELEMENTS:
        if e in df.columns: md.update_column(e, sdtype="numerical")
    ctgan.fit(df, metadata=md)
    print("Sampling CTGAN conditionally...")
    ctg = conditional_sample(ctgan, df, group_cols, args.n_per_class)
    if not ctg.empty:
        ctg = enforce_composition(ctg, chem_bounds)
        ctg.to_csv(outp/"ctgan.csv", index=False)

    # ---------------- GaussianCopula ----------------
    print("Training GaussianCopula...")
    from sdv.tabular import GaussianCopula
    gc = GaussianCopula(default_distribution="norm", categorical_transformer="one_hot")
    gc.fit(df, metadata=md)
    print("Sampling GaussianCopula conditionally...")
    gcs = conditional_sample(gc, df, group_cols, args.n_per_class)
    if not gcs.empty:
        gcs = enforce_composition(gcs, chem_bounds)
        gcs.to_csv(outp/"gaussiancopula.csv", index=False)

    # QC summary
    summary = {
        "view": args.view,
        "out_dir": str(outp),
        "group_cols": group_cols,
        "n_per_class": args.n_per_class,
        "chem_window": args.chem_window,
        "rows": {
            "source_view": int(len(df)),
            "jitter": int(len(jitter)) if isinstance(jitter, pd.DataFrame) else 0,
            "ctgan": int(len(ctg)) if isinstance(ctg, pd.DataFrame) else 0,
            "gaussiancopula": int(len(gcs)) if isinstance(gcs, pd.DataFrame) else 0,
        }
    }
    (outp/"synth_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

if __name__ == "__main__":
    main()

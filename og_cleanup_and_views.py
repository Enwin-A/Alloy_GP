#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
og_cleanup_and_views.py — Clean OG dataset; standardize units and export BASE_MASTER + VIEW_Ts.

Conventions:
  • Length: mm
  • Temperature: °C
  • Time: seconds (s)

Created columns (canonical strings):
  • Homogenization (Celsius/Seconds)              e.g., "540/18000 + 490/86400"
  • Recrystallization annealing (Celsius/Seconds) e.g., "350/7200"

Created numeric features:
  • homog_steps, homog_time_total_s, homog_temp_max_C, homog_T_time_weighted_C,
    homog_time_at_ge_520C_s, adequate_homog
  • recryst_type, recryst_steps, recryst_time_total_s, recryst_temp_max_C,
    recryst_T_time_weighted_C, adequate_recryst, adequate_recryst_batch, adequate_recryst_flash

Keeps RAW text in Homogenization_RAW and Recrystallization annealing_RAW for audit.
"""

import argparse, json, re, csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ---------- Regex ----------
# Basic tokens
TEMP_PATTERN = re.compile(r"(\d{2,3}(?:\.\d+)?)\s*°?\s*C", re.I)
HOUR_PATTERN = re.compile(r"(\d+(?:\.\d+)?)\s*(?:hours?|hrs?|h)\b", re.I)
MIN_PATTERN  = re.compile(r"(\d+(?:\.\d+)?)\s*(?:minutes?|mins?|min|m)\b", re.I)
SEC_PATTERN  = re.compile(r"(\d+(?:\.\d+)?)\s*(?:seconds?|secs?|sec|s)\b", re.I)

# e.g., "2x 3 hours"
MULTIPLIER_PATTERN = re.compile(r"(\d+(?:\.\d+)?)\s*[x×]\s*(\d+(?:\.\d+)?)\s*(hours?|hrs?|h)\b", re.I)

# e.g., "5 - 6 hours" or "5–6 min" -> average
RANGE_HOUR_PATTERN = re.compile(r"(\d+(?:\.\d+)?)\s*[-–—]\s*(\d+(?:\.\d+)?)\s*(hours?|hrs?|h)\b", re.I)
RANGE_MIN_PATTERN  = re.compile(r"(\d+(?:\.\d+)?)\s*[-–—]\s*(\d+(?:\.\d+)?)\s*(minutes?|mins?|min|m)\b", re.I)

# General segment: "T °C .... v unit"
SEGMENT_PATTERN = re.compile(
    r"(?P<T>\d{2,3}(?:\.\d+)?)\s*°?\s*C[^0-9]*(?P<v>\d+(?:\.\d+)?)\s*(?P<u>hours?|hrs?|h|minutes?|mins?|min|m|seconds?|secs?|sec|s)\b",
    re.I,
)

ELEMENTS = ["Al","Si","Fe","Cu","Mn","Mg","Cr","Ni","Zn","Ti","Zr","Sc","Other"]
TARGETS  = [
    ("YS (MPa)", "YS"),
    ("UTS (MPa)", "UTS"),
    ("Fracture EL (percentage)", "FractureEL"),
    ("Uniform EL (percentage)", "UniformEL"),
    ("YPE (percentage)", "YPE"),
]

HOMOG_TEXT_PRIORITY = [
    "Homogenization",
    "Homogenization_RAW",
    "Homogenization (Celsius/Minutes)",
    "Homogenization (Celsius/Seconds)",
]
RECRYST_TEXT_PRIORITY = [
    "Recrystallization annealing",
    "Recrystallization annealing_RAW",
    "Recrystallization annealing (Celsius/Minutes)",
    "Recrystallization annealing (Celsius/Seconds)",
]
AMBIGUOUS_HOMOG_COLUMNS = ["Homogenization", "Homogenization (Celsius/Minutes)"]
AMBIGUOUS_RECRYST_COLUMNS = ["Recrystallization annealing", "Recrystallization annealing (Celsius/Minutes)"]

CANONICAL_PART_PATTERN = re.compile(r"\s*(?P<T>\d{2,3}(?:\.\d+)?)\s*/\s*(?P<D>\d+(?:\.\d+)?)\s*")

# ---------- IO helpers ----------
def _read_csv_robust(path: str) -> pd.DataFrame:
    for enc in ("utf-8", "utf-8-sig", "latin1"):
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            continue
    return pd.read_csv(path)

def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    # Excel-friendly, keeps °C intact
    frame.to_csv(path, index=False, encoding="utf-8-sig", quoting=csv.QUOTE_MINIMAL)

# ---------- Text helpers ----------
def clean_visible_artifacts(s: str) -> str:
    if s is None or (isinstance(s, float) and np.isnan(s)): return ""
    out = str(s)
    # Fix encoding, normalize separators that confuse parsing
    out = (out.replace("Â°", "°")
              .replace("�C", "°C")
              .replace(">", " ")        # make "540°C > 5 hours" parseable
              .replace(",", " ")        # commas to spaces
              .replace(";", " ")
           )
    out = out.strip()
    if out.lower() in {"", "nan", "none"}:
        return ""
    return out

def expand_multipliers(text: str) -> str:
    if not text: return ""
    return MULTIPLIER_PATTERN.sub(lambda m: f"{float(m.group(1))*float(m.group(2))} {m.group(3)}", text)

def expand_ranges(text: str) -> str:
    # Replace "5–6 hours" -> "5.5 hours"; same for minutes
    def _avg_hours(m):
        a, b, u = float(m.group(1)), float(m.group(2)), m.group(3)
        return f"{(a+b)/2.0} {u}"
    def _avg_minutes(m):
        a, b, u = float(m.group(1)), float(m.group(2)), m.group(3)
        return f"{(a+b)/2.0} {u}"

    text = RANGE_HOUR_PATTERN.sub(_avg_hours, text)
    text = RANGE_MIN_PATTERN.sub(_avg_minutes, text)
    return text

def parse_time_to_hours(val, unit) -> float:
    u = unit.lower()
    if u.startswith("h"): return float(val)
    if u.startswith("m"): return float(val)/60.0
    return float(val)/3600.0

def hours_to_seconds(h: float) -> int:
    return int(round(h * 3600.0))

def infer_unit_hint(col_name: Optional[str]) -> str:
    if not col_name:
        return ""
    name = col_name.lower()
    if "minutes" in name:
        return "minutes"
    if "seconds" in name:
        return "seconds"
    return ""

def parse_canonical_segment_text(text: str, unit_hint: str = "") -> Optional[List[Tuple[float, float]]]:
    if not text or "/" not in text:
        return None
    unit = unit_hint or "seconds"
    parts = [p.strip() for p in text.split("+")]
    segments: List[Tuple[float, float]] = []
    for part in parts:
        if not part:
            continue
        m = CANONICAL_PART_PATTERN.fullmatch(part)
        if not m:
            return None
        T = float(m.group("T"))
        duration = float(m.group("D"))
        hours = duration / 60.0 if unit == "minutes" else duration / 3600.0
        segments.append((T, hours))
    return segments

def select_text_source_column(df: pd.DataFrame, candidates: List[str]) -> str:
    for col in candidates:
        if col in df.columns:
            return col
    return ""

# ---------- Segment parsing ----------
def parse_homog_segments(text: str, unit_hint: str = "") -> List[Tuple[float,float]]:
    """Return list of (T_C, time_h) homogenization segments. Filters out low-T nonsense."""
    if text is None or (isinstance(text, float) and np.isnan(text)):
        return []
    s = clean_visible_artifacts(text)
    if not s:
        return []
    canonical = parse_canonical_segment_text(s, unit_hint)
    if canonical is None:
        s = expand_multipliers(expand_ranges(s))
        segs = []
        for m in SEGMENT_PATTERN.finditer(s):
            T = float(m.group("T")); v = float(m.group("v")); u = m.group("u")
            segs.append((T, parse_time_to_hours(v, u)))
    else:
        segs = canonical
    return [(T,h) for (T,h) in segs if T >= 300 and h > 0]

def parse_recryst_segments(text: str, unit_hint: str = "") -> Dict[str, object]:
    """Return {'type', 'segs'=[(T_C, time_h)...], 'direct_exit_T'} for recrystallization/anneal."""
    if text is None or (isinstance(text, float) and np.isnan(text)):
        return {"type":"none","segs":[],"direct_exit_T":np.nan}
    s = clean_visible_artifacts(text)
    if not s or s.lower() in {"n/a","na","nan"}:
        return {"type":"none","segs":[],"direct_exit_T":np.nan}

    canonical = parse_canonical_segment_text(s, unit_hint)
    if canonical is not None:
        segs = [(T,h) for (T,h) in canonical if T >= 200 and h >= 0]
        return {"type":"other","segs":segs,"direct_exit_T":np.nan}

    s = expand_multipliers(expand_ranges(s))
    s_low = s.lower()
    if s_low in {"no","none"} or ("no" in s_low and ("anneal" in s_low or "recryst" in s_low)):
        return {"type":"none","segs":[],"direct_exit_T":np.nan}

    if "flash" in s_low:   rtype = "flash"
    elif "batch" in s_low: rtype = "batch"
    elif "direct" in s_low:rtype = "direct"
    else:                  rtype = "other"

    segs = []
    for m in SEGMENT_PATTERN.finditer(s):
        T = float(m.group("T")); v = float(m.group("v")); u = m.group("u")
        segs.append((T, parse_time_to_hours(v, u)))

    direct_exit_T = np.nan
    if rtype=="direct" and not segs:
        m = TEMP_PATTERN.search(s)
        if m:
            direct_exit_T = float(m.group(1))
            segs = [(direct_exit_T, 0.0)]

    segs = [(T,h) for (T,h) in segs if T >= 200 and h >= 0]
    return {"type": rtype, "segs": segs, "direct_exit_T": direct_exit_T}

# ---------- Summaries (with seconds) ----------
def summarize_homog(segs: List[Tuple[float,float]]) -> Dict[str, float]:
    if not segs:
        return {"homog_steps":0,"homog_time_total_s":0,"homog_temp_max_C":np.nan,
                "homog_T_time_weighted_C":np.nan,"homog_time_at_ge_520C_s":0,"adequate_homog":0}
    ttot_h = sum(h for _,h in segs)
    Tmax = max(T for T,_ in segs)
    Ttw  = sum(T*h for T,h in segs)/ttot_h if ttot_h>0 else np.nan
    t520_h = sum(h for T,h in segs if T>=520)
    adequate = int((Tmax>=520) and (t520_h>=2.0))  # simple adequacy rule
    return {"homog_steps":len(segs),
            "homog_time_total_s": hours_to_seconds(ttot_h),
            "homog_temp_max_C":Tmax,
            "homog_T_time_weighted_C":Ttw,
            "homog_time_at_ge_520C_s": hours_to_seconds(t520_h),
            "adequate_homog":adequate}

def summarize_recryst(rec: Dict[str, object]) -> Dict[str, float]:
    rtype = rec["type"]; segs: List[Tuple[float,float]] = rec["segs"]
    if not segs:
        Tmax, Ttw, ttot_h, t300_h = (np.nan, np.nan, 0.0, 0.0)
    else:
        ttot_h = sum(h for _,h in segs)
        Tmax = max(T for T,_ in segs)
        Ttw  = sum(T*h for T,h in segs)/ttot_h if ttot_h>0 else (segs[0][0] if len(segs)==1 else np.nan)
        t300_h = sum(h for T,h in segs if T>=300)
    adequate_batch = int(rtype=="batch" and (t300_h >= 1.0))
    adequate_flash = int(rtype=="flash" and (t300_h >= 1.0/60.0))
    adequate_any   = int(adequate_batch or adequate_flash)
    out = {
        "recryst_type": rtype,
        "recryst_steps": len(segs),
        "recryst_time_total_s": hours_to_seconds(ttot_h),
        "recryst_temp_max_C": Tmax,
        "recryst_T_time_weighted_C": Ttw,
        "adequate_recryst": adequate_any,
        "adequate_recryst_batch": adequate_batch,
        "adequate_recryst_flash": adequate_flash,
    }
    if rtype=="direct" and pd.isna(Tmax) and not pd.isna(rec.get("direct_exit_T", np.nan)):
        out["recryst_temp_max_C"] = float(rec["direct_exit_T"])
        out["recryst_T_time_weighted_C"] = float(rec["direct_exit_T"])
    return out

# ---------- Composition handling (auditable) ----------
def fix_composition_with_audit(df: pd.DataFrame, tol_small=0.5, tol_tiny=0.05) -> Dict[str,int]:
    rep = {"rows_al_balance_applied":0,"rows_full_comp_renorm":0,"rows_negative_clipped":0,"rows_adds_gt_100":0}
    comp_cols = [c for c in ELEMENTS if c in df.columns]
    if not comp_cols: return rep

    # raw snapshots
    for c in comp_cols:
        df[f"{c}_RAW"] = df[c]

    for c in comp_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    if "Al" not in df.columns:
        df["Al"] = np.nan

    adds_cols = [c for c in comp_cols if c != "Al"]
    df["comp_sum_before"] = df[comp_cols].fillna(0).sum(axis=1)
    df["comp_fix_applied"] = 0

    def row_fix(row):
        nonlocal rep
        # clip negatives in additions
        for c in adds_cols:
            v = row.get(c)
            if pd.notna(v) and v < 0:
                row[c] = 0.0
                rep["rows_negative_clipped"] += 1

        adds = float(row[adds_cols].fillna(0).sum())
        Al   = float(row["Al"]) if pd.notna(row["Al"]) else np.nan

        if adds > 100:
            for c in adds_cols:
                if pd.notna(row[c]):
                    row[c] = row[c] / adds * 100.0
            row["Al"] = 0.0
            row["comp_fix_applied"] = 1
            rep["rows_adds_gt_100"] += 1
            return row

        if (pd.isna(Al)) or (Al < 50.0) or (abs((Al + adds) - 100.0) > tol_small):
            row["Al"] = max(0.0, 100.0 - adds)
            row["comp_fix_applied"] = 1
            rep["rows_al_balance_applied"] += 1
            return row

        tot = Al + adds
        if abs(tot - 100.0) > tol_tiny and tot > 0:
            scale = 100.0 / tot
            for c in [*adds_cols, "Al"]:
                if pd.notna(row[c]): row[c] = row[c] * scale
            row["comp_fix_applied"] = 1
            rep["rows_full_comp_renorm"] += 1
        return row

    df.loc[:, [*adds_cols, "Al","comp_fix_applied"]] = df.apply(row_fix, axis=1)[[*adds_cols,"Al","comp_fix_applied"]]
    df["comp_sum_after"] = df[[*adds_cols,"Al"]].fillna(0).sum(axis=1)
    return rep

# ---------- Coverage ----------
def compute_coverage(df: pd.DataFrame, group_cols: List[str], target_col: str) -> pd.DataFrame:
    d = df.assign(has_label=df[target_col].notna().astype(int))
    cov = d.groupby(group_cols, dropna=False)["has_label"].agg(["sum","count"]).reset_index()
    cov = cov.rename(columns={"sum":"n_labeled", "count":"n_rows"})
    cov["frac_labeled"] = np.where(cov["n_rows"]>0, cov["n_labeled"]/cov["n_rows"], 0.0)
    return cov

# ---------- Main ----------
def main(input_path: str, out_dir: str, group_cols: List[str]) -> None:
    outp = Path(out_dir); outp.mkdir(parents=True, exist_ok=True)

    # Read robustly; fix column names
    df = _read_csv_robust(input_path)
    unnamed = [c for c in df.columns if c.lower().startswith("unnamed")]
    if unnamed: df = df.drop(columns=unnamed)
    df.columns = [c.strip().replace("�", "µ") for c in df.columns]

    # Default group columns
    if not group_cols:
        group_cols = df.columns[:3].tolist()

    # Composition fix (auditable)
    comp_rep = fix_composition_with_audit(df)

    # ---------- Standardize units ----------
    # Ingot thickness -> mm (numeric)
    if "Ingot thickness" in df.columns:
        def thick_to_mm(v):
            if pd.isna(v): return np.nan
            s = str(v).strip().lower()
            m = re.search(r"(\d+(?:\.\d+)?)", s)
            if not m: return np.nan
            mg = float(m.group(1))
            if '"' in s or re.search(r"\b(in|inch|inches)\b", s): return mg*25.4
            if re.search(r"\bcm\b", s): return mg*10.0
            if re.search(r"\bm\b", s):  return mg*1000.0
            if re.search(r"\b(µm|um|micron|microns)\b", s): return mg/1000.0
            return mg
        df["Ingot thickness (mm)"] = df["Ingot thickness"].apply(thick_to_mm)
        df = df.drop(columns=["Ingot thickness"])

    # ---------- Homogenization: canonical "Celsius/Seconds" + numeric features ----------
    homog_source_col = select_text_source_column(df, HOMOG_TEXT_PRIORITY)
    if homog_source_col:
        source_text = df[homog_source_col].astype(str)
        if homog_source_col in AMBIGUOUS_HOMOG_COLUMNS and "Homogenization_RAW" not in df.columns:
            df["Homogenization_RAW"] = source_text
        unit_hint = infer_unit_hint(homog_source_col)
        segs_series = source_text.apply(lambda txt: parse_homog_segments(txt, unit_hint))
        H = segs_series.apply(summarize_homog).apply(pd.Series)
        df = df.join(H)

        def to_celsius_seconds_str(segs):
            if not segs: return ""
            parts = [f"{int(round(T))}/{hours_to_seconds(h)}" for (T,h) in segs]
            return " + ".join(parts)

        df["Homogenization (Celsius/Seconds)"] = segs_series.apply(to_celsius_seconds_str)
    else:
        for c in ["homog_steps","homog_time_total_s","homog_temp_max_C","homog_T_time_weighted_C","homog_time_at_ge_520C_s","adequate_homog"]:
            df[c] = np.nan
        df["Homogenization (Celsius/Seconds)"] = ""

    for col in AMBIGUOUS_HOMOG_COLUMNS:
        if col in df.columns:
            df = df.drop(columns=[col])

    # ---------- Recrystallization: canonical "Celsius/Seconds" + numeric features ----------
    recryst_source_col = select_text_source_column(df, RECRYST_TEXT_PRIORITY)
    if recryst_source_col:
        source_text = df[recryst_source_col].astype(str)
        if recryst_source_col in AMBIGUOUS_RECRYST_COLUMNS and "Recrystallization annealing_RAW" not in df.columns:
            df["Recrystallization annealing_RAW"] = source_text
        unit_hint = infer_unit_hint(recryst_source_col)
        rec_series = source_text.apply(lambda txt: parse_recryst_segments(txt, unit_hint))
        R = rec_series.apply(summarize_recryst).apply(pd.Series)
        df = df.join(R)

        def rec_to_celsius_seconds_str(rec):
            segs = rec["segs"]
            if not segs:
                if rec["type"]=="direct" and pd.notna(rec.get("direct_exit_T", np.nan)):
                    return f"{int(round(rec['direct_exit_T']))}/0"
                return ""
            parts = [f"{int(round(T))}/{hours_to_seconds(h)}" for (T,h) in segs]
            return " + ".join(parts)

        df["Recrystallization annealing (Celsius/Seconds)"] = rec_series.apply(rec_to_celsius_seconds_str)
    else:
        for c in ["recryst_type","recryst_steps","recryst_time_total_s","recryst_temp_max_C","recryst_T_time_weighted_C",
                  "adequate_recryst","adequate_recryst_batch","adequate_recryst_flash"]:
            df[c] = np.nan
        df["Recrystallization annealing (Celsius/Seconds)"] = ""

    for col in AMBIGUOUS_RECRYST_COLUMNS:
        if col in df.columns:
            df = df.drop(columns=[col])

    # ---------- Targets numeric & integrity ----------
    for full,_ in TARGETS:
        if full in df.columns:
            df[full] = pd.to_numeric(df[full], errors="coerce")

    if {"YS (MPa)","UTS (MPa)"} <= set(df.columns):
        bad = df["YS (MPa)"].notna() & df["UTS (MPa)"].notna() & (df["YS (MPa)"] > df["UTS (MPa)"])
        df = df.loc[~bad]

    if {"Uniform EL (percentage)","Fracture EL (percentage)"} <= set(df.columns):
        mask = df["Uniform EL (percentage)"].notna() & df["Fracture EL (percentage)"].notna() & \
               (df["Uniform EL (percentage)"] > df["Fracture EL (percentage)"])
        df.loc[mask, "Uniform EL (percentage)"] = df.loc[mask, "Fracture EL (percentage)"]

    # Missing-label flags for baseline only (no imputation)
    for full, short in TARGETS:
        if full in df.columns:
            df[f"flag_missing_{short}"] = df[full].isna().astype(int)

    # Gentle clipping (guard-rails)
    clip_rules = {
        "YS (MPa)": (50, 300), "UTS (MPa)": (100, 400),
        "Uniform EL (percentage)": (0, 60), "Fracture EL (percentage)": (0, 60),
        "YPE (percentage)": (0, 5),
        "Mean grain size (µm)": (0.5, 500),
        "Hot rolling reduction (percentage)": (0, 100),
        "Cold rolling reduction (percentage)": (0, 100),
    }
    for col,(lo,hi) in clip_rules.items():
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").clip(lo, hi)

    # ---------- Column ordering ----------
    prefer_order = [
        "file_name","alloy","card",
        *[c for c in ELEMENTS if c in df.columns],
        "YS (MPa)","UTS (MPa)","Uniform EL (percentage)","Fracture EL (percentage)","YPE (percentage)",
        "Casting",
        "Ingot thickness (mm)",
        "Homogenization (Celsius/Seconds)",
        "homog_steps","homog_time_total_s","homog_temp_max_C","homog_T_time_weighted_C","homog_time_at_ge_520C_s","adequate_homog",
        "Hot rolling reduction (percentage)","Cold rolling reduction (percentage)",
        "Recrystallization annealing (Celsius/Seconds)",
        "recryst_type","recryst_steps","recryst_time_total_s","recryst_temp_max_C","recryst_T_time_weighted_C","adequate_recryst","adequate_recryst_batch","adequate_recryst_flash",
        "Mean grain size (µm)","Secondary phase particle size and density","Crystallographic texture",
        "flag_missing_YS","flag_missing_UTS","flag_missing_FractureEL","flag_missing_UniformEL","flag_missing_YPE",
        "Homogenization_RAW","Recrystallization annealing_RAW",
        "comp_sum_before","comp_fix_applied","comp_sum_after",
        *[c for c in df.columns if c.endswith("_RAW") and c not in ("Homogenization_RAW","Recrystallization annealing_RAW")],
    ]
    cols_final = [c for c in prefer_order if c in df.columns] + [c for c in df.columns if c not in prefer_order]
    df = df[cols_final]

    # ---------- EXPORTS ----------
    outp.mkdir(parents=True, exist_ok=True)
    _write_csv(outp / "BASE_MASTER.csv", df)

    # Views, coverage, labels-needed
    labels_needed = []
    for full, short in TARGETS:
        if full not in df.columns: continue
        view = df[df[full].notna()].copy()
        _write_csv(outp / f"VIEW_{short}.csv", view)
        cov = compute_coverage(df, group_cols, full)
        _write_csv(outp / f"coverage_{short}.csv", cov)
        zero = cov[cov["n_labeled"]==0]
        for _, r in zero.iterrows():
            rec = {f"group_{i+1}": r[group_cols[i]] for i in range(len(group_cols))}
            rec.update({"target": short, "n_rows": int(r["n_rows"])})
            labels_needed.append(rec)
    labels_needed_df = pd.DataFrame(labels_needed, columns=[*(f"group_{i+1}" for i in range(len(group_cols))), "target", "n_rows"])
    _write_csv(outp / "labels_needed.csv", labels_needed_df)

    # QC report
    report = {
        "rows_final": int(len(df)),
        "composition_fix": comp_rep,
        "base_master": str(outp / "BASE_MASTER.csv"),
        "views": {short: str(outp / f"VIEW_{short}.csv") for _, short in TARGETS if (outp / f"VIEW_{short}.csv").exists()},
        "coverages": {short: str(outp / f"coverage_{short}.csv") for _, short in TARGETS if (outp / f"coverage_{short}.csv").exists()},
        "labels_needed": str(outp / "labels_needed.csv"),
        "notes": [
            "Homogenization/Recrystallization parsed to seconds and numeric features.",
            "RAW text retained in *_RAW; ambiguous source columns dropped after parsing.",
            "Targets are never imputed; use VIEW_<TARGET>.csv for supervised training/synthesis."
        ]
    }
    (outp / "cleaning_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Clean OG dataset; standardize units and export BASE_MASTER + VIEW_Ts.")
    ap.add_argument("--input", required=True, help="Input OG CSV (e.g., OG_dataset_cards_all_one_row_cleaned.csv)")
    ap.add_argument("--out-dir", required=True, help="Output directory")
    ap.add_argument("--group-cols", default="", help="Comma-separated class/group columns; default: first 3")
    args = ap.parse_args()
    group_cols = [c.strip() for c in args.group_cols.split(",") if c.strip()] if args.group_cols else []
    main(args.input, args.out_dir, group_cols)

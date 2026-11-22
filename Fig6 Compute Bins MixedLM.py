#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Reproduce Figure 6 a/b/c from DataS4-Plant Response Analysis.xlsx.

Usage:
    python Fig6_Recompute_Bins_MixedLM.py --input "DataS4-Plant Response Analysis.xlsx" --outdir Fig6_recalculated

This script uses the supplied effect-level data, not the previously fitted
statistics. It independently re-assigns bins using the dose and each row's
pollutant-specific P90, fits four study-random-intercept models using REML,
reconstructs Fig. 6c at the study x pollutant level, and compares results
against the stored outcomes in sheet 03_Analysis_Results.

The observational units are treatment-control comparisons, not papers.
There is no inverse-variance weighting: variance is available for only a
minority of the effects. Study ID is the random intercept grouping variable.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels
import statsmodels.formula.api as smf

BANDS = [
    "Within field range",
    "1 to 10x above field P90",
    "10 to 100x above field P90",
    "More than 100x above field P90",
]
SCENARIOS = {
    "Primary_83": {
        "p90": "Primary_P90_mgkg",
        "original_band": "Primary_Dose_Band",
        "design_scenario": "Primary_83",
    },
    "Sensitivity_583.59": {
        "p90": "Sensitivity_G3_P90_mgkg",
        "original_band": "Sensitivity_G3_Band",
        "design_scenario": "Sensitivity_G3_583.59",
    },
}
PANELS = {
    "Fig6a_growth_lnRR": {"response": "lnRR", "only_fig6a": True},
    "Fig6b_all_abs_lnRR": {"response": "abs_lnRR", "only_fig6a": False},
}


def check_columns(frame: pd.DataFrame, required: list[str], sheet_name: str) -> None:
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"{sheet_name}: missing required columns: {missing}")


def calculate_band(dose: pd.Series, p90: pd.Series) -> pd.Series:
    values = pd.to_numeric(dose, errors="coerce").to_numpy(dtype=float)
    ref = pd.to_numeric(p90, errors="coerce").to_numpy(dtype=float)
    if not (np.isfinite(values).all() and np.isfinite(ref).all()):
        raise ValueError("dose_mgkg and P90 must contain finite numeric values")
    if (values <= 0).any() or (ref <= 0).any():
        raise ValueError("dose_mgkg and P90 must be strictly positive")
    ratio = values / ref
    cut = np.select([ratio <= 1.0, ratio <= 10.0, ratio <= 100.0], BANDS[:3], default=BANDS[3])
    return pd.Series(cut, index=dose.index, dtype="string")


def validate_effects(effects: pd.DataFrame) -> pd.DataFrame:
    required = ["Effect_ID", "study_id", "pollutant_group", "dose_mgkg", "lnRR", "abs_lnRR", "In_Fig6a"]
    for config in SCENARIOS.values():
        required += [config["p90"], config["original_band"]]
    check_columns(effects, required, "01_Effect_CV_Data")
    if effects["Effect_ID"].duplicated().any():
        raise ValueError("Duplicate Effect_ID found; check Data S4")
    if effects[["study_id", "lnRR", "abs_lnRR"]].isna().any().any():
        raise ValueError("Study ID or effect sizes are missing")
    if not np.allclose(effects["abs_lnRR"].to_numpy(), np.abs(effects["lnRR"].to_numpy()), atol=1e-9):
        raise ValueError("abs_lnRR does not equal abs(lnRR) for all records")
    out = effects.copy()
    for scenario, cfg in SCENARIOS.items():
        name = f"Recomputed_{scenario}_Band"
        ratio_name = f"Recomputed_{scenario}_Dose_over_P90"
        out[name] = calculate_band(out["dose_mgkg"], out[cfg["p90"]])
        out[ratio_name] = out["dose_mgkg"] / out[cfg["p90"]]
        different = (out[name].astype(str) != out[cfg["original_band"]].astype(str)).to_numpy()
        if different.any():
            examples = out.loc[different, ["Effect_ID", "dose_mgkg", cfg["p90"], name, cfg["original_band"]]].head(5)
            raise ValueError(f"{scenario}: {different.sum()} bin labels disagree with Data S4:\n{examples}")
    n6a = int(out["In_Fig6a"].eq(True).sum())
    if n6a != 371 or len(out) != 763:
        raise ValueError(f"Unexpected dataset sizes: Fig6a={n6a} (expected 371); Fig6b={len(out)} (expected 763)")
    return out


def fit_models(effects: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, diagnostics = [], []
    for scenario in SCENARIOS:
        band_col = f"Recomputed_{scenario}_Band"
        for panel, meta in PANELS.items():
            subset = effects.loc[effects["In_Fig6a"].eq(True)].copy() if meta["only_fig6a"] else effects.copy()
            subset["Band"] = pd.Categorical(subset[band_col], categories=BANDS, ordered=True)
            response = meta["response"]
            formula = f"{response} ~ 0 + C(Band)"
            model = smf.mixedlm(formula, subset, groups=subset["study_id"], re_formula="1")
            fit = model.fit(reml=True, method="powell", maxiter=2000, disp=False)
            if not fit.converged:
                raise RuntimeError(f"MixedLM failed to converge: {scenario} {panel}")
            intervals = fit.conf_int(alpha=0.05)
            for position, band in enumerate(BANDS, start=1):
                coefficient = f"C(Band)[{band}]"
                mean = float(fit.fe_params[coefficient])
                lower, upper = intervals.loc[coefficient].to_numpy(dtype=float)
                in_bin = subset.loc[subset["Band"].eq(band)]
                rows.append({
                    "Reference_scenario": scenario,
                    "Panel": panel,
                    "Band_order": position,
                    "Band": band,
                    "N_comparisons": len(in_bin),
                    "N_studies": in_bin["study_id"].nunique(),
                    "Model_mean": mean,
                    "CI95_lower": float(lower),
                    "CI95_upper": float(upper),
                    "Significant_vs_zero_by_CI": bool(lower > 0 or upper < 0),
                    "Model_formula": formula,
                    "Estimation": "REML",
                    "Optimizer": "Powell",
                    "Converged": bool(fit.converged),
                })
            diagnostics.append({
                "Reference_scenario": scenario,
                "Panel": panel,
                "N_comparisons": len(subset),
                "N_studies": subset["study_id"].nunique(),
                "Random_intercept_variance": float(fit.cov_re.iloc[0, 0]),
                "Residual_variance": float(fit.scale),
                "Log_likelihood": float(fit.llf),
                "Converged": bool(fit.converged),
                "Formula": formula,
                "REML": True,
                "Optimizer": "Powell",
            })
    return pd.DataFrame(rows), pd.DataFrame(diagnostics)


def fig6c_summary(design: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    check_columns(design, ["study_id", "pollutant_group", "dose_mgkg", "Primary_P90_mgkg", "Sensitivity_G3_P90_mgkg"], "02_Dose_Design")
    all_rows, detail = [], []
    for model_scenario, cfg in SCENARIOS.items():
        groups = []
        for (pollutant, study), group in design.groupby(["pollutant_group", "study_id"], dropna=False, sort=True):
            values = pd.to_numeric(group["dose_mgkg"], errors="coerce")
            positive = set(values.loc[values > 0].to_numpy(dtype=float))
            p90_values = group[cfg["p90"]].dropna().unique()
            if len(p90_values) != 1:
                raise ValueError(f"Ambiguous P90 for {study}/{pollutant}/{model_scenario}: {p90_values}")
            p90 = float(p90_values[0])
            n_low = sum(value <= p90 for value in positive)
            rec = {
                "Scenario": cfg["design_scenario"],
                "Pollutant_group": pollutant,
                "study_id": study,
                "N_distinct_positive_doses": len(positive),
                "N_distinct_positive_doses_le_P90": n_low,
                "Any_positive_dose_le_P90": int(n_low >= 1),
                "Two_or_more_positive_doses_le_P90": int(n_low >= 2),
                "Only_one_positive_dose_anywhere": int(len(positive) == 1),
                "P90_mgkg": p90,
            }
            detail.append(rec)
            groups.append(rec)
        grouped = pd.DataFrame(groups)
        for pollutant, subset in list(grouped.groupby("Pollutant_group", sort=True)) + [("ALL (study x pollutant; may overlap)", grouped)]:
            all_rows.append({
                "Scenario": cfg["design_scenario"],
                "Pollutant_group": pollutant,
                "Study_pollutant_groups": len(subset),
                "Studies_any_positive_dose_le_P90": int(subset["Any_positive_dose_le_P90"].sum()),
                "Studies_two_or_more_positive_doses_le_P90": int(subset["Two_or_more_positive_doses_le_P90"].sum()),
                "Studies_single_positive_dose_anywhere": int(subset["Only_one_positive_dose_anywhere"].sum()),
            })
    return pd.DataFrame(all_rows), pd.DataFrame(detail)


def compare_stored(target: pd.DataFrame, modeled: pd.DataFrame, fig6c: pd.DataFrame, tolerance: float = 1e-4) -> pd.DataFrame:
    issues = []
    for _, row in modeled.iterrows():
        matched = target.loc[
            target["Panel"].eq(row["Panel"]) & target["Reference_scenario"].eq(row["Reference_scenario"]) &
            target["Band_order"].eq(float(row["Band_order"]))
        ]
        if len(matched) != 1:
            issues.append({"kind": "Model", "item": str((row["Reference_scenario"], row["Panel"], row["Band"])), "status": "MISSING_OR_DUPLICATE_TARGET"})
            continue
        old = matched.iloc[0]
        for key in ["Model_mean", "CI95_lower", "CI95_upper"]:
            delta = abs(float(row[key]) - float(old[key]))
            issues.append({"kind": "Model", "item": f"{row['Reference_scenario']}|{row['Panel']}|{row['Band']}|{key}",
                           "calculated": float(row[key]), "stored": float(old[key]), "absolute_difference": delta,
                           "status": "PASS" if delta <= tolerance else "FAIL"})
        difference = int(row["N_studies"]) - int(old["N_studies"])
        issues.append({"kind": "N_studies", "item": f"{row['Reference_scenario']}|{row['Panel']}|{row['Band']}",
                       "calculated": int(row["N_studies"]), "stored": int(old["N_studies"]),
                       "absolute_difference": abs(difference), "status": "PASS" if not difference else "FAIL"})
    for _, row in fig6c.iterrows():
        matched = target.loc[target["Analysis"].eq("Fig6c_Study_Summary") &
                             target["Scenario"].eq(row["Scenario"]) &
                             target["Pollutant_group"].eq(row["Pollutant_group"])]
        if len(matched) != 1:
            issues.append({"kind": "Fig6c", "item": str((row["Scenario"], row["Pollutant_group"])), "status": "MISSING_OR_DUPLICATE_TARGET"})
            continue
        old = matched.iloc[0]
        for key in ["Study_pollutant_groups", "Studies_any_positive_dose_le_P90",
                    "Studies_two_or_more_positive_doses_le_P90", "Studies_single_positive_dose_anywhere"]:
            delta = abs(float(row[key]) - float(old[key]))
            issues.append({"kind": "Fig6c", "item": f"{row['Scenario']}|{row['Pollutant_group']}|{key}",
                           "calculated": float(row[key]), "stored": float(old[key]),
                           "absolute_difference": delta, "status": "PASS" if delta <= tolerance else "FAIL"})
    return pd.DataFrame(issues)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, required=True, help="Path to DataS4-Plant Response Analysis.xlsx")
    parser.add_argument("--outdir", type=Path, default=Path("Fig6_recomputed"), help="Destination directory")
    parser.add_argument("--tolerance", type=float, default=1e-4, help="Absolute comparison tolerance with saved Data S4 outputs")
    args = parser.parse_args()
    if not args.input.is_file():
        parser.error(f"Input file not found: {args.input}")
    args.outdir.mkdir(parents=True, exist_ok=True)
    print(f"Reading workbook: {args.input}", flush=True)
    data = pd.read_excel(args.input, sheet_name="01_Effect_CV_Data")
    design = pd.read_excel(args.input, sheet_name="02_Dose_Design")
    target = pd.read_excel(args.input, sheet_name="03_Analysis_Results")
    data = validate_effects(data)
    modeled, diagnostics = fit_models(data)
    summary, study_details = fig6c_summary(design)
    checks = compare_stored(target, modeled, summary, args.tolerance)
    data[["Effect_ID", "study_id", "pollutant_group", "dose_mgkg", "In_Fig6a",
          "Primary_P90_mgkg", "Sensitivity_G3_P90_mgkg", "Primary_Dose_Band", "Sensitivity_G3_Band",
          "Recomputed_Primary_83_Band", "Recomputed_Sensitivity_583.59_Band",
          "Recomputed_Primary_83_Dose_over_P90", "Recomputed_Sensitivity_583.59_Dose_over_P90"]].to_csv(
              args.outdir / "Fig6_Dose_Bin_Check.csv", index=False, encoding="utf-8-sig")
    modeled.loc[modeled.Panel.eq("Fig6a_growth_lnRR")].to_csv(args.outdir / "Fig6a_MixedLM.csv", index=False, encoding="utf-8-sig")
    modeled.loc[modeled.Panel.eq("Fig6b_all_abs_lnRR")].to_csv(args.outdir / "Fig6b_MixedLM.csv", index=False, encoding="utf-8-sig")
    diagnostics.to_csv(args.outdir / "Fig6_MixedLM_Diagnostics.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(args.outdir / "Fig6c_Study_Design_Summary.csv", index=False, encoding="utf-8-sig")
    study_details.to_csv(args.outdir / "Fig6c_Study_Design_Details.csv", index=False, encoding="utf-8-sig")
    checks.to_csv(args.outdir / "Fig6_Reproduction_QC.csv", index=False, encoding="utf-8-sig")
    report = {
        "source_input": str(args.input), "python": sys.version.split()[0],
        "pandas": pd.__version__, "statsmodels": statsmodels.__version__,
        "n_total_effects": len(data), "n_fig6a_growth_effects": int(data["In_Fig6a"].eq(True).sum()),
        "n_total_studies": int(data["study_id"].nunique()),
        "n_design_records": len(design),
        "n_study_pollutant_pairs": len(study_details) // len(SCENARIOS),
        "model_definition": "REML MixedLM; fixed-effect 4-level P90 dose band with no global intercept; random intercept by study_id",
        "weighting": "Unweighted; variance data are incomplete",
        "p90_sources": "Per-row input columns; plasticizer P90 is not replaced by the microplastic P90",
        "total_checks": len(checks), "failed_checks": int(checks.status.ne("PASS").sum()),
        "max_abs_difference": float(checks.absolute_difference.max()),
        "all_checks_passed": bool(checks.status.eq("PASS").all()),
    }
    (args.outdir / "Fig6_Reproduction_Report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nFig6a/b models:\n", modeled[["Reference_scenario", "Panel", "Band", "N_comparisons", "N_studies", "Model_mean", "CI95_lower", "CI95_upper"]].to_string(index=False), flush=True)
    print("\nFig6c:\n", summary.to_string(index=False), flush=True)
    print("\nQC:", json.dumps({"checks": report["total_checks"], "failed": report["failed_checks"], "max_abs_difference": report["max_abs_difference"]}, indent=2), flush=True)
    print("Outputs:", args.outdir.resolve(), flush=True)
    return 0 if report["all_checks_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())

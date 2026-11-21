#!/usr/bin/env python3
"""Reproduce the nanoplastic-inclusive Fig. 5c and Fig. 5d analyses.

The analysis population is fixed at 763 effect-size records from 53 studies:
472 microplastic, 187 nanoplastic, and 104 plasticizer records.

Input
-----
Preferred input: the 763-record analysis workbook produced by the study, with
these worksheets:
  * ``04_主分析效应量``: raw/derived analysis records used for CatBoost CV.
  * ``09_Fig4c_归因输入``: study-demeaned variables used for Shapley R2.

Alternatively, provide the original 763-row ``effects_analysis_dataset.csv``
containing the required model and within-study fields.

Important
---------
This script does not draw or modify Fig. 5. It writes the numerical outputs
used to check Fig. 5c and Fig. 5d. It deliberately rejects the 576-row
microplastic-plus-plasticizer dataset and any input that omits nanoplastics.

Example
-------
python Fig5c_d_Nanoplastic_Inclusive_Analysis.py \\
  --input Fig4_180_study_analysis.xlsx --outdir Fig5c_d_results
"""
from __future__ import annotations

import argparse
import itertools
import json
import warnings
from math import factorial
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from sklearn.metrics import r2_score

warnings.filterwarnings("ignore")

CAT = [
    "pollutant_group", "polymer_cls", "response_axis", "organ", "crop_family",
    "shape", "aging", "setting", "coexposure",
]
NUMERIC = ["log10_dose", "log10_size_um", "duration_d", "soil_pH", "SOM_gkg"]
CV_FEATURES = NUMERIC + CAT
SHAPLEY_BLOCKS = [
    ("response_axis", "Response endpoint class"),
    ("organ", "Plant organ"),
    ("polymer_cls", "Polymer or compound"),
    ("shape", "Particle shape"),
    ("aging", "Aging status"),
    ("crop_family", "Crop family"),
    ("coexposure", "Co-exposure background"),
]
EXPECTED_COUNTS = {"Microplastic": 472, "Nanoplastic": 187, "Plasticizer": 104}
REFERENCE_FIGURE = {
    "Fig5c_R2": 0.1985397011,
    "Fig5d_Random_CV_R2": 0.3104515021705396,
    "Fig5d_Random_CV_R2_SD": 0.0221717180489346,
    "Fig5d_Study_Grouped_CV_R2": -0.11040217581530114,
    "Fig5d_Study_Grouped_CV_R2_SD": 0.0600430352154017,
}


def load_inputs(path: Path, cv_sheet: str | None, shapley_sheet: str | None):
    """Load CV records and Shapley records from CSV or the supplied workbook."""
    if path.suffix.lower() == ".csv":
        d = pd.read_csv(path, low_memory=False)
        return d.copy(), d.copy(), "CSV", "CSV"
    if path.suffix.lower() not in {".xlsx", ".xlsm", ".xls"}:
        raise ValueError("Input must be a CSV or Excel workbook.")

    book = pd.ExcelFile(path)
    cv_name = cv_sheet or ("04_主分析效应量" if "04_主分析效应量" in book.sheet_names else book.sheet_names[0])
    shp_name = shapley_sheet or (
        "09_Fig4c_归因输入" if "09_Fig4c_归因输入" in book.sheet_names else cv_name
    )
    if cv_name not in book.sheet_names:
        raise ValueError(f"CV worksheet not found: {cv_name}")
    if shp_name not in book.sheet_names:
        raise ValueError(f"Shapley worksheet not found: {shp_name}")
    return (
        pd.read_excel(path, sheet_name=cv_name),
        pd.read_excel(path, sheet_name=shp_name),
        cv_name,
        shp_name,
    )


def validate_population(cv_data: pd.DataFrame, shapley_data: pd.DataFrame) -> None:
    for name, data in [("CV input", cv_data), ("Shapley input", shapley_data)]:
        required = {"study_id", "record_id", "pollutant_group"}
        missing = sorted(required - set(data.columns))
        if missing:
            raise ValueError(f"{name} is missing required columns: {', '.join(missing)}")
        counts = data["pollutant_group"].value_counts().to_dict()
        if len(data) != 763 or data["study_id"].nunique() != 53 or counts != EXPECTED_COUNTS:
            raise ValueError(
                f"{name} must contain exactly 763 records from 53 studies "
                f"(472 microplastics, 187 nanoplastics, 104 plasticizers); "
                f"found {len(data)} records, {data['study_id'].nunique()} studies, {counts}. "
                "Do not use the 576-record nanoplastic-excluded dataset."
            )

    cv_ids = set(cv_data["record_id"].astype(str))
    shp_ids = set(shapley_data["record_id"].astype(str))
    if cv_ids != shp_ids:
        raise ValueError("CV and Shapley worksheets do not contain the same 763 record IDs.")


def _within_study_shapley_from_precomputed(data: pd.DataFrame):
    """Use the already demeaned columns from 09_Fig4c_归因输入 when available."""
    needed = ["y_dev", "dose_dev", "dur_dev", "size_dev", "dur_dev_miss", "size_dev_miss"]
    if not set(needed).issubset(data.columns):
        return None

    D = data.copy()
    for col in needed:
        D[col] = pd.to_numeric(D[col], errors="coerce")
    if not np.isfinite(D[needed].to_numpy(dtype=float)).all():
        raise ValueError("Precomputed within-study variables contain missing or non-finite values.")

    blocks = {
        "Exposure concentration": ["dose_dev"],
        "Exposure duration": ["dur_dev", "dur_dev_miss"],
        "Particle size": ["size_dev", "size_dev_miss"],
    }
    for col, label in SHAPLEY_BLOCKS:
        if col not in D.columns:
            raise ValueError(f"Shapley input is missing categorical field: {col}")
        dummy = pd.get_dummies(
            D[col].fillna("Not reported").astype(str),
            prefix=col,
            drop_first=True,
            dtype=float,
        )
        dummy = dummy - dummy.groupby(D["study_id"]).transform("mean")
        keep = [name for name in dummy.columns if dummy[name].abs().sum() > 1e-9]
        for name in keep:
            D[name] = dummy[name]
        blocks[label] = keep

    return _calculate_lmg(D, blocks, "y_dev")


def _within_study_shapley_from_raw(data: pd.DataFrame):
    """Fallback calculation from raw effect and dose columns."""
    required = ["study_id", "abs_lnRR", "log10_dose", "duration_d", "log10_size_um"]
    missing = [c for c in required if c not in data.columns]
    if missing:
        raise ValueError("Raw Shapley input is missing columns: " + ", ".join(missing))

    D = data.copy()
    for col in ["abs_lnRR", "log10_dose", "duration_d", "log10_size_um"]:
        D[col] = pd.to_numeric(D[col], errors="coerce")
    D["y_dev"] = D["abs_lnRR"] - D.groupby("study_id")["abs_lnRR"].transform("mean")
    D["dose_dev"] = D["log10_dose"] - D.groupby("study_id")["log10_dose"].transform("mean")
    D["dur_dev"] = D["duration_d"] - D.groupby("study_id")["duration_d"].transform("mean")
    D["size_dev"] = D["log10_size_um"] - D.groupby("study_id")["log10_size_um"].transform("mean")
    for col in ["dur_dev", "size_dev"]:
        D[col + "_miss"] = D[col].isna().astype(float)
        D[col] = D[col].fillna(0.0)

    blocks = {
        "Exposure concentration": ["dose_dev"],
        "Exposure duration": ["dur_dev", "dur_dev_miss"],
        "Particle size": ["size_dev", "size_dev_miss"],
    }
    for col, label in SHAPLEY_BLOCKS:
        dummy = pd.get_dummies(
            D[col].fillna("Not reported").astype(str),
            prefix=col,
            drop_first=True,
            dtype=float,
        )
        dummy = dummy - dummy.groupby(D["study_id"]).transform("mean")
        keep = [name for name in dummy.columns if dummy[name].abs().sum() > 1e-9]
        for name in keep:
            D[name] = dummy[name]
        blocks[label] = keep
    for col in ["dur_dev", "size_dev", "dur_dev_miss", "size_dev_miss"]:
        D[col] = D[col] - D.groupby("study_id")[col].transform("mean")
    return _calculate_lmg(D, blocks, "y_dev")


def _calculate_lmg(data: pd.DataFrame, blocks: dict[str, list[str]], target: str):
    blocks = {name: cols for name, cols in blocks.items() if cols}
    names = list(blocks)
    y = pd.to_numeric(data[target], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(y).all() or np.sum(y ** 2) <= 0:
        raise ValueError("Shapley response vector is invalid or has zero variance.")

    cache: dict[frozenset, float] = {}
    for n in range(len(names) + 1):
        for subset in itertools.combinations(names, n):
            columns = [col for block in subset for col in blocks[block]]
            if not columns:
                value = 0.0
            else:
                X = data[columns].to_numpy(dtype=float)
                if not np.isfinite(X).all():
                    raise ValueError("Non-finite values found in the Shapley design matrix.")
                beta, *_ = np.linalg.lstsq(X, y, rcond=None)
                residual = y - X @ beta
                value = 1.0 - float(np.sum(residual ** 2) / np.sum(y ** 2))
            cache[frozenset(subset)] = value

    n_blocks = len(names)
    contributions: dict[str, float] = {}
    for block in names:
        others = [name for name in names if name != block]
        total = 0.0
        for n in range(len(others) + 1):
            for subset in itertools.combinations(others, n):
                weight = factorial(n) * factorial(n_blocks - n - 1) / factorial(n_blocks)
                total += weight * (
                    cache[frozenset(subset) | {block}] - cache[frozenset(subset)]
                )
        contributions[block] = total

    full_r2 = cache[frozenset(names)]
    table = pd.DataFrame({"Factor": list(contributions), "Shapley_R2": list(contributions.values())})
    table["Share_of_Explained_Variance_Percent"] = 100 * table["Shapley_R2"] / full_r2
    return table.sort_values("Shapley_R2", ascending=False), full_r2


def grouped_folds(study_ids: np.ndarray, seed: int, n_splits: int = 5) -> np.ndarray:
    unique_studies = np.unique(study_ids)
    permuted = np.random.default_rng(seed).permutation(len(unique_studies))
    fold_map = {study: int(i) % n_splits for study, i in zip(unique_studies, permuted)}
    return np.asarray([fold_map[study] for study in study_ids], dtype=int)


def catboost_model(seed: int) -> CatBoostRegressor:
    return CatBoostRegressor(
        iterations=600,
        depth=4,
        learning_rate=0.04,
        l2_leaf_reg=6,
        loss_function="RMSE",
        random_seed=seed,
        verbose=0,
        allow_writing_files=False,
    )


def prepare_cv_data(data: pd.DataFrame) -> pd.DataFrame:
    missing = sorted(set(CV_FEATURES + ["study_id", "record_id", "abs_lnRR"]) - set(data.columns))
    if missing:
        raise ValueError("CV input is missing columns: " + ", ".join(missing))
    X = data[CV_FEATURES].copy()
    for col in CAT:
        X[col] = X[col].fillna("Not reported").astype(str)
    for col in NUMERIC:
        X[col] = pd.to_numeric(X[col], errors="coerce")
        # Do not convert unknown values to zero; CatBoost handles numeric NaNs.
        X[col] = X[col].replace([np.inf, -np.inf], np.nan)
    return X


def run_cv(data: pd.DataFrame) -> pd.DataFrame:
    X = prepare_cv_data(data)
    y = pd.to_numeric(data["abs_lnRR"], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(y).all():
        raise ValueError("CV response abs_lnRR contains missing or non-finite values.")
    study_ids = data["study_id"].astype(str).to_numpy()
    rows: list[dict] = []

    for repeat in range(5):
        random_folds = np.random.default_rng(repeat).integers(0, 5, len(y))
        study_folds = grouped_folds(study_ids, repeat, n_splits=5)
        for validation, folds in [
            ("Random folds", random_folds),
            ("Study-grouped folds", study_folds),
        ]:
            predictions = np.zeros(len(y), dtype=float)
            for fold in range(5):
                train = folds != fold
                test = folds == fold
                model = catboost_model(repeat)
                train_pool = Pool(X.loc[train, CV_FEATURES], y[train], cat_features=CAT)
                test_pool = Pool(X.loc[test, CV_FEATURES], cat_features=CAT)
                model.fit(train_pool)
                predictions[test] = model.predict(test_pool)
            rows.append({
                "Validation": validation,
                "Repeat": repeat + 1,
                "R2": float(r2_score(y, predictions)),
                "N_Records": int(len(y)),
                "N_Studies": int(pd.Series(study_ids).nunique()),
            })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="763-record CSV/XLSX analysis input")
    parser.add_argument("--cv-sheet", default=None, help="Excel worksheet for CV records")
    parser.add_argument("--shapley-sheet", default=None, help="Excel worksheet for Shapley records")
    parser.add_argument("--outdir", type=Path, default=Path("Fig5c_d_results"))
    args = parser.parse_args()

    cv_data, shapley_data, cv_sheet_name, shapley_sheet_name = load_inputs(
        args.input, args.cv_sheet, args.shapley_sheet
    )
    validate_population(cv_data, shapley_data)

    # Prefer the precomputed within-study deviations in the 763-record analysis
    # workbook. This reproduces the study-demeaned Shapley calculation exactly.
    shapley_result = _within_study_shapley_from_precomputed(shapley_data)
    shapley_source = "precomputed_within_study_deviations"
    if shapley_result is None:
        shapley_result = _within_study_shapley_from_raw(shapley_data)
        shapley_source = "recalculated_from_raw_fields"
    shapley, full_r2 = shapley_result

    cv = run_cv(cv_data)
    args.outdir.mkdir(parents=True, exist_ok=True)
    shapley.to_csv(args.outdir / "Fig5c_Shapley_R2_Decomposition.csv", index=False)
    cv.to_csv(args.outdir / "Fig5d_Cross_Validation_R2.csv", index=False)

    cv_summary = cv.groupby("Validation")["R2"].agg(["mean", "std", "min", "max"]).reset_index()
    cv_summary.to_csv(args.outdir / "Fig5d_Cross_Validation_Summary.csv", index=False)
    summary = {
        "Input_File": str(args.input),
        "CV_Worksheet": cv_sheet_name,
        "Shapley_Worksheet": shapley_sheet_name,
        "N_Records": int(len(cv_data)),
        "N_Studies": int(cv_data["study_id"].nunique()),
        "Pollutant_Record_Counts": cv_data["pollutant_group"].value_counts().to_dict(),
        "Nanoplastic_Included": True,
        "Fig5c_ShAP_R2": float(full_r2),
        "Fig5c_ShAP_R2_Sum": float(shapley["Shapley_R2"].sum()),
        "Fig5c_ShAP_Source": shapley_source,
        "Fig5d_Random_CV_R2_Mean": float(cv.loc[cv["Validation"].eq("Random folds"), "R2"].mean()),
        "Fig5d_Random_CV_R2_SD": float(cv.loc[cv["Validation"].eq("Random folds"), "R2"].std(ddof=1)),
        "Fig5d_Study_Grouped_CV_R2_Mean": float(cv.loc[cv["Validation"].eq("Study-grouped folds"), "R2"].mean()),
        "Fig5d_Study_Grouped_CV_R2_SD": float(cv.loc[cv["Validation"].eq("Study-grouped folds"), "R2"].std(ddof=1)),
        "Figure_Reference_Values": REFERENCE_FIGURE,
    }
    # A clear warning is preferable to silently claiming exact reproduction.
    differences = {
        "Fig5c_R2": summary["Fig5c_ShAP_R2"] - REFERENCE_FIGURE["Fig5c_R2"],
        "Fig5d_Random_CV_R2": summary["Fig5d_Random_CV_R2_Mean"] - REFERENCE_FIGURE["Fig5d_Random_CV_R2"],
        "Fig5d_Study_Grouped_CV_R2": summary["Fig5d_Study_Grouped_CV_R2_Mean"] - REFERENCE_FIGURE["Fig5d_Study_Grouped_CV_R2"],
        "Fig5d_Random_CV_R2_SD": summary["Fig5d_Random_CV_R2_SD"] - REFERENCE_FIGURE["Fig5d_Random_CV_R2_SD"],
        "Fig5d_Study_Grouped_CV_R2_SD": summary["Fig5d_Study_Grouped_CV_R2_SD"] - REFERENCE_FIGURE["Fig5d_Study_Grouped_CV_R2_SD"],
    }
    summary["Difference_From_Current_Figure_Reference"] = differences
    summary["Matches_Current_Figure_Reference"] = bool(
        abs(differences["Fig5c_R2"]) <= 0.002
        and abs(differences["Fig5d_Random_CV_R2"]) <= 0.01
        and abs(differences["Fig5d_Study_Grouped_CV_R2"]) <= 0.01
        and abs(differences["Fig5d_Random_CV_R2_SD"]) <= 0.005
        and abs(differences["Fig5d_Study_Grouped_CV_R2_SD"]) <= 0.005
    )
    (args.outdir / "Fig5c_d_Summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if not summary["Matches_Current_Figure_Reference"]:
        print(
            "WARNING: the supplied 763-row input does not reproduce all current Fig. 5c/d reference values. "
            "Check that the input is the exact nanoplastic-inclusive source file used for the manuscript figures."
        )


if __name__ == "__main__":
    main()

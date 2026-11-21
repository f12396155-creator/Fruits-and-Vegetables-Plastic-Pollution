#!/usr/bin/env python3
"""Reproduce the CatBoost + SHAP dose-determinant analysis for Fig. 4a.

Accepted input formats
----------------------
1. The original 286-row workbook with a worksheet named ``模型8因素输入``.
2. A workbook/CSV containing the equivalent columns under the aliases handled
   below (for example, the 286-row model-benchmark worksheet).

The analysis retains all three pollutant groups, including nanoplastics.
It uses 10 repeats of study-grouped five-fold cross-validation and the model
settings used in the supplied original Fig. 4a analysis.

Example
-------
python Fig4a_Dose_Determinants_SHAP.py --input Fig4a_Model_Input.xlsx
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# The exact model feature order used in the original analysis.
FEATURES = [
    "poll_group_model",
    "unit_convention_model",
    "size_class_model",
    "compound_model",
    "country_model",
    "crop_family_model",
    "year",
    "n_levels_model",
]
CAT_FEATURES = FEATURES[:6]
ID_COLUMNS = ["Study_ID", "Plot_point_ID"]
TARGET = "Target_Y"
EXPECTED_GROUP_COUNTS = {
    "Microplastic": 203,
    "Nanoplastic": 32,
    "Plasticizer": 51,
}
MODEL_PARAMS = {
    "iterations": 500,
    "depth": 4,
    "learning_rate": 0.05,
    "l2_leaf_reg": 5,
    "loss_function": "RMSE",
}

# Canonical name -> accepted source aliases, in priority order.
ALIASES = {
    "poll_group_model": ["poll_group_model", "poll_group", "Pollutant_class", "Pollutant_Group"],
    "unit_convention_model": ["unit_convention_model", "unit_convention", "Unit_group", "Unit_convention"],
    "size_class_model": ["size_class_model", "size_class", "Particle_size_class"],
    "compound_model": ["compound_model", "compound_feature", "compound", "polymer", "Polymer"],
    "country_model": ["country_model", "country", "Country"],
    "crop_family_model": ["crop_family_model", "crop_family", "Crop_family"],
    "year": ["year", "publication_year", "Publication_year"],
    "n_levels_model": ["n_levels_model", "n_levels", "Dose_level_count"],
    "Target_Y": ["Target_Y", "Y", "log10_exceed", "log10_dose_over_field_P50"],
    "Study_ID": ["Study_ID", "study_id", "Study_ID"],
    "Plot_point_ID": ["Plot_point_ID", "plot_point_id", "Record_ID", "record_id"],
}


def _first_existing(columns: Iterable[str], candidates: list[str]) -> str | None:
    cols = set(columns)
    return next((name for name in candidates if name in cols), None)


def normalize_model_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Map source column variants onto the canonical model schema."""
    rename: dict[str, str] = {}
    missing: list[str] = []
    for canonical, candidates in ALIASES.items():
        found = _first_existing(frame.columns, candidates)
        if found is None:
            missing.append(f"{canonical} (accepted: {', '.join(candidates)})")
        elif found != canonical:
            rename[found] = canonical
    if missing:
        raise ValueError("Input table is missing required model fields:\n- " + "\n- ".join(missing))
    out = frame.rename(columns=rename).copy()
    return out[FEATURES + [TARGET] + ID_COLUMNS]


def load_model_input(path: Path, sheet: str | None = None) -> tuple[pd.DataFrame, str]:
    """Read CSV/XLSX; auto-detect the first worksheet matching the model schema."""
    if path.suffix.lower() in {".csv", ".tsv"}:
        sep = "\t" if path.suffix.lower() == ".tsv" else ","
        raw = pd.read_csv(path, sep=sep, low_memory=False)
        return normalize_model_table(raw), "CSV"

    if path.suffix.lower() not in {".xlsx", ".xlsm", ".xls"}:
        raise ValueError("Input must be a CSV or Excel workbook.")

    book = pd.ExcelFile(path)
    candidates = [sheet] if sheet else []
    if not sheet:
        # Prefer the original model sheet, then scan all worksheets.
        candidates = ["模型8因素输入", "Model_Input", "Model_8_Factors"] + book.sheet_names
    seen: set[str] = set()
    errors: list[str] = []
    for sheet_name in candidates:
        if sheet_name in seen or sheet_name not in book.sheet_names:
            continue
        seen.add(sheet_name)
        raw = pd.read_excel(path, sheet_name=sheet_name)
        try:
            return normalize_model_table(raw), sheet_name
        except ValueError as exc:
            errors.append(f"{sheet_name}: {str(exc).splitlines()[0]}")
    raise ValueError(
        "No worksheet could be mapped to the Fig. 4a model schema. Checked: "
        + "; ".join(errors)
    )


def make_study_grouped_folds(study_ids: np.ndarray, seed: int, n_splits: int = 5) -> np.ndarray:
    unique_studies = np.unique(study_ids)
    permuted = np.random.default_rng(seed).permutation(len(unique_studies))
    fold_map = {study: int(i) % n_splits for study, i in zip(unique_studies, permuted)}
    return np.asarray([fold_map[study] for study in study_ids], dtype=int)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="286-row model-input workbook or CSV")
    parser.add_argument("--sheet", default=None, help="Optional worksheet name; otherwise auto-detected")
    parser.add_argument("--outdir", type=Path, default=Path("Fig4a_SHAP_results"))
    args = parser.parse_args()

    data, source_sheet = load_model_input(args.input, args.sheet)
    data[TARGET] = pd.to_numeric(data[TARGET], errors="coerce")
    data["year"] = pd.to_numeric(data["year"], errors="coerce")
    data["n_levels_model"] = pd.to_numeric(data["n_levels_model"], errors="coerce")
    data = data.dropna(subset=[TARGET, "Study_ID", "Plot_point_ID"]).reset_index(drop=True)

    # Do not silently substitute a different model population.
    if len(data) != 286 or data["Study_ID"].nunique() != 86:
        raise ValueError(
            f"Expected 286 model records from 86 studies; found {len(data)} records "
            f"from {data['Study_ID'].nunique()} studies."
        )
    group_counts = data["poll_group_model"].value_counts().to_dict()
    if group_counts != EXPECTED_GROUP_COUNTS:
        raise ValueError(
            "Pollutant counts do not match the final 286-record model set. "
            f"Expected {EXPECTED_GROUP_COUNTS}; found {group_counts}."
        )

    X = data[FEATURES].copy()
    for col in CAT_FEATURES:
        X[col] = X[col].fillna("Not reported").astype(str)
    X["year"] = X["year"].fillna(X["year"].median())
    X["n_levels_model"] = X["n_levels_model"].fillna(1)
    y = data[TARGET].to_numpy(dtype=float)
    study_ids = data["Study_ID"].astype(str).to_numpy()

    args.outdir.mkdir(parents=True, exist_ok=True)
    fold_rows: list[dict] = []
    repeat_oof: list[np.ndarray] = []
    repeat_shap: list[np.ndarray] = []

    for repeat in range(10):
        fold_ids = make_study_grouped_folds(study_ids, repeat, n_splits=5)
        predictions = np.zeros(len(data), dtype=float)
        shap_values = np.zeros((len(data), len(FEATURES)), dtype=float)

        for fold in range(5):
            train = fold_ids != fold
            test = fold_ids == fold
            model = CatBoostRegressor(
                **MODEL_PARAMS,
                random_seed=repeat,
                verbose=0,
                allow_writing_files=False,
                thread_count=1,
            )
            train_pool = Pool(X.loc[train, FEATURES], y[train], cat_features=CAT_FEATURES)
            test_pool = Pool(X.loc[test, FEATURES], cat_features=CAT_FEATURES)
            model.fit(train_pool)
            predictions[test] = model.predict(test_pool)
            labelled_test_pool = Pool(X.loc[test, FEATURES], y[test], cat_features=CAT_FEATURES)
            shap_values[test] = model.get_feature_importance(labelled_test_pool, type="ShapValues")[:, :-1]

            train_predictions = model.predict(Pool(X.loc[train, FEATURES], cat_features=CAT_FEATURES))
            fold_rows.append({
                "Repeat": repeat + 1,
                "Fold": fold + 1,
                "N_Train": int(train.sum()),
                "N_Test": int(test.sum()),
                "N_Train_Studies": int(pd.Series(study_ids[train]).nunique()),
                "N_Test_Studies": int(pd.Series(study_ids[test]).nunique()),
                "Train_R2": float(r2_score(y[train], train_predictions)),
                "Test_R2": float(r2_score(y[test], predictions[test])),
                "Test_RMSE": float(mean_squared_error(y[test], predictions[test]) ** 0.5),
                "Test_MAE": float(mean_absolute_error(y[test], predictions[test])),
            })

        repeat_oof.append(predictions)
        repeat_shap.append(shap_values)

    # Match the original aggregation: average signed SHAP across repeats first,
    # then take the mean absolute contribution across observations.
    shap_mean = np.mean(np.stack(repeat_shap), axis=0)
    mean_abs_shap = np.abs(shap_mean).mean(axis=0)
    importance_pct = mean_abs_shap / mean_abs_shap.sum() * 100
    repeat_pct = []
    for values in repeat_shap:
        imp = np.abs(values).mean(axis=0)
        repeat_pct.append(imp / imp.sum() * 100)
    repeat_pct = np.asarray(repeat_pct)

    importance = pd.DataFrame({
        "Feature": FEATURES,
        "Mean_Absolute_SHAP": mean_abs_shap,
        "Importance_Percent": importance_pct,
        "Repeat_P2_5": np.quantile(repeat_pct, 0.025, axis=0),
        "Repeat_P97_5": np.quantile(repeat_pct, 0.975, axis=0),
    }).sort_values("Importance_Percent", ascending=False)
    folds = pd.DataFrame(fold_rows)
    oof = pd.DataFrame({
        "Plot_point_ID": data["Plot_point_ID"],
        "Study_ID": study_ids,
        "Pollutant_Group": data["poll_group_model"],
        "Observed_Target_Y": y,
        "Mean_OOF_Prediction": np.mean(np.stack(repeat_oof), axis=0),
    })

    importance.to_csv(args.outdir / "Fig4a_SHAP_Importance.csv", index=False)
    folds.to_csv(args.outdir / "Fig4a_Study_Grouped_Cross_Validation.csv", index=False)
    oof.to_csv(args.outdir / "Fig4a_OOF_Predictions.csv", index=False)
    summary = {
        "Input_File": str(args.input),
        "Input_Worksheet": source_sheet,
        "N_Records": int(len(data)),
        "N_Studies": int(data["Study_ID"].nunique()),
        "Pollutant_Record_Counts": group_counts,
        "Cross_Validation_Repeats": 10,
        "Folds_Per_Repeat": 5,
        "Model_Parameters": MODEL_PARAMS,
        "Mean_Fold_Test_R2": float(folds["Test_R2"].mean()),
        "SD_Fold_Test_R2": float(folds["Test_R2"].std(ddof=1)),
        "Mean_Train_R2": float(folds["Train_R2"].mean()),
        "Mean_Test_RMSE": float(folds["Test_RMSE"].mean()),
        "Mean_Test_MAE": float(folds["Test_MAE"].mean()),
        "SHAP_Importance": importance.to_dict(orient="records"),
    }
    (args.outdir / "Fig4a_Model_Summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

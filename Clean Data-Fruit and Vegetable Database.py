# -*- coding: utf-8 -*-
"""
v13_fast_onecall_lowcost 的 JSONL -> Excel。

一行 = 一个真实 result：
condition × sampling_time × matrix_tissue × endpoint_name。

不修改旧JSON/旧Excel。
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd


HERE = Path(__file__).resolve().parent
TOKENS = {"NR", "NA", "ND"}


def find_schema(path: str = "") -> Path:
    if path:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"字段表不存在：{p}")
        return p

    candidates = sorted(
        [p for p in HERE.rglob("Table_S1_字段表*.xlsx") if not p.name.startswith("~$")],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError("找不到 Table S1 字段表，请使用 --schema。")
    return candidates[0]


def load_schema(xlsx: Path) -> Tuple[Dict[str, List[str]], pd.DataFrame]:
    df = pd.read_excel(xlsx, sheet_name="Table S1" if "Table S1" in pd.ExcelFile(xlsx).sheet_names else 0)

    # 第一行通常是字段说明表头；若已经被pandas读取为正常列名，也兼容。
    if "Field Name / 字段名" in df.columns:
        pass
    else:
        # 常见情况：第一行被读成标题，重新无header读取
        raw = pd.read_excel(xlsx, sheet_name="Table S1" if "Table S1" in pd.ExcelFile(xlsx).sheet_names else 0, header=None)
        if raw.shape[0] >= 3 and raw.shape[1] >= 4:
            header = raw.iloc[1].tolist()
            raw2 = raw.iloc[2:].copy()
            raw2.columns = header
            df = raw2

    cols = list(df.columns)
    if len(cols) < 4:
        raise RuntimeError("字段表列数不足。")

    # 按实际表头定位
    def find_col(names):
        for n in names:
            if n in df.columns:
                return n
        return None

    level_col = find_col(["Extraction Level / 提取层级"])
    field_col = find_col(["Field Name / 字段名"])
    cn_col = find_col(["Chinese Name / 中文名"])

    if not level_col or not field_col:
        # 最后兜底：前四列
        work = df.copy()
        work.columns = ["category", "level", "field", "cn"] + list(work.columns[4:])
        level_col, field_col, cn_col = "level", "field", "cn"
    else:
        work = df.copy()

    schema = {"study": [], "condition": [], "result": []}

    for _, r in work.iterrows():
        level = str(r.get(level_col, "") or "").strip().lower()
        field = str(r.get(field_col, "") or "").strip()
        if not field or field.lower() in {"nan", "none"}:
            continue

        if level.startswith("study"):
            schema["study"].append(field)
        elif level.startswith("condition"):
            schema["condition"].append(field)
        elif level.startswith("result"):
            schema["result"].append(field)

    schema["study"] = [x for x in schema["study"] if x != "study_id"]
    schema["condition"] = [x for x in schema["condition"] if x != "condition_id"]
    schema["result"] = [x for x in schema["result"] if x != "record_id"]

    return schema, df


def norm(v: Any) -> str:
    if v is None:
        return "NR"
    if isinstance(v, float) and pd.isna(v):
        return "NR"
    s = re.sub(r"\s+", " ", str(v).replace("\u00a0", " ")).strip()
    if not s or s.lower() in {"nan", "none"}:
        return "NR"
    if s.upper() in TOKENS:
        return s.upper()
    return s


def read_jsonl(path: Path) -> Tuple[List[Dict[str, Any]], List[str]]:
    objs = []
    errors = []
    with path.open("r", encoding="utf-8-sig") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
                if isinstance(obj, dict):
                    objs.append(obj)
            except Exception as e:
                errors.append(f"第{line_no}行：{e}")
    return objs, errors


def study_table(objs, schema):
    fields = ["study_id"] + schema["study"]
    rows = []
    for obj in objs:
        sid = norm(obj.get("study_id"))
        meta = obj.get("study_meta") if isinstance(obj.get("study_meta"), dict) else {}
        row = {f: "NR" for f in fields}
        row["study_id"] = sid
        for f in schema["study"]:
            row[f] = norm(meta.get(f))
        rows.append(row)
    return pd.DataFrame(rows, columns=fields)


def condition_table(objs, schema):
    fields = ["study_id", "condition_id", "condition_index"] + schema["condition"]
    rows = []

    for obj in objs:
        sid = norm(obj.get("study_id"))
        conds = obj.get("conditions")
        if not isinstance(conds, list):
            continue

        for c in conds:
            if not isinstance(c, dict):
                continue
            try:
                ci = int(c.get("condition_index"))
            except Exception:
                continue

            row = {f: "NR" for f in fields}
            row["study_id"] = sid
            row["condition_index"] = ci
            row["condition_id"] = c.get("condition_id") or f"{sid}-C{ci:03d}"

            for f in schema["condition"]:
                row[f] = norm(c.get(f))
            rows.append(row)

    return pd.DataFrame(rows, columns=fields)


def unified_table(objs, schema):
    fields = (
        ["study_id", "condition_id", "record_id"]
        + schema["study"] + schema["condition"] + schema["result"]
    )
    rows = []
    qc = []

    for obj in objs:
        sid = norm(obj.get("study_id"))
        meta = obj.get("study_meta") if isinstance(obj.get("study_meta"), dict) else {}

        conds = obj.get("conditions")
        results = obj.get("results")
        if not isinstance(conds, list):
            conds = []
        if not isinstance(results, list):
            results = []

        cmap = {}
        for c in conds:
            if not isinstance(c, dict):
                continue
            try:
                ci = int(c.get("condition_index"))
            except Exception:
                continue
            cmap[ci] = c

        if not results:
            qc.append([sid, "无Result", "", "该文献没有生成Result记录"])
            continue

        for ri, r in enumerate(results, 1):
            if not isinstance(r, dict):
                continue

            try:
                ci = int(r.get("condition_index"))
            except Exception:
                qc.append([sid, "Result无condition_index", f"R{ri}", "无法定位Condition"])
                continue

            c = cmap.get(ci)
            if c is None:
                qc.append([sid, "Result找不到Condition", f"R{ri}", f"condition_index={ci}"])
                continue

            row = {f: "NR" for f in fields}
            row["study_id"] = sid
            row["condition_id"] = c.get("condition_id") or f"{sid}-C{ci:03d}"
            row["record_id"] = r.get("record_id") or f"{sid}-R{ri:04d}"

            for f in schema["study"]:
                row[f] = norm(meta.get(f))
            for f in schema["condition"]:
                row[f] = norm(c.get(f))
            for f in schema["result"]:
                row[f] = norm(r.get(f))

            # 结果级QC
            if row.get("endpoint_name", "NR").upper() == "NR":
                qc.append([sid, "结果指标缺失", row["record_id"], "endpoint_name=NR"])
            if row.get("source_location", "NR").upper() == "NR":
                qc.append([sid, "来源位置缺失", row["record_id"], "source_location=NR"])

            numeric = [
                x for x in [
                    "result_value", "result_min", "result_max",
                    "control_mean", "treatment_mean",
                    "reported_percent_change"
                ] if x in row
            ]
            if numeric and all(norm(row[x]).upper() in TOKENS for x in numeric):
                qc.append([
                    sid, "无直接数值", row["record_id"],
                    "可能是定性结果/图无数字/表格未成功映射，建议核查"
                ])

            rows.append(row)

    return pd.DataFrame(rows, columns=fields), qc


def field_stats(df):
    rows = []
    for f in df.columns:
        s = df[f].map(norm).str.upper()
        total = len(df)
        nr = int((s == "NR").sum())
        na = int((s == "NA").sum())
        nd = int((s == "ND").sum())
        reported = total - nr - na - nd
        applicable = total - na

        rows.append([
            f, total, reported, nr, na, nd,
            round(100 * reported / total, 1) if total else 0,
            round(100 * nr / applicable, 1) if applicable else 0,
        ])

    return pd.DataFrame(
        rows,
        columns=[
            "field", "total_rows", "reported_count",
            "NR_count", "NA_count", "ND_count",
            "reported_percent", "NR_percent_among_applicable"
        ],
    )


def summary_table(objs, unified):
    ucounts = {}
    if not unified.empty and "study_id" in unified.columns:
        ucounts = unified.groupby("study_id").size().to_dict()

    rows = []
    for obj in objs:
        sid = norm(obj.get("study_id"))
        conds = obj.get("conditions")
        results = obj.get("results")
        conds = conds if isinstance(conds, list) else []
        results = results if isinstance(results, list) else []

        rows.append([
            sid, len(conds), len(results), ucounts.get(sid, 0),
            obj.get("_model", ""),
            obj.get("_pages", ""),
            obj.get("_evidence_chars", ""),
            obj.get("_finish_reason", ""),
        ])

    return pd.DataFrame(
        rows,
        columns=[
            "study_id", "condition_count", "result_count_in_json",
            "result_rows_in_excel", "model", "pages", "evidence_chars",
            "finish_reason"
        ],
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--schema", default="")
    args = ap.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        raise FileNotFoundError(f"JSONL不存在：{input_path}")

    schema_path = find_schema(args.schema)
    schema, schema_df = load_schema(schema_path)

    objs, json_errors = read_jsonl(input_path)
    if not objs:
        raise RuntimeError("JSONL中没有有效文献记录。")

    study_df = study_table(objs, schema)
    condition_df = condition_table(objs, schema)
    unified_df, qc = unified_table(objs, schema)

    if unified_df.empty:
        # 即使全部是无result，也输出Study/Condition/QC，方便诊断
        print("[WARN] 统一母表没有Result行。")

    stats_df = field_stats(unified_df) if not unified_df.empty else pd.DataFrame()
    result_stats = (
        stats_df[stats_df["field"].isin(schema["result"])].copy()
        if not stats_df.empty else pd.DataFrame()
    )
    summary_df = summary_table(objs, unified_df)

    qc_rows = list(qc)
    for e in json_errors:
        qc_rows.append(["JSONL", "JSON解析错误", "", e])

    qc_df = pd.DataFrame(
        qc_rows,
        columns=["study_id", "问题类型", "record_or_location", "问题"]
    )

    # 数值待核查：来源像Table/Figure/Supplement，但定量字段全为空。
    numeric_fields = [
        x for x in [
            "result_value", "result_min", "result_max",
            "control_mean", "treatment_mean",
            "reported_percent_change"
        ] if x in unified_df.columns
    ]

    if not unified_df.empty and numeric_fields:
        src = unified_df["source_location"].map(norm)
        missing_all = unified_df[numeric_fields].apply(
            lambda col: col.map(norm).str.upper().isin(TOKENS)
        ).all(axis=1)
        table_like = src.str.contains(
            r"Table|Supplement|Fig\.|Figure",
            case=False, regex=True, na=False
        )

        cols = [
            "study_id", "condition_id", "record_id",
            "sampling_time", "matrix_tissue", "endpoint_name",
            "result_value", "control_mean", "treatment_mean",
            "result_unit", "source_location"
        ]
        cols = [c for c in cols if c in unified_df.columns]
        numeric_gap_df = unified_df.loc[missing_all & table_like, cols].copy()
    else:
        numeric_gap_df = pd.DataFrame()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        unified_df.to_excel(writer, sheet_name="统一母表", index=False)
        study_df.to_excel(writer, sheet_name="Study表", index=False)
        condition_df.to_excel(writer, sheet_name="Condition表", index=False)
        summary_df.to_excel(writer, sheet_name="研究汇总", index=False)
        stats_df.to_excel(writer, sheet_name="字段非空统计", index=False)
        result_stats.to_excel(writer, sheet_name="Results字段统计", index=False)
        qc_df.to_excel(writer, sheet_name="QC_问题清单", index=False)
        numeric_gap_df.to_excel(writer, sheet_name="数值待核查", index=False)
        schema_df.to_excel(writer, sheet_name="字段说明", index=False)

    print("========== v13 JSONL → Excel ==========")
    print("输入JSONL：", input_path)
    print("字段表：", schema_path)
    print("输出Excel：", output_path)
    print("文献数：", len(objs))
    print("Study行数：", len(study_df))
    print("Condition行数：", len(condition_df))
    print("Result行数：", len(unified_df))
    print("QC问题数：", len(qc_df))
    print("数值待核查行：", len(numeric_gap_df))
    if not result_stats.empty:
        print("\nResults字段覆盖率：")
        print(result_stats[["field", "reported_count", "reported_percent"]].to_string(index=False))
    print("\n完成。旧JSON/旧Excel不会被修改。")


if __name__ == "__main__":
    main()

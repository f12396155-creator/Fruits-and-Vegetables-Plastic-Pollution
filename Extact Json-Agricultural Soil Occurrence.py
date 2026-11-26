# -*- coding: utf-8 -*-
"""
soil_occurrence_v1_fast_complete
------------------------
目标：速度快、费用低、结果尽可能完整。

设计：
1. 每篇论文默认 1 次 DeepSeek API。
2. 使用 DeepSeek JSON mode + thinking disabled，降低调用失败率和等待时间。
3. PDF 本地先筛选：Methods + Results首尾 + 主文表格/数值页 + 补充表格/数值页。
4. Results 使用紧凑别名输出，节省 completion tokens，把空间留给真实数值。
5. 一篇论文完成后立即追加写入 JSONL，旧文件完全不碰。
6. 不自动二次补提、不自动猜图、不做笛卡尔积。
7. --resume 可在同一个“新输出文件”上断点续跑。

注意：
- “每个字段都有数据”不能强行实现。论文没有报告只能是 NR/NA/ND。
- 真正目标是：论文明确报告的事实尽可能全部提取。
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Tuple

import requests

try:
    import pymupdf as fitz
except Exception:
    try:
        import fitz
    except Exception:
        fitz = None

try:
    from dotenv import load_dotenv
except Exception:
    load_dotenv = None

try:
    import openpyxl
except Exception:
    openpyxl = None


HERE = Path(__file__).resolve().parent
DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-flash"

# 速度/完整度平衡：
# 输入尽量控制在约 50k 字符左右；表格/结果优先。
MAX_FRONT_CHARS = 2500
MAX_METHODS_CHARS = 4500
MAX_RESULTS_CHARS = 18000
MAX_MAIN_TABLE_CHARS = 12500
MAX_SUPP_TABLE_CHARS = 12000
MAX_TOTAL_EVIDENCE_CHARS = 52000

# 紧凑JSON输出，避免以前 12k/14k tokens 导致长文献被截断。
MAX_OUTPUT_TOKENS = 8500
TIMEOUT = 120
MAX_RETRIES = 1
RESCUE_MAX_TOKENS = 6500

LIB_RE = re.compile(r"(?i)(?<![A-Z0-9])LIB[-_ ]?(\d{1,8})(?!\d)")
MISSING = {"NR", "NA", "ND"}

# result字段的短别名：模型输出短键，Python还原成原始字段名。
RESULT_ALIAS = {
    "t": "sampling_time",
    "m": "matrix_tissue",
    "e": "endpoint_name",
    "a": "measured_analyte",
    "am": "assay_method",
    "s": "result_scope",
    "vt": "result_value_type",
    "v": "result_value",
    "lo": "result_min",
    "hi": "result_max",
    "u": "result_unit",
    "b": "result_basis",
    "n": "result_n",
    "cm": "control_mean",
    "ce": "control_error_value",
    "cet": "control_error_type",
    "cn": "control_n",
    "tm": "treatment_mean",
    "te": "treatment_error_value",
    "tet": "treatment_error_type",
    "tn": "treatment_n",
    "pc": "reported_percent_change",
    "mc": "measurement_category",
    "aset": "analyte_set",
    "sl": "statistical_level",
    "erv": "result_error_value",
    "ert": "result_error_type",
    "dl": "detection_limit",
    "ql": "quantification_limit",
    "ps": "particle_detection_lower_size",
    "src": "source_location",
    "note": "extraction_note",
}


def die(msg: str) -> None:
    raise RuntimeError(msg)


def find_schema(path: str = "") -> Path:
    if path:
        p = Path(path)
        if not p.exists():
            die(f"字段表不存在：{p}")
        return p

    preferred = HERE / "Table_SoilOccurrence_S1_fields_v1.xlsx"
    if preferred.exists():
        return preferred
    candidates = sorted(
        [p for p in HERE.rglob("Table_SoilOccurrence_S1_fields*.xlsx") if not p.name.startswith("~$")],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        die("找不到 Table S1 字段表，请使用 --schema。")
    return candidates[0]


def load_schema(xlsx_path: Path) -> Tuple[Dict[str, List[str]], Dict[str, str]]:
    if openpyxl is None:
        die("缺少 openpyxl，请安装 openpyxl。")

    wb = openpyxl.load_workbook(xlsx_path, read_only=True, data_only=True)
    ws = wb["Table S1"] if "Table S1" in wb.sheetnames else wb[wb.sheetnames[0]]

    rows = list(ws.iter_rows(values_only=True))
    if len(rows) < 3:
        wb.close()
        die("字段表内容不足。")

    headers = [str(x).strip() if x is not None else "" for x in rows[1]]
    idx = {v: i for i, v in enumerate(headers) if v}
    level_col = idx.get("Extraction Level / 提取层级")
    field_col = idx.get("Field Name / 字段名")
    cn_col = idx.get("Chinese Name / 中文名")
    rule_col = idx.get("Exact Extraction Rule / 怎么提、提成什么样")

    if level_col is None or field_col is None:
        wb.close()
        die("字段表缺少 Extraction Level / 提取层级 或 Field Name / 字段名。")

    schema = {"study": [], "condition": [], "result": []}
    desc: Dict[str, str] = {}

    for row in rows[2:]:
        level = str(row[level_col] or "").strip().lower()
        field = str(row[field_col] or "").strip()
        cn = str(row[cn_col] or "").strip() if cn_col is not None else ""
        rule = str(row[rule_col] or "").strip() if rule_col is not None else ""

        if not field or field.lower() in {"nan", "none"}:
            continue

        if level.startswith("study"):
            schema["study"].append(field)
        elif level.startswith("condition"):
            schema["condition"].append(field)
        elif level.startswith("result"):
            schema["result"].append(field)

        if rule.lower() in {"nan", "none"}:
            rule = ""
        desc[field] = f"{cn}；{rule}" if rule else cn

    wb.close()

    schema["study"] = [x for x in schema["study"] if x not in {"study_id"}]
    schema["condition"] = [x for x in schema["condition"] if x not in {"condition_id"}]
    schema["result"] = [x for x in schema["result"] if x not in {"record_id"}]

    return schema, desc


def study_id_from_name(name: str) -> str | None:
    m = LIB_RE.search(name)
    return f"LIB-{int(m.group(1)):05d}" if m else None


def source_role(path: Path) -> str:
    n = path.name.lower()
    marks = [
        "supplement", "supplementary", "supporting", "appendix",
        "esi", "_si", "si_", "补充", "附录"
    ]
    return "SUPPLEMENT" if any(x in n for x in marks) else "MAIN"


def clean_text(t: str) -> str:
    t = t or ""
    t = t.replace("\x00", "").replace("\u00ad", "")
    t = t.replace("\r\n", "\n").replace("\r", "\n")
    return t


def read_pdf_pages(path: Path) -> List[str]:
    if fitz is None:
        die("缺少 PyMuPDF，请安装 pymupdf。")
    doc = fitz.open(str(path))
    pages = []
    try:
        for i, page in enumerate(doc, 1):
            txt = clean_text(page.get_text("text") or "")
            pages.append(f"\n--- SOURCE FILE: {path.name} ---\n--- PAGE {i} ---\n{txt}")
    finally:
        doc.close()
    return pages


def remove_reference_tail(text: str) -> str:
    low = text.lower()
    hits = []
    for pat in [
        r"^\s*references\s*$",
        r"^\s*bibliography\s*$",
        r"^\s*acknowledg(?:e)?ments\s*$",
    ]:
        m = re.search(pat, low, flags=re.M)
        if m:
            hits.append(m.start())
    return text[:min(hits)] if hits else text


def section(text: str, starts: List[str], ends: List[str]) -> str:
    low = text.lower()
    start_hits = []
    for pat in starts:
        for m in re.finditer(pat, low, flags=re.I | re.M):
            start_hits.append(m.start())
    if not start_hits:
        return ""
    st = min(start_hits)

    end_hits = []
    for pat in ends:
        m = re.search(pat, low[st + 1:], flags=re.I | re.M)
        if m:
            end_hits.append(st + 1 + m.start())
    en = min(end_hits) if end_hits else len(text)
    return text[st:en]


def page_score(page: str, supplement: bool = False) -> int:
    low = page.lower()
    score = 0

    # 强表格信号
    if re.search(r"\btable\s*[s]?\s*\d+[a-z]?\b", low):
        score += 12
    if "supplementary table" in low or "supporting table" in low:
        score += 14
    if re.search(r"\bfig(?:ure)?\.?\s*\d+[a-z]?\b", low):
        score += 5

    # 数值信号
    if "mean" in low:
        score += 3
    if re.search(r"\b(?:sd|se|sem|median)\b", low):
        score += 3
    if re.search(r"\bp\s*[<=>]\s*0?\.\d+", low):
        score += 4
    if "±" in page or "+/-" in low:
        score += 4
    if re.search(r"\d+(?:\.\d+)?\s*(?:%|mg/kg|µg/kg|ug/kg|mg/g|µg/g|ug/g|mg/l|mg/L|cm|mm|days?|weeks?|hours?|h)\b", low):
        score += 4

    # 结果主题
    for term in [
        "abundance", "concentration", "content", "residue", "accumulation",
        "biomass", "fresh weight", "dry weight", "chlorophyll",
        "sod", "pod", "cat", "mda", "h2o2", "root", "shoot", "yield",
        "microplastic", "plastic", "polymer", "agricultural soil", "soil abundance",
        "particles/kg", "items/kg", "mass concentration", "phthalate", "paes",
        "detection limit", "limit of detection", "lod", "loq", "dry weight",
        "wet weight", "soil sample", "field soil", "total paes", "sigma-16"
    ]:
        if term in low:
            score += 1

    # 纯参考文献页强降权
    if "references" in low or "bibliography" in low:
        score -= 20

    if supplement:
        score += 1

    # 页面数字密度：帮助识别没有清晰Table标题的PDF表格页
    nums = re.findall(r"(?<!\w)-?\d+(?:\.\d+)?", page)
    if len(nums) >= 12:
        score += 4
    if len(nums) >= 25:
        score += 4

    return score


def select_pages(pages: List[str], max_chars: int, supplement: bool = False) -> str:
    if not pages or max_chars <= 0:
        return ""

    scored = [(page_score(p, supplement), i, p) for i, p in enumerate(pages)]
    scored.sort(key=lambda x: (-x[0], x[1]))

    selected = set()
    total = 0

    for score, i, p in scored:
        if score <= 0:
            continue

        # 表格页尽量带相邻页，避免跨页表格只取半页。
        candidates = [i]
        if score >= 10:
            candidates += [i - 1, i + 1]

        for j in candidates:
            if j < 0 or j >= len(pages) or j in selected:
                continue
            q = pages[j]
            if total + len(q) > max_chars:
                continue
            selected.add(j)
            total += len(q)

        if total >= max_chars:
            break

    return "".join(pages[i] for i in sorted(selected))


def head_tail(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    a = max_chars // 2
    b = max_chars - a
    return text[:a] + "\n...[RESULTS MIDDLE OMITTED FOR SPEED]...\n" + text[-b:]


def build_evidence(paths: List[Path]) -> Dict[str, Any]:
    main_pages, supp_pages = [], []

    for p in paths:
        pages = read_pdf_pages(p)
        if source_role(p) == "SUPPLEMENT":
            supp_pages.extend(pages)
        else:
            main_pages.extend(pages)

    main_full = remove_reference_tail("".join(main_pages))
    supp_full = remove_reference_tail("".join(supp_pages))

    methods = section(
        main_full,
        [r"^\s*(?:\d+[\.\)]?\s*)?(?:materials\s+and\s+methods|methods|methodology|experimental\s+design|experimental\s+section)\b"],
        [r"^\s*(?:\d+[\.\)]?\s*)?(?:results|results\s+and\s+discussion)\b"],
    )
    if not methods:
        methods = main_full[:MAX_METHODS_CHARS]

    results = section(
        main_full,
        [r"^\s*(?:\d+[\.\)]?\s*)?(?:results|results\s+and\s+discussion)\b"],
        [r"^\s*(?:\d+[\.\)]?\s*)?(?:discussion|conclusions?|conclusion\s+and\s+perspectives)\b"],
    )
    if not results:
        results = main_full[int(len(main_full) * 0.35):]

    # 结果正文首尾 + 表格/数值密集页；避免把一整篇讨论/参考文献送进模型。
    result_part = head_tail(results, MAX_RESULTS_CHARS)
    main_tables = select_pages(main_pages, MAX_MAIN_TABLE_CHARS, False)
    supp_tables = select_pages(supp_pages, MAX_SUPP_TABLE_CHARS, True)

    front = "".join(main_pages[:5])[:MAX_FRONT_CHARS]

    evidence = (
        "===== ARTICLE FRONT =====\n" + front +
        "\n===== METHODS =====\n" + methods[:MAX_METHODS_CHARS] +
        "\n===== RESULTS =====\n" + result_part +
        "\n===== MAIN TABLE/FIGURE/Numeric PAGES =====\n" + main_tables +
        "\n===== SUPPLEMENT TABLE/FIGURE/Numeric PAGES =====\n" + supp_tables
    )

    # 最终硬上限，优先保留结果与表格。
    if len(evidence) > MAX_TOTAL_EVIDENCE_CHARS:
        evidence = (
            "===== ARTICLE FRONT =====\n" + front[:1800] +
            "\n===== METHODS =====\n" + methods[:4000] +
            "\n===== RESULTS =====\n" + head_tail(results, 18000) +
            "\n===== MAIN TABLE/FIGURE/Numeric PAGES =====\n" + main_tables[:13500] +
            "\n===== SUPPLEMENT TABLE/FIGURE/Numeric PAGES =====\n" + supp_tables[:12000]
        )
        evidence = evidence[:MAX_TOTAL_EVIDENCE_CHARS]

    return {
        "text": evidence,
        "pages": len(main_pages) + len(supp_pages),
        "main_pages": len(main_pages),
        "supp_pages": len(supp_pages),
        "chars": len(evidence),
    }


def field_guide(schema: Dict[str, List[str]], desc: Dict[str, str]) -> str:
    lines = []
    lines.append("[STUDY FIELDS]")
    for f in schema["study"]:
        lines.append(f"{f} = {desc.get(f, f)}")

    lines.append("\n[CONDITION FIELDS]")
    for f in schema["condition"]:
        lines.append(f"{f} = {desc.get(f, f)}")

    lines.append("\n[RESULT SHORT KEYS]")
    lines.extend([
        "t=sampling_time, m=matrix_tissue, e=endpoint_name, a=measured_analyte, aset=analyte_set, "
        "am=assay_method, s=result_scope, sl=statistical_level, mc=measurement_category, "
        "vt=result_value_type, v=result_value, lo=result_min, hi=result_max, u=result_unit, "
        "b=result_basis, n=result_n, erv=result_error_value, ert=result_error_type, "
        "dl=detection_limit, ql=quantification_limit, ps=particle_detection_lower_size, "
        "src=source_location, note=extraction_note"
    ])
    return "\n".join(lines)


SYSTEM_PROMPT = r"""
You are a high-recall structured data extractor for a database of measured plastic contamination in real-world agricultural soils. The immediate downstream task is to reconstruct empirical soil concentration distributions for microplastic abundance, microplastic mass concentration, and plasticizer concentrations. Do not write a narrative summary.

CORE GOAL
Extract every directly reported, relevant soil measurement from the current paper and its supplied supplementary files while minimizing unnecessary output. Completeness of eligible concentration records and traceability take priority over unrelated study details. Never invent, estimate, calculate, or infer a value that the paper does not report.

MISSINGNESS
NR = applicable but not reported; NA = structurally not applicable; ND = explicitly not detected. Preserve <LOD, <LOQ, inequality signs, ranges, and censoring exactly. Never replace ND or censored values with zero.

STUDY AND CONDITION RULES
- Study metadata: publication_year, title, doi, study_location.
- A condition is one distinct sampling stratum that the paper actually distinguishes, such as a site/plot, land-use category, soil depth, sampling period, or explicitly separate composite sample. Do not split merely because endpoint or analyte differs.
- Preserve the authors' explicit site labels. Never invent sampling_site_id values. If the table has separate site rows, represent them as separate conditions when the source identifies them.
- Classify occurrence_status using exactly one of: field_occurrence, ambient_soil_sample, controlled_spiking, incubation_experiment, other, unclear. Field/ambient environmental measurements are potential environmental-reference data. Artificially spiked/amended or incubation measurements must be clearly marked and must not be mistaken for ambient occurrence data.
- Do not assume that every soil measurement is an environmental field measurement. Read Methods and table notes. If uncertain, use unclear and explain briefly in extraction_note.
- Do not create conditions from values cited from other papers in the Introduction or Discussion.

RESULT RULES
- A result row represents one reported combination of condition × sampling_time × matrix_tissue × endpoint_name × measured_analyte × statistical_level. Keep different analytes, total-vs-component results, and distinct sampling strata separate.
- Extract all explicit relevant numeric soil results in the available evidence, including rows in main tables and supplementary tables. Do not only select representative values.
- Prioritize these measurement_category values:
  1) MP_number_abundance: particle/item counts per soil mass or area;
  2) MP_mass_concentration: microplastic mass per soil mass;
  3) total_PAE_concentration: explicitly reported sum of phthalate esters;
  4) individual_PAE_concentration: one named phthalate ester;
  5) other_plastic_additive_concentration;
  6) other_soil_result;
  7) unclear.
- Do not infer category from a unit alone if the endpoint is ambiguous. Use unclear if necessary.
- Distinguish total MPs from polymer-specific counts; distinguish total PAEs (e.g. Sigma-6 PAEs, Sigma-16 PAEs) from individual compounds. Never sum individual analytes yourself and never treat a total and its components as independent values in the same threshold distribution.
- Do not combine results with different units, weight bases, sample strata, or statistical levels.
- Preserve the reported statistical_level as one of: individual_sample, site_mean, land_use_mean, regional_mean, study_summary, range_only, pooled_sample, unclear. Do not treat a study summary and its component site-level values as independent observations in the same distribution. If the paper reports only a mean, do not create individual samples from mean +/- SD.
- result_value contains a directly reported main value (mean, median, single value, or censored value). If only an explicit range is reported, set result_value to NR and fill result_min/result_max. Fill result_min/result_max only when explicitly reported as a range or bounds; never calculate them from mean and error.
- Copy error values and type only when reported. Use result_error_type such as SD, SE, SEM, CI95, IQR, range, other, NR, or NA. Do not convert SE to SD.
- result_n is the independent sample size for that reported statistic; do not count technical injections as independent samples.
- Copy original units and basis exactly. Do not perform unit conversions. Record DW/FW, dry soil/wet soil, per area/per volume, and denominator basis in result_basis when reported. If unknown, use NR.
- Copy LOD and LOQ into detection_limit and quantification_limit if reported. Copy the analytical lower particle-size detection/identification limit into particle_detection_lower_size; do not confuse it with the observed particle-size range.
- source_location is required wherever possible and should begin with Main: or Supplement: and identify table/figure/section/page. Prefer exact table values over repeated prose values. If the same value appears in a table and text, retain it once.
- Do not estimate numeric values from unlabeled figure marks. If a relevant figure is the only source and has no explicit numeric labels, record the qualitative/availability fact in extraction_note but leave numeric fields NR.
- If the paper reports field/ambient concentrations and separate laboratory-spiked concentrations, extract and classify them separately.

OUTPUT COMPLETENESS AND COST CONTROL
- Extract all relevant concentration rows present in the provided evidence, not all unrelated plant physiology endpoints.
- Avoid verbose prose. Use compact JSON and short keys for result fields.
- Do not output NR keys; the cleaner fills missing fields with NR. Explicit NA and ND must be retained.
- The result aliases are: t=sampling_time, m=matrix_tissue, e=endpoint_name, a=measured_analyte, aset=analyte_set, am=assay_method, s=result_scope, sl=statistical_level, mc=measurement_category, vt=result_value_type, v=result_value, lo=result_min, hi=result_max, u=result_unit, b=result_basis, n=result_n, erv=result_error_value, ert=result_error_type, dl=detection_limit, ql=quantification_limit, ps=particle_detection_lower_size, src=source_location, note=extraction_note.
- Condition fields must use the exact field names supplied in the field guide. Each condition uses i as its condition_index; each result uses c to refer to that condition_index.
- Output one valid JSON object only, no Markdown fences, with this structure:
{"study":{...},"conditions":[{"i":1,"study_design":"...","occurrence_status":"field_occurrence",...}],"results":[{"c":1,"t":"...","m":"soil","e":"microplastic abundance","mc":"MP_number_abundance","sl":"site_mean","vt":"mean","v":1250,"u":"items/kg DW","b":"DW","n":12,"src":"Main: Table 2"}],"questions":[]}
"""


RESCUE_SYSTEM_PROMPT = r"""
You repair or complete a soil-occurrence extraction using only the supplied paper evidence. Prioritize explicit soil concentration records for microplastic number abundance, microplastic mass concentration, total PAEs, and individual PAEs. Read relevant tables row by row. Never invent values, estimate unlabeled graph points, calculate totals, or convert units. Preserve the exact unit, weight basis, statistical level, sample size, LOD/LOQ, and source location. Clearly classify ambient field measurements versus controlled spiking/incubation. Output one compact valid JSON object only: {"study":{...},"conditions":[{"i":1,...}],"results":[{"c":1,"t":"...","m":"soil","e":"...","mc":"...","sl":"...","v":...,"u":"...","src":"Main: Table 2"}],"questions":[]}.
"""

def _find_json_section(text: str, key: str):
    pat = re.compile(r'"' + re.escape(key) + r'"\s*:\s*([\[\{])')
    m = pat.search(text)
    if not m:
        return None, None, None
    start = m.start(1)
    opener = m.group(1)
    closer = ']' if opener == '[' else '}'
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == '\\':
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
            continue
        if ch == opener:
            depth += 1
        elif ch == closer:
            depth -= 1
            if depth == 0:
                return text[start:i+1], opener, i+1
    return text[start:], opener, None

def _parse_complete_array_objects(array_text: str):
    if not array_text:
        return []
    body = array_text[1:] if array_text.startswith('[') else array_text
    dec = json.JSONDecoder()
    out = []
    pos = 0
    while pos < len(body):
        while pos < len(body) and body[pos] in ' \t\r\n,':
            pos += 1
        if pos >= len(body):
            break
        try:
            obj, end = dec.raw_decode(body, pos)
        except Exception:
            break
        if isinstance(obj, dict):
            out.append(obj)
        pos = end
    return out

def salvage_truncated_json(content: str):
    try:
        return json.loads(content), False
    except Exception:
        pass

    study_text, _, _ = _find_json_section(content, "study")
    cond_text, _, _ = _find_json_section(content, "conditions")
    res_text, _, _ = _find_json_section(content, "results")

    study, conditions, results = {}, [], []

    if study_text:
        try:
            x = json.loads(study_text)
            if isinstance(x, dict):
                study = x
        except Exception:
            pass

    if cond_text:
        try:
            x = json.loads(cond_text)
            if isinstance(x, list):
                conditions = [z for z in x if isinstance(z, dict)]
        except Exception:
            conditions = _parse_complete_array_objects(cond_text)

    if res_text:
        try:
            x = json.loads(res_text)
            if isinstance(x, list):
                results = [z for z in x if isinstance(z, dict)]
        except Exception:
            results = _parse_complete_array_objects(res_text)

    if not study and not conditions and not results:
        return None, False

    return {"study": study, "conditions": conditions, "results": results, "questions": []}, True


def build_prompt(sid: str, paths: List[Path], schema: Dict[str, List[str]], desc: Dict[str, str], evidence: Dict[str, Any]) -> str:
    guide = field_guide(schema, desc)
    return (
        f"Study ID: {sid}\n"
        f"Source files: {', '.join(p.name for p in paths)}\n\n"
        f"{guide}\n\n"
        "[TASK RULES]\n"
        "This is a soil-occurrence threshold dataset. Extract every explicit soil concentration row from the supplied Results, tables and supplements; focus on ambient soil MP abundance, MP mass concentration, total PAEs and individual PAEs. Keep total and component records separate, preserve statistical_level and source_location, and mark controlled_spiking/incubation separately from ambient occurrence. Do not calculate or convert values.\n\n"
        "===== PAPER EVIDENCE START =====\n"
        + evidence["text"]
        + "\n===== PAPER EVIDENCE END =====\n"
    )


def deepseek_call(
    api_key: str,
    base_url: str,
    model: str,
    prompt: str,
    timeout: int,
    retries: int,
    max_tokens: int,
    system_prompt: str = SYSTEM_PROMPT,
) -> Dict[str, Any]:
    url = base_url.rstrip("/") + "/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        "response_format": {"type": "json_object"},
        "thinking": {"type": "disabled"},
        "temperature": 0,
        "max_tokens": max_tokens,
        "stream": False,
    }

    last_err = ""
    for attempt in range(1, retries + 2):
        try:
            r = requests.post(url, headers=headers, json=payload, timeout=timeout)
            if r.status_code in (408, 409, 429, 500, 502, 503, 504):
                last_err = f"HTTP {r.status_code}"
                if attempt <= retries:
                    time.sleep(1.5 * attempt)
                    continue
                raise RuntimeError(last_err)
            r.raise_for_status()
            data = r.json()
            choice = (data.get("choices") or [None])[0]
            if not choice:
                raise RuntimeError("API没有返回choices")
            msg = choice.get("message") or {}
            content = (msg.get("content") or "").strip()
            if not content:
                raise RuntimeError("API返回空content")
            return {
                "content": content,
                "usage": data.get("usage") or {},
                "finish_reason": choice.get("finish_reason") or "",
                "model": data.get("model") or model,
            }
        except Exception as e:
            last_err = str(e)
            if attempt <= retries:
                time.sleep(1.5 * attempt)
                continue
            raise RuntimeError(f"DeepSeek调用失败：{last_err}")
    raise RuntimeError(last_err or "DeepSeek调用失败")


def norm(v: Any) -> Any:
    if v is None:
        return "NR"
    if isinstance(v, bool):
        return str(v).lower()
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return v
    s = re.sub(r"\s+", " ", str(v).replace("\u00a0", " ")).strip()
    return s if s else "NR"


def normalize_result(raw: Dict[str, Any]) -> Dict[str, Any]:
    out = {"type": "result"}

    for short, field in RESULT_ALIAS.items():
        if short in raw:
            out[field] = norm(raw[short])
        elif field in raw:
            # 兼容模型偶尔输出完整字段名
            out[field] = norm(raw[field])

    try:
        out["condition_index"] = int(raw.get("c", raw.get("condition_index")))
    except Exception:
        out["condition_index"] = "NR"

    return out


def normalize_condition(raw: Dict[str, Any]) -> Dict[str, Any]:
    out = {"type": "condition"}

    try:
        out["condition_index"] = int(raw.get("i", raw.get("condition_index")))
    except Exception:
        out["condition_index"] = "NR"

    for k, v in raw.items():
        if k in {"i", "condition_index", "type"}:
            continue
        out[k] = norm(v)

    return out


def normalize_obj(sid: str, raw: Dict[str, Any], schema: Dict[str, List[str]], paths: List[Path], evidence: Dict[str, Any], model: str) -> Dict[str, Any]:
    if not isinstance(raw, dict):
        raise RuntimeError("模型返回不是JSON对象")

    study_raw = raw.get("study")
    if not isinstance(study_raw, dict):
        study_raw = {}

    study_meta = {f: norm(study_raw.get(f)) for f in schema["study"]}

    cond_raw = raw.get("conditions")
    if not isinstance(cond_raw, list):
        cond_raw = []

    conditions = []
    seen_ci = set()
    for c in cond_raw:
        if not isinstance(c, dict):
            continue
        cc = normalize_condition(c)
        ci = cc.get("condition_index")
        if not isinstance(ci, int) or ci in seen_ci:
            continue
        seen_ci.add(ci)
        conditions.append(cc)
    conditions.sort(key=lambda x: x["condition_index"])

    result_raw = raw.get("results")
    if not isinstance(result_raw, list):
        result_raw = []

    results = []
    for r in result_raw:
        if not isinstance(r, dict):
            continue
        rr = normalize_result(r)
        if isinstance(rr.get("condition_index"), int):
            results.append(rr)

    # 轻量去重，不合并不同source_location的独立记录。
    seen = set()
    dedup = []
    for r in results:
        key = (
            r.get("condition_index"),
            str(r.get("sampling_time", "")),
            str(r.get("matrix_tissue", "")),
            str(r.get("endpoint_name", "")),
            str(r.get("measurement_category", "")),
            str(r.get("measured_analyte", "")),
            str(r.get("result_scope", "")),
            str(r.get("statistical_level", "")),
            str(r.get("result_value", "")),
            str(r.get("result_min", "")),
            str(r.get("result_max", "")),
            str(r.get("result_unit", "")),
            str(r.get("source_location", "")),
        )
        if key in seen:
            continue
        seen.add(key)
        dedup.append(r)
    results = dedup

    # 程序生成ID；模型不负责。
    idx_to_id = {}
    for i, c in enumerate(conditions, 1):
        cid = f"{sid}-C{i:03d}"
        c["condition_id"] = cid
        idx_to_id[c["condition_index"]] = cid

    for i, r in enumerate(results, 1):
        r["condition_id"] = idx_to_id.get(r.get("condition_index"), "NR")
        r["record_id"] = f"{sid}-R{i:04d}"

    # study_id写到最外层；清洗程序会展开。
    return {
        "study_id": sid,
        "study_meta": study_meta,
        "conditions": conditions,
        "results": results,
        "_model": model,
        "_pages": evidence["pages"],
        "_main_pages": evidence["main_pages"],
        "_supp_pages": evidence["supp_pages"],
        "_evidence_chars": evidence["chars"],
        "_finish_reason": "",
        "_extracted_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "_source_files": [p.name for p in paths],
    }


def load_done_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    done = set()
    with path.open("r", encoding="utf-8-sig") as f:
        for line in f:
            try:
                obj = json.loads(line)
                sid = obj.get("study_id")
                if sid:
                    done.add(str(sid))
            except Exception:
                continue
    return done


def build_rescue_prompt(sid: str, paths: List[Path], schema: Dict[str, List[str]], desc: Dict[str, str], evidence: Dict[str, Any]) -> str:
    result_fields = [
        "sampling_time", "matrix_tissue", "endpoint_name", "measurement_category",
        "measured_analyte", "analyte_set", "assay_method", "result_scope",
        "statistical_level", "result_value_type", "result_value", "result_min",
        "result_max", "result_unit", "result_basis", "result_n",
        "result_error_value", "result_error_type", "detection_limit",
        "quantification_limit", "particle_detection_lower_size", "source_location",
        "extraction_note"
    ]
    return (
        f"Study ID: {sid}\n"
        f"Source files: {', '.join(p.name for p in paths)}\n\n"
        "Repair the extraction using only the paper evidence. Read relevant main and supplementary tables row by row. Prioritize soil microplastic number abundance, microplastic mass concentration, total PAEs and individual PAEs. Keep ambient field measurements separate from controlled spiking/incubation. Preserve original units, weight basis, statistics, sample size, LOD/LOQ and source location. Do not calculate, estimate graph values, sum analytes or convert units.\n\n"
        "CONDITION FIELDS:\n" + ", ".join(schema["condition"]) +
        "\nRESULT FIELDS:\n" + ", ".join(result_fields) +
        "\n\nOutput one compact JSON object only, using result aliases from the system prompt. Example: "
        '{"study":{...},"conditions":[{"i":1,"occurrence_status":"field_occurrence",...}],"results":[{"c":1,"t":"...","m":"soil","e":"...","mc":"MP_number_abundance","sl":"site_mean","v":1250,"u":"items/kg DW","src":"Main: Table 2"}],"questions":[]}'
        "\n\n===== PAPER EVIDENCE =====\n" + evidence["text"]
    )

def build_targeted_prompt(sid: str, paths: List[Path], schema: Dict[str, List[str]], desc: Dict[str, str], evidence: Dict[str, Any], previous_obj: Dict[str, Any]) -> str:
    # Keep this follow-up compact: only call it when the first response hit the output limit.
    existing = []
    for r in (previous_obj.get("results") or []):
        existing.append({k: r.get(k) for k in ["condition_index", "sampling_time", "matrix_tissue", "endpoint_name", "measurement_category", "measured_analyte", "result_scope", "statistical_level", "result_value", "result_min", "result_max", "result_unit", "source_location"]})
    existing_conditions = []
    for c in (previous_obj.get("conditions") or []):
        existing_conditions.append({k: v for k, v in c.items() if k not in {"condition_id"}})
    return (
        f"Study ID: {sid}\nSource files: {', '.join(p.name for p in paths)}\n\n"
        "The first extraction hit the output token limit. Do a targeted completion pass. Do NOT repeat records already represented below. Read the supplied evidence and return only additional soil concentration records that are clearly present but absent from the existing extraction. Preserve existing condition_index values for known sampling strata; create a new condition index only for a genuinely distinct sampling stratum not listed. Never infer, calculate, estimate graph values, or convert units. If no additional eligible record is visible, return an empty results array. Output one compact JSON object only.\n\n"
        "EXISTING CONDITIONS:\n" + json.dumps(existing_conditions, ensure_ascii=False, separators=(",", ":")) +
        "\nEXISTING RESULTS (do not repeat):\n" + json.dumps(existing, ensure_ascii=False, separators=(",", ":")) +
        "\nFIELD GUIDE:\n" + field_guide(schema, desc) +
        "\nPAPER EVIDENCE:\n" + evidence["text"]
    )


def merge_targeted_obj(primary: Dict[str, Any], extra: Dict[str, Any], sid: str) -> Dict[str, Any]:
    # Merge conditions by model condition_index; retain the first non-missing value on conflicts.
    conds = {}
    for source in [primary.get("conditions", []), extra.get("conditions", [])]:
        for c in source:
            idx = c.get("condition_index")
            if not isinstance(idx, int):
                continue
            if idx not in conds:
                conds[idx] = dict(c)
            else:
                for k, v in c.items():
                    if k in {"condition_id"}:
                        continue
                    if norm(conds[idx].get(k)) in {"NR", "NA", "ND", ""} and norm(v) not in {"NR", "NA", "ND", ""}:
                        conds[idx][k] = v
    results = list(primary.get("results", []))
    seen = set()
    def signature(r):
        return tuple(str(r.get(k, "")) for k in ["condition_index", "sampling_time", "matrix_tissue", "endpoint_name", "measurement_category", "measured_analyte", "result_scope", "statistical_level", "result_value", "result_min", "result_max", "result_unit", "source_location"])
    for r in results:
        seen.add(signature(r))
    for r in extra.get("results", []):
        sig = signature(r)
        if sig not in seen:
            results.append(r)
            seen.add(sig)
    out = dict(primary)
    out["conditions"] = [conds[k] for k in sorted(conds)]
    out["results"] = results
    # Regenerate IDs after merging; preserve condition_index links.
    idx_to_id = {}
    for i, c in enumerate(out["conditions"], 1):
        c["condition_id"] = f"{sid}-C{i:03d}"
        idx_to_id[c["condition_index"]] = c["condition_id"]
    for i, r in enumerate(out["results"], 1):
        r["condition_id"] = idx_to_id.get(r.get("condition_index"), "NR")
        r["record_id"] = f"{sid}-R{i:04d}"
    out["_api_calls"] = int(primary.get("_api_calls", 1)) + int(extra.get("_api_calls", 1))
    out["_targeted_completion"] = True
    return out


def process_one(sid: str, paths: List[Path], cfg: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    t0 = time.perf_counter()
    print(f"[START] {sid} | files={len(paths)}", flush=True)

    evidence = build_evidence(paths)
    prompt = build_prompt(sid, paths, cfg["schema"], cfg["desc"], evidence)

    total_api_calls = 0
    response = deepseek_call(
        cfg["api_key"], cfg["base_url"], cfg["model"],
        prompt, cfg["timeout"], 0, cfg["max_tokens"], SYSTEM_PROMPT
    )
    total_api_calls += 1
    first_finish_reason = response.get("finish_reason", "")
    total_usage = dict(response.get("usage") or {})

    raw = None
    salvaged = False

    try:
        raw = json.loads(response["content"])
    except Exception:
        # 先本地恢复已经完整结束的对象；成功则0额外API。
        raw, salvaged = salvage_truncated_json(response["content"])

    need_rescue = raw is None
    if raw is not None and salvaged and response.get("finish_reason") == "length":
        # 如果截断发生在results之前，本地只能恢复study/conditions，这时才救援。
        if not isinstance(raw.get("results"), list) or len(raw.get("results") or []) == 0:
            need_rescue = True

    if need_rescue:
        # 只有本地恢复不了、或截断后完全没有results的异常论文才第二次调用。
        rescue_prompt = build_rescue_prompt(
            sid, paths, cfg["schema"], cfg["desc"], evidence
        )
        rescue_response = deepseek_call(
            cfg["api_key"], cfg["base_url"], cfg["model"],
            rescue_prompt, cfg["timeout"], 0, RESCUE_MAX_TOKENS, RESCUE_SYSTEM_PROMPT
        )
        total_api_calls += 1
        for k, v in (rescue_response.get("usage") or {}).items():
            if isinstance(v, (int, float)):
                total_usage[k] = total_usage.get(k, 0) + v
        try:
            raw = json.loads(rescue_response["content"])
            response = rescue_response
        except Exception as e:
            raw2, salvaged2 = salvage_truncated_json(rescue_response["content"])
            if raw2 is None:
                raise RuntimeError(
                    f"主提取和异常救援均无法解析：{e}; "
                    f"finish_reason={response.get('finish_reason')}"
                )
            raw = raw2
            salvaged = salvaged or salvaged2
            response = rescue_response

    obj = normalize_obj(
        sid, raw, cfg["schema"], paths, evidence, cfg["model"]
    )
    obj["_finish_reason"] = response.get("finish_reason", "")
    obj["_salvaged"] = bool(salvaged)
    obj["_api_calls"] = total_api_calls

    # Spend a second call only when the first response was actually truncated.
    if first_finish_reason == "length" and total_api_calls < 2:
        targeted_prompt = build_targeted_prompt(sid, paths, cfg["schema"], cfg["desc"], evidence, obj)
        try:
            targeted_response = deepseek_call(
                cfg["api_key"], cfg["base_url"], cfg["model"],
                targeted_prompt, cfg["timeout"], 0, RESCUE_MAX_TOKENS, RESCUE_SYSTEM_PROMPT
            )
            total_api_calls += 1
            for k, v in (targeted_response.get("usage") or {}).items():
                if isinstance(v, (int, float)):
                    total_usage[k] = total_usage.get(k, 0) + v
            try:
                targeted_raw = json.loads(targeted_response["content"])
            except Exception:
                targeted_raw, _ = salvage_truncated_json(targeted_response["content"])
            if isinstance(targeted_raw, dict):
                extra_obj = normalize_obj(sid, targeted_raw, cfg["schema"], paths, evidence, cfg["model"])
                extra_obj["_api_calls"] = 1
                obj = merge_targeted_obj(obj, extra_obj, sid)
                obj["_finish_reason"] = targeted_response.get("finish_reason", "")
                obj["_targeted_finish_reason"] = targeted_response.get("finish_reason", "")
            else:
                obj["_targeted_completion_failed"] = True
        except Exception as e:
            obj["_targeted_completion_failed"] = True
            obj["_targeted_completion_error"] = str(e)
        obj["_api_calls"] = total_api_calls

    ncond = len(obj["conditions"])
    nres = len(obj["results"])

    warning = []
    if ncond == 0:
        warning.append("没有condition")
    if nres == 0:
        warning.append("没有result")
    if response.get("finish_reason") == "length" or (first_finish_reason == "length" and total_api_calls >= 2 and not obj.get("_targeted_completion")):
        warning.append("输出达到max_tokens或截断后已用完补提次数")
    if salvaged:
        warning.append("JSON截断，已本地保留完整结束的对象")
    if obj.get("_targeted_completion_failed"):
        warning.append("截断后定向补提失败，需人工检查")
    if obj.get("_targeted_finish_reason") == "length":
        warning.append("定向补提仍达到token上限，需人工检查")

    usage = total_usage
    runtime = round(time.perf_counter() - t0, 2)
    status = "OK" if not warning else "WARN"

    log = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "study_id": sid,
        "status": status,
        "conditions": ncond,
        "results": nres,
        "api_calls": total_api_calls,
        "prompt_tokens": usage.get("prompt_tokens", ""),
        "completion_tokens": usage.get("completion_tokens", ""),
        "total_tokens": usage.get("total_tokens", ""),
        "finish_reason": response.get("finish_reason", ""),
        "runtime_s": runtime,
        "evidence_chars": evidence["chars"],
        "warning": "; ".join(warning),
    }

    print(
        f"[DONE] {sid} | conditions={ncond} | results={nres} | "
        f"api={total_api_calls} | time={runtime}s | status={status}",
        flush=True,
    )
    return obj, log


def main():
    if load_dotenv:
        load_dotenv(HERE / ".env")

    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf_dir", default=str(HERE / "pdfs"))
    ap.add_argument("--schema", default="")
    ap.add_argument("--output_jsonl", required=True)
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--only", default="")
    ap.add_argument("--model", default="")
    ap.add_argument("--timeout", type=int, default=TIMEOUT)
    ap.add_argument("--retries", type=int, default=MAX_RETRIES)
    ap.add_argument("--max_tokens", type=int, default=MAX_OUTPUT_TOKENS)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    schema_path = find_schema(args.schema)
    schema, desc = load_schema(schema_path)

    api_key = (os.getenv("DEEPSEEK_API_KEY") or os.getenv("DEEPSEEK_KEY") or "").strip()
    if not api_key:
        die("没有读取到 DEEPSEEK_API_KEY，请检查 .env。")

    base_url = (os.getenv("DEEPSEEK_BASE_URL") or DEFAULT_BASE_URL).strip()
    model = (args.model or os.getenv("DEEPSEEK_MODEL") or DEFAULT_MODEL).strip()

    pdf_dir = Path(args.pdf_dir)
    if not pdf_dir.exists():
        die(f"PDF目录不存在：{pdf_dir}")

    groups: Dict[str, List[Path]] = {}
    for p in sorted(pdf_dir.rglob("*.pdf")):
        sid = study_id_from_name(p.name)
        if sid:
            groups.setdefault(sid, []).append(p)

    items = sorted(groups.items())
    if args.only:
        items = [(sid, ps) for sid, ps in items if sid.upper() == args.only.upper()]
    if args.limit > 0:
        items = items[:args.limit]

    if not items:
        die("没有找到带LIB编号的PDF。")

    out = Path(args.output_jsonl)
    out.parent.mkdir(parents=True, exist_ok=True)

    if out.exists() and not args.resume:
        out.unlink()

    done = load_done_ids(out) if args.resume else set()
    if done:
        print(f"[RESUME] 已完成 {len(done)} 篇，将跳过。")

    items = [(sid, ps) for sid, ps in items if sid not in done]
    if not items:
        print("没有新的文献需要处理。")
        return

    log_csv = out.with_name(out.stem + "_log.csv")
    if log_csv.exists() and not args.resume:
        log_csv.unlink()

    cfg = {
        "schema": schema,
        "desc": desc,
        "api_key": api_key,
        "base_url": base_url,
        "model": model,
        "timeout": args.timeout,
        "retries": args.retries,
        "max_tokens": args.max_tokens,
    }

    print("========== Soil occurrence v1 extraction ==========")
    print("字段表：", schema_path)
    print("PDF目录：", pdf_dir)
    print("文献数：", len(items))
    print("模型：", model)
    print("每篇默认API：1次；仅无法本地恢复的异常论文才第2次救援")
    print("thinking：disabled")
    print("JSON mode：enabled")
    print("输出token上限：", args.max_tokens)
    print("输出JSONL：", out)
    print("")

    logs = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
        futures = {
            ex.submit(process_one, sid, paths, cfg): sid
            for sid, paths in items
        }

        for fut in as_completed(futures):
            sid = futures[fut]
            try:
                obj, log = fut.result()

                # 一篇完成立刻追加；不会等全部论文完成才写。
                with out.open("a", encoding="utf-8", newline="\n") as f:
                    f.write(json.dumps(obj, ensure_ascii=False, separators=(",", ":")) + "\n")
                    f.flush()

                logs.append(log)

            except Exception as e:
                log = {
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "study_id": sid,
                    "status": "FAIL",
                    "conditions": 0,
                    "results": 0,
                    "api_calls": 1,
                    "prompt_tokens": "",
                    "completion_tokens": "",
                    "total_tokens": "",
                    "finish_reason": "",
                    "runtime_s": "",
                    "evidence_chars": "",
                    "warning": str(e),
                }
                logs.append(log)
                print(f"[FAIL] {sid} | {e}", flush=True)

    cols = [
        "timestamp", "study_id", "status", "conditions", "results",
        "api_calls", "prompt_tokens", "completion_tokens", "total_tokens",
        "finish_reason", "runtime_s", "evidence_chars", "warning"
    ]
    mode = "a" if args.resume and log_csv.exists() else "w"
    with log_csv.open(mode, encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        if mode == "w":
            w.writeheader()
        for row in logs:
            w.writerow({c: row.get(c, "") for c in cols})

    print("")
    print("========== 完成 ==========")
    print("成功：", sum(x["status"] == "OK" for x in logs))
    print("需关注：", sum(x["status"] == "WARN" for x in logs))
    print("失败：", sum(x["status"] == "FAIL" for x in logs))
    print("API调用：", sum(int(x.get("api_calls") or 0) for x in logs))
    print("Results：", sum(int(x.get("results") or 0) for x in logs))
    print("JSONL：", out)
    print("日志：", log_csv)


if __name__ == "__main__":
    main()

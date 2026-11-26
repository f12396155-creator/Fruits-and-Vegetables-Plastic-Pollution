# -*- coding: utf-8 -*-
"""
v13_fast_onecall_lowcost
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
MAX_METHODS_CHARS = 6000
MAX_RESULTS_CHARS = 22000
MAX_MAIN_TABLE_CHARS = 13000
MAX_SUPP_TABLE_CHARS = 7000
MAX_TOTAL_EVIDENCE_CHARS = 52000

# 紧凑JSON输出，避免以前 12k/14k tokens 导致长文献被截断。
MAX_OUTPUT_TOKENS = 8500
TIMEOUT = 120
MAX_RETRIES = 1

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

    candidates = sorted(
        [p for p in HERE.rglob("Table_S1_字段表*.xlsx") if not p.name.startswith("~$")],
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
        "microplastic", "plastic", "polymer"
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
            "\n===== RESULTS =====\n" + head_tail(results, 20000) +
            "\n===== MAIN TABLE/FIGURE/Numeric PAGES =====\n" + main_tables[:17000] +
            "\n===== SUPPLEMENT TABLE/FIGURE/Numeric PAGES =====\n" + supp_tables[:8000]
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
        "t=sampling_time, m=matrix_tissue, e=endpoint_name, a=measured_analyte, "
        "am=assay_method, s=result_scope, vt=result_value_type, v=result_value, "
        "lo=result_min, hi=result_max, u=result_unit, b=result_basis, n=result_n, "
        "cm=control_mean, ce=control_error_value, cet=control_error_type, cn=control_n, "
        "tm=treatment_mean, te=treatment_error_value, tet=treatment_error_type, tn=treatment_n, "
        "pc=reported_percent_change, src=source_location, note=extraction_note"
    ])
    return "\n".join(lines)


SYSTEM_PROMPT = r"""
你是“果蔬塑料污染证据数据库”的原始事实提取器。

目标不是写摘要，而是从当前论文中尽可能完整地抽取可用于数据库分析的原始事实。

【第一原则】
论文明确报告的数据必须尽可能提全；不要为了省输出而只挑几个代表性结果。
但是绝对不能猜测、估算或补造数字。

【缺失】
NR = 论文未报告；
NA = 结构上不适用；
ND = 明确未检出。
可以省略缺失键，清洗程序会补NR；明确的NA/ND必须保留。

【SOURCE优先级】
1. Main/Supplementary Table中的实际数字
2. Results正文中的实际数字
3. Figure caption/正文中明确给出的数字
4. Methods只能帮助理解方法、单位、n，不能单独制造result。
如果表格有多行、多列，逐行读取，不要只提第一行。

【CONDITION】
一个condition = 一个独立的处理或实测采样层级。
不同污染物、聚合物、剂量、老化状态、共污染物、处理方式或明确不同的采样层级/土地利用来源，可形成不同condition。
同一condition的不同时间、组织、endpoint必须放在results中，不要把时间或endpoint单独变成condition。
control绝不能单独建立condition。
对于实测赋存研究，可按论文真实采样层级/样点/土地利用/深度等建立condition。
不要为了凑数量拆condition。
不要因为培养方式是hydroponic就删除；真实研究设计照实提取，cultivation_medium写hydroponic/soil等。

【RESULT】
一条result = 一个明确的：
condition × sampling_time × matrix_tissue × endpoint_name
组合。
只要论文Results/Table/Figure明确报告，就尽量建立result。

应优先提取：
- 微塑料/塑料颗粒丰度、浓度、含量、残留、积累、粒径、形态、聚合物组成；
- 污染物浓度、残留和组织分布；
- 根、茎、叶、果实等组织中的污染结果；
- 生长、生物量、产量、根长、株高、鲜重、干重等；
- 叶绿素、光合、抗氧化酶、MDA、H2O2等生理生化指标；
- 与暴露直接相关的土壤理化结果；
- 其他明确属于本研究结果的定量/定性endpoint。
不要提与研究问题无关的纯材料制备/材料表征结果。

【数值规则】
A. 表格必须尽量逐行逐组读取。
B. 同一指标有多个处理剂量：每个剂量对应相应condition，分别建立result。
C. 有control_mean和treatment_mean时，严格按同一时间、组织、指标、处理行匹配。
D. 有多个treatment时，不要把多个值塞成一个无法对应的字符串；分别建立对应condition的result。
E. result_value用于论文直接报告的主要单值/均值/中位数等；如果论文直接给出了control/treatment均值，则同时填cm/tm。
F. result_min/result_max只在论文明确报告范围时填写，不能由均值±SD计算。
G. reported_percent_change只抄作者明确报告的百分比变化；不要自己计算。
H. SD、SE、SEM原样记录到error字段，并标明error_type。
I. n只填独立实验重复/样本量；不把技术重复当n。
J. 单位原样复制，不自行换算。
K. 只有图形没有数字标签时，不能估算柱高；可以建立result，但数值字段留空。
L. source_location必须尽量写：Main: Table 2 / Main: Fig. 3 / Main: Results 3.2 / Supplement: Table S4等。
M. 如果同一结果有正文和表格两个来源，优先表格数字，并可在note中简短说明。
N. measured_analyte只在确有具体分析物时填写；普通植物生长/生理指标可为NA。

【完整度要求】
不要只提“显著增加/降低”的方向结果。
如果论文给了均值、误差、n、单位，就全部提。
如果一个表有10个endpoint、4个处理、3个时间，尽量把实际报告的每个可对应组合都提出来。
不要生成论文中不存在的组合。

【最省token输出】
- 不写解释、摘要、结论。
- 不输出Markdown。
- 结果字段使用短键。
- 不输出NR键。
- 每条result只写实际有信息的键。
- condition字段可以使用完整字段名；result使用短键。
- 不输出study_id/condition_id/record_id，这些由程序生成。
- study/conditions/results必须是数组/对象组成的一个JSON对象。

【必须输出】
{
  "study": {...},
  "conditions": [
    {"i":1, ...},
    {"i":2, ...}
  ],
  "results": [
    {"c":1, "t":"...", "m":"...", "e":"...", "v":..., "u":"...", "src":"..."},
    ...
  ],
  "questions": []
}

其中results中的c=condition的i。
严格只输出一个合法JSON对象，不要```，不要解释。
"""


def build_prompt(sid: str, paths: List[Path], schema: Dict[str, List[str]], desc: Dict[str, str], evidence: Dict[str, Any]) -> str:
    guide = field_guide(schema, desc)
    return (
        f"Study ID: {sid}\n"
        f"Source files: {', '.join(p.name for p in paths)}\n\n"
        f"{guide}\n\n"
        "【执行要求】\n"
        "先根据Methods/本研究表格建立真实conditions，再从Results/Table/Supplementary Table提取所有明确报告的results。"
        "优先真实数值，少写描述。不要为了输出漂亮而减少result数量。\n\n"
        "===== PAPER EVIDENCE START =====\n"
        + evidence["text"]
        + "\n===== PAPER EVIDENCE END =====\n"
    )


def deepseek_call(api_key: str, base_url: str, model: str, prompt: str, timeout: int, retries: int, max_tokens: int) -> Dict[str, Any]:
    url = base_url.rstrip("/") + "/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
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
            else:
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
            str(r.get("result_value", "")),
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


def process_one(sid: str, paths: List[Path], cfg: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    t0 = time.perf_counter()
    print(f"[START] {sid} | files={len(paths)}", flush=True)

    evidence = build_evidence(paths)
    prompt = build_prompt(sid, paths, cfg["schema"], cfg["desc"], evidence)

    response = deepseek_call(
        cfg["api_key"], cfg["base_url"], cfg["model"],
        prompt, cfg["timeout"], cfg["retries"], cfg["max_tokens"]
    )

    try:
        raw = json.loads(response["content"])
    except Exception as e:
        # JSON mode正常时不应发生；不做昂贵的二次API修复。
        raise RuntimeError(
            f"JSON解析失败：{e}; finish_reason={response.get('finish_reason')}"
        )

    obj = normalize_obj(
        sid, raw, cfg["schema"], paths, evidence, cfg["model"]
    )
    obj["_finish_reason"] = response.get("finish_reason", "")

    ncond = len(obj["conditions"])
    nres = len(obj["results"])

    warning = []
    if ncond == 0:
        warning.append("没有condition")
    if nres == 0:
        warning.append("没有result")
    if response.get("finish_reason") == "length":
        warning.append("输出达到max_tokens，尾部可能截断")

    usage = response.get("usage") or {}
    runtime = round(time.perf_counter() - t0, 2)

    log = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "study_id": sid,
        "status": "OK" if not warning else "WARN",
        "conditions": ncond,
        "results": nres,
        "api_calls": 1,
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
        f"time={runtime}s | status={log['status']}",
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

    print("========== v13 快速单次提取 ==========")
    print("字段表：", schema_path)
    print("PDF目录：", pdf_dir)
    print("文献数：", len(items))
    print("模型：", model)
    print("每篇默认API：1次")
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

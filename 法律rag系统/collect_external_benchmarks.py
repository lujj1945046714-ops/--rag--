# -*- coding: utf-8 -*-
"""
从外部开源法律数据集构建补充评测集，合并到 test_qa_pair_verify.json。

支持的数据集：
  1. LawBench（doolayer/LawBench）—— HuggingFace 自动下载
     - 3-1: 法条适用（给定案情+罪名，找相关法条）
     - 3-3: 罪名预测（给定案情，预测罪名）
  2. JEC-QA（hails/agieval-jec-qa-kd）—— HuggingFace 自动下载
     - 司法考试单选题

用法：
    # 仅使用 LawBench（自动下载）
    python collect_external_benchmarks.py --lawbench

    # 仅使用 JEC-QA（自动下载）
    python collect_external_benchmarks.py --jecqa

    # 两者都用，并限制每个数据集最多 100 条
    python collect_external_benchmarks.py --lawbench --jecqa --max-per-source 100

    # 合并后覆盖 test_qa_pair_verify.json（默认追加）
    python collect_external_benchmarks.py --lawbench --jecqa --overwrite
"""

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

_LAW_NAME_RE = re.compile(r'《([^》]{2,30})》')
_ARTICLE_RE = re.compile(r'第[零一二三四五六七八九十百千万\d]+条')
_LEGAL_TERMS = [
    "合同", "侵权", "赔偿", "违约", "解除", "无效", "撤销",
    "劳动合同", "加班费", "工伤", "仲裁", "诉讼", "管辖",
    "继承", "遗嘱", "婚姻", "离婚", "抚养", "监护",
    "刑事责任", "犯罪", "量刑", "缓刑", "假释",
    "行政处罚", "行政许可", "行政复议",
    "知识产权", "著作权", "专利", "商标",
]


def _uid(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def _extract_keywords(text: str) -> list:
    kws = []
    for m in _LAW_NAME_RE.finditer(text):
        name = m.group(1)
        if name not in kws:
            kws.append(name)
    for a in _ARTICLE_RE.findall(text)[:3]:
        if a not in kws:
            kws.append(a)
    for term in _LEGAL_TERMS:
        if term in text and term not in kws:
            kws.append(term)
    return kws[:8]


# ---------------------------------------------------------------------------
# LawBench（doolayer/LawBench）
# ---------------------------------------------------------------------------

def load_lawbench(max_per_source: int) -> list:
    """
    从 HuggingFace 加载 doolayer/LawBench。
    使用 3-1（法条适用）和 3-3（罪名预测）两个配置。
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("[LawBench] 需要安装 datasets：pip install datasets")
        return []

    items = []
    configs = {
        "3-1": ("法条适用", "instruction", "question", "answer"),
        "3-3": ("罪名预测", "instruction", "question", "answer"),
    }

    for config_id, (desc, inst_col, q_col, a_col) in configs.items():
        print(f"[LawBench] 正在下载 config={config_id}（{desc}）...")
        try:
            ds = load_dataset("doolayer/LawBench", config_id)
        except Exception as e:
            print(f"[LawBench] config={config_id} 下载失败: {e}")
            continue

        split = ds.get("test") or ds.get("train") or list(ds.values())[0]
        count = 0
        per_config_limit = max_per_source // 2 if max_per_source else 0

        for row in split:
            instruction = str(row.get(inst_col, "")).strip()
            question = str(row.get(q_col, "")).strip()
            answer = str(row.get(a_col, "")).strip()

            if not question or not answer:
                continue

            # 将 instruction 作为问题前缀，让问题更完整
            full_question = f"{instruction}\n{question}" if instruction else question

            # 清理答案中的格式标记，如 [罪名]盗窃<eoa> → 盗窃
            answer_clean = re.sub(r'\[.*?\]', '', answer)
            answer_clean = re.sub(r'<eoa>', '', answer_clean).strip()
            if not answer_clean:
                answer_clean = answer

            items.append({
                "unique_id": _uid(f"lawbench_{config_id}_{question}"),
                "question": full_question,
                "answer": answer_clean,
                "keywords": _extract_keywords(full_question + answer_clean),
                "_source": f"lawbench-{config_id}",
            })
            count += 1
            if per_config_limit and count >= per_config_limit:
                break

        print(f"[LawBench] config={config_id} 加载 {count} 条")

    print(f"[LawBench] 合计加载 {len(items)} 条")
    return items


# ---------------------------------------------------------------------------
# JEC-QA（hails/agieval-jec-qa-kd，单选题）
# ---------------------------------------------------------------------------

def load_jecqa(max_per_source: int) -> list:
    """
    从 HuggingFace 加载 hails/agieval-jec-qa-kd（司法考试单选题）。
    将选择题转换为 question + 正确选项文字 的 QA 格式。
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("[JEC-QA] 需要安装 datasets：pip install datasets")
        return []

    print("[JEC-QA] 正在下载 hails/agieval-jec-qa-kd...")
    try:
        ds = load_dataset("hails/agieval-jec-qa-kd")
    except Exception as e:
        print(f"[JEC-QA] 下载失败: {e}")
        return []

    split = ds.get("test") or ds.get("train") or list(ds.values())[0]
    items = []

    for row in split:
        query = str(row.get("query", "")).strip()
        choices = row.get("choices", [])
        gold = row.get("gold", [])

        if not query or not choices or not gold:
            continue

        # 单选题：gold 是 [index]
        gold_idx = gold[0] if isinstance(gold, list) else gold
        if gold_idx >= len(choices):
            continue

        answer = choices[gold_idx].strip()
        # 去掉选项前缀 "(A)" 等
        answer = re.sub(r'^\([A-D]\)\s*', '', answer)

        # 从 query 中提取纯问题（去掉"答案：从A到D, 我们应选择"后缀）
        question = re.sub(r'\n答案：.*$', '', query, flags=re.DOTALL).strip()
        # 去掉"问题："前缀
        question = re.sub(r'^问题：', '', question).strip()

        items.append({
            "unique_id": _uid(f"jecqa_{question}"),
            "question": question,
            "answer": answer,
            "keywords": _extract_keywords(question + answer),
            "_source": "jecqa",
        })

        if max_per_source and len(items) >= max_per_source:
            break

    print(f"[JEC-QA] 加载 {len(items)} 条")
    return items


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="集成外部法律评测集")
    parser.add_argument("--lawbench", action="store_true", help="从 HuggingFace 下载 LawBench（doolayer/LawBench）")
    parser.add_argument("--jecqa", action="store_true", help="从 HuggingFace 下载 JEC-QA（hails/agieval-jec-qa-kd）")
    parser.add_argument("--max-per-source", type=int, default=200, help="每个数据集最多取多少条（0=不限）")
    parser.add_argument("--output", type=str, default="data/qa_pairs/test_qa_pair_verify.json")
    parser.add_argument("--overwrite", action="store_true", help="覆盖输出文件（默认追加去重）")
    args = parser.parse_args()

    if not args.lawbench and not args.jecqa:
        parser.print_help()
        sys.exit(0)

    new_items = []
    if args.lawbench:
        new_items += load_lawbench(args.max_per_source)
    if args.jecqa:
        new_items += load_jecqa(args.max_per_source)

    if not new_items:
        print("没有加载到任何数据，退出。")
        return

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 加载已有数据（追加模式）
    existing = []
    if not args.overwrite and output_path.exists():
        existing = json.loads(output_path.read_text(encoding="utf-8"))
        print(f"已有 {len(existing)} 条，追加去重...")

    # 去重（按 unique_id）
    seen_ids = {item["unique_id"] for item in existing}
    added = []
    for item in new_items:
        if item["unique_id"] not in seen_ids:
            item.pop("_source", None)
            existing.append(item)
            added.append(item)
            seen_ids.add(item["unique_id"])

    output_path.write_text(
        json.dumps(existing, ensure_ascii=False, indent=4),
        encoding="utf-8"
    )
    print(f"\n新增 {len(added)} 条，合计 {len(existing)} 条 → {output_path}")

    # 预览
    print("\n--- 新增样例预览 ---")
    for item in added[:3]:
        print(f"问题: {item['question'][:80]}")
        print(f"答案: {item['answer'][:80]}")
        print(f"关键词: {item['keywords']}")
        print()


if __name__ == "__main__":
    main()

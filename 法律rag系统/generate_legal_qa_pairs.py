# -*- coding: utf-8 -*-
"""
从 LegalBench-RAG/benchmarks/chinese_legal_cases.json 生成
final_score.py 所需的 data/qa_pairs/test_qa_pair_verify.json。

用法：
    python generate_legal_qa_pairs.py
"""

import hashlib
import json
import re
from pathlib import Path

# 提取《法律名称》
_LAW_NAME_RE = re.compile(r'《([^》]{2,30})》')
# 提取第X条
_ARTICLE_RE = re.compile(r'第[零一二三四五六七八九十百千万\d]+条')
# 提取关键法律术语
_LEGAL_TERMS = [
    "仲裁时效", "劳动报酬", "加班费", "解除劳动合同", "赔偿金",
    "举证责任", "连带赔偿责任", "工伤", "劳务派遣",
    "遗嘱", "继承", "撤销婚姻", "重大疾病",
    "公序良俗", "合同无效", "惩罚性赔偿", "食品安全",
    "格式条款", "消费者权益", "预付卡",
    "行政协议", "行政优益权", "补偿安置", "特许权协议",
    "诚实信用", "法定解除", "情势变更",
]


def extract_keywords(answer: str) -> list:
    """从答案文本中提取关键词：法律名称 + 条文编号 + 法律术语"""
    keywords = []

    # 法律名称（去掉书名号）
    for m in _LAW_NAME_RE.finditer(answer):
        name = m.group(1)
        if name not in keywords:
            keywords.append(name)

    # 条文编号（取前3个，避免过多）
    articles = _ARTICLE_RE.findall(answer)
    for a in articles[:3]:
        if a not in keywords:
            keywords.append(a)

    # 法律术语
    for term in _LEGAL_TERMS:
        if term in answer and term not in keywords:
            keywords.append(term)

    return keywords[:8]  # 最多8个关键词


def make_uid(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def main():
    benchmark_path = Path("LegalBench-RAG/benchmarks/chinese_legal_cases.json")
    output_path = Path("data/qa_pairs/test_qa_pair_verify.json")

    if not benchmark_path.exists():
        raise FileNotFoundError(f"未找到: {benchmark_path}\n请先运行 generate_legal_benchmark.py")

    data = json.loads(benchmark_path.read_text(encoding="utf-8"))
    tests = data.get("tests", [])
    print(f"读取测试条目: {len(tests)}")

    qa_pairs = []
    for item in tests:
        query = item.get("query", "").strip()
        snippets = item.get("snippets", [])
        if not query or not snippets:
            continue

        answer = snippets[0].get("answer", "").strip()
        if not answer:
            continue

        keywords = extract_keywords(answer)

        qa_pairs.append({
            "unique_id": make_uid(query),
            "question": query,
            "answer": answer,
            "keywords": keywords,
        })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(qa_pairs, ensure_ascii=False, indent=4),
        encoding="utf-8"
    )
    print(f"已生成 {len(qa_pairs)} 条问答对 → {output_path}")

    # 预览前3条
    print("\n--- 预览 ---")
    for item in qa_pairs[:3]:
        print(f"问题: {item['question']}")
        print(f"答案前80字: {item['answer'][:80]}")
        print(f"关键词: {item['keywords']}")
        print()


if __name__ == "__main__":
    main()

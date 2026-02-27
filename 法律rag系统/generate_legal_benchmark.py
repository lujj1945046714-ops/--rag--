# -*- coding: utf-8 -*-
"""
从 Laws 仓库案例文件自动生成中文法律测评集。
输出格式与 evaluate_legalbench_rag.py 兼容。

用法：
    python generate_legal_benchmark.py \
        --laws-dir .cache/laws_repo \
        --output-dir LegalBench-RAG
"""

import argparse
import json
import re
from pathlib import Path

# 答案段优先级：先找这些段落作为 ground-truth span
_ANSWER_SECTIONS = [
    "案例分析",
    "裁判要旨",
    "典型意义",
    "适用解析",
    "处理结果",
    "裁判结果",
]

# 段落标题正则
_SECTION_RE = re.compile(r'^#{1,3}\s*(.+?)\s*$', re.MULTILINE)


def read_text(path: Path) -> str:
    for enc in ("utf-8", "utf-8-sig", "gbk"):
        try:
            return path.read_text(encoding=enc)
        except UnicodeDecodeError:
            continue
    return path.read_text(encoding="utf-8", errors="ignore")


def extract_title(text: str, fallback: str) -> str:
    """提取文件标题：优先取第一个 # 标题，其次用文件名"""
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            title = stripped.lstrip("#").strip()
            # 跳过 INFO END 注释行和空标题
            if title and "INFO" not in title:
                return title
    return fallback


def extract_answer_span(text: str) -> tuple:
    """
    找到优先级最高的答案段落，返回 (start, end, content)。
    start/end 是字符位置（相对于整个文件内容）。
    """
    # 找出所有段落标题的位置
    section_matches = list(_SECTION_RE.finditer(text))

    for target in _ANSWER_SECTIONS:
        for i, m in enumerate(section_matches):
            if m.group(1).strip() == target:
                # 段落内容从标题行结束到下一个段落标题开始
                content_start = m.end()
                content_end = section_matches[i + 1].start() if i + 1 < len(section_matches) else len(text)
                content = text[content_start:content_end].strip()
                if len(content) < 30:
                    continue
                # span 包含标题行本身，让检索器能定位到段落
                return m.start(), content_end, content

    return None, None, None


def is_case_file(path: Path, root: Path) -> bool:
    """判断是否是案例文件（在 案例/ 目录下，不是 _index.md）"""
    try:
        rel = path.relative_to(root)
    except ValueError:
        return False
    parts = rel.parts
    return len(parts) >= 2 and parts[0] == "案例" and path.name != "_index.md"


def build_benchmark(laws_dir: Path) -> list:
    """遍历案例目录，生成测试条目列表"""
    tests = []
    case_files = sorted(
        p for p in laws_dir.rglob("*.md") if is_case_file(p, laws_dir)
    )

    for path in case_files:
        text = read_text(path)
        rel_path = path.relative_to(laws_dir).as_posix()

        # 提取 query
        title = extract_title(text, path.stem)
        if not title:
            continue

        # 提取 answer span
        span_start, span_end, answer_text = extract_answer_span(text)
        if span_start is None:
            # 兜底：用整个文件（去掉 INFO END 注释前的元数据）
            info_end = text.find("<!-- INFO END -->")
            span_start = (text.find("\n", info_end) + 1) if info_end >= 0 else 0
            span_end = len(text)
            answer_text = text[span_start:span_end].strip()

        if not answer_text:
            continue

        tests.append({
            "query": title,
            "snippets": [
                {
                    "file_path": rel_path,
                    "span": [span_start, span_end],
                    "answer": answer_text[:500],  # 仅供人工核查，不影响评测
                }
            ]
        })

    return tests


def main():
    parser = argparse.ArgumentParser(description="生成中文法律案例测评集")
    parser.add_argument("--laws-dir", default=".cache/laws_repo", help="Laws 仓库根目录")
    parser.add_argument("--output-dir", default="LegalBench-RAG", help="输出目录（与 evaluate 脚本兼容）")
    args = parser.parse_args()

    laws_dir = Path(args.laws_dir).resolve()
    output_dir = Path(args.output_dir)

    if not laws_dir.exists():
        raise FileNotFoundError(f"Laws 目录不存在: {laws_dir}")

    # 生成测试条目
    tests = build_benchmark(laws_dir)
    print(f"生成测试条目数: {len(tests)}")

    # 写入 benchmarks/chinese_legal_cases.json
    benchmarks_dir = output_dir / "benchmarks"
    benchmarks_dir.mkdir(parents=True, exist_ok=True)
    benchmark_file = benchmarks_dir / "chinese_legal_cases.json"
    benchmark_file.write_text(
        json.dumps({"tests": tests}, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    print(f"测评集已写入: {benchmark_file}")

    # corpus 目录软链接或提示
    corpus_dir = output_dir / "corpus"
    corpus_dir.mkdir(parents=True, exist_ok=True)

    # 把案例文件复制到 corpus/案例/ 下，供 evaluate 脚本按相对路径读取
    import shutil
    case_src = laws_dir / "案例"
    case_dst = corpus_dir / "案例"
    if case_dst.exists():
        shutil.rmtree(case_dst)
    shutil.copytree(case_src, case_dst)
    print(f"案例语料已复制到: {case_dst}")

    # 打印几条样例
    print("\n--- 样例预览 ---")
    for t in tests[:3]:
        print(f"query: {t['query']}")
        print(f"file:  {t['snippets'][0]['file_path']}")
        print(f"span:  {t['snippets'][0]['span']}")
        print(f"answer前50字: {t['snippets'][0]['answer'][:50]}")
        print()

    print(f"运行评测：")
    print(f"  python evaluate_legalbench_rag.py --legalbench-root {output_dir}")


if __name__ == "__main__":
    main()

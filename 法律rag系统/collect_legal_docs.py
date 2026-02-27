import argparse
import json
from pathlib import Path

from src.parser.legal_source_ingest import collect_legal_documents


def parse_args():
    parser = argparse.ArgumentParser(description="Collect legal documents from official websites or open-source Laws")
    parser.add_argument("--source", choices=["official", "laws", "both"], required=True)
    parser.add_argument(
        "--official-mode",
        choices=["npc_api", "crawler", "legal_rag_seed"],
        default="npc_api",
        help="官网采集模式: npc_api(参考 Laws) / crawler(通用爬取) / legal_rag_seed(参考 legal_rag 种子网址)",
    )
    parser.add_argument(
        "--official-start-url",
        action="append",
        default=None,
        help="官网爬取入口，可重复传多个",
    )
    parser.add_argument("--npc-search-type", type=str, default="1,9", help="NPC API searchType 末尾值，默认 1,9")
    parser.add_argument("--npc-max-pages", type=int, default=20, help="NPC API 最大翻页数")
    parser.add_argument("--npc-page-size", type=int, default=10, help="NPC API 每页数量")
    parser.add_argument("--official-depth", type=int, default=2)
    parser.add_argument("--max-official-docs", type=int, default=200)
    parser.add_argument("--laws-dir", type=str, default=None, help="开源 Laws 数据目录")
    parser.add_argument(
        "--laws-repo-url",
        type=str,
        default="https://github.com/LawRefBook/Laws.git",
        help="未提供 laws-dir 时，将自动 clone 该仓库",
    )
    parser.add_argument("--laws-repo-branch", type=str, default="master")
    parser.add_argument("--laws-repo-local-dir", type=str, default=".cache/laws_repo")
    parser.add_argument("--max-laws-docs", type=int, default=5000)
    parser.add_argument("--output-jsonl", type=str, default=None, help="导出到 jsonl 文件")
    return parser.parse_args()


def main():
    args = parse_args()
    docs = collect_legal_documents(
        source=args.source,
        official_mode=args.official_mode,
        official_start_urls=args.official_start_url,
        npc_search_type=args.npc_search_type,
        npc_max_pages=args.npc_max_pages,
        npc_page_size=args.npc_page_size,
        laws_dir=args.laws_dir,
        laws_repo_url=args.laws_repo_url,
        laws_repo_branch=args.laws_repo_branch,
        laws_repo_local_dir=args.laws_repo_local_dir,
        max_official_docs=args.max_official_docs,
        official_depth=args.official_depth,
        max_laws_docs=args.max_laws_docs,
    )
    print(f"采集完成，总文档数: {len(docs)}")

    if args.output_jsonl:
        output_path = Path(args.output_jsonl)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            for doc in docs:
                payload = {
                    "page_content": doc.page_content,
                    "metadata": doc.metadata,
                }
                f.write(json.dumps(payload, ensure_ascii=False) + "\n")
        print(f"已导出: {output_path}")

    if docs:
        sample = docs[0]
        print("样例标题:", sample.metadata.get("title", ""))
        print("样例来源:", sample.metadata.get("source", ""))


if __name__ == "__main__":
    main()

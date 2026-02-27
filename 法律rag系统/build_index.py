# -*- coding: utf-8 -*-
# --------------------------------------------
# 项目名称: LLM任务型对话Agent
# --------------------------------------------

import argparse
import os
import pickle

from src.constant import (
    clean_docs_path,
    legal_raw_docs_path,
    legal_split_docs_path,
    raw_docs_path,
    split_docs_path,
)
from src.parser.legal_source_ingest import collect_legal_documents
from src.parser.pdf_parse import load_pdf, texts_split, save_2_mongo
from src.parser.legal_splitter import legal_texts_split
from src.retriever.bm25_retriever import BM25
from src.retriever.milvus_retriever import MilvusRetriever


def _ensure_parent(path: str):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def _save_pickle(path: str, data):
    _ensure_parent(path)
    with open(path, "wb") as f:
        pickle.dump(data, f)


def _load_pickle(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


def _with_tag(path: str, tag: str) -> str:
    if not tag:
        return path
    root, ext = os.path.splitext(path)
    return f"{root}_{tag}{ext}"


def parse_args():
    parser = argparse.ArgumentParser(description="Build retrieval index")
    parser.add_argument(
        "--source",
        choices=["pdf", "official", "laws", "both"],
        default="pdf",
        help="文档来源: pdf(原流程) / official(官网爬取) / laws(开源法律库) / both(两者)",
    )
    parser.add_argument(
        "--official-start-url",
        action="append",
        default=None,
        help="官网爬取入口，可重复传多个；不传则默认 https://flk.npc.gov.cn/",
    )
    parser.add_argument(
        "--official-mode",
        choices=["npc_api", "crawler", "legal_rag_seed"],
        default="npc_api",
        help="官网采集模式: npc_api(参考 Laws) / crawler(通用爬取) / legal_rag_seed(参考 legal_rag 种子网址)",
    )
    parser.add_argument("--npc-search-type", type=str, default="1,9", help="NPC API searchType 末尾值，默认 1,9")
    parser.add_argument("--npc-max-pages", type=int, default=20, help="NPC API 最大翻页数")
    parser.add_argument("--npc-page-size", type=int, default=10, help="NPC API 每页数量")
    parser.add_argument("--official-depth", type=int, default=2, help="官网爬取深度")
    parser.add_argument("--max-official-docs", type=int, default=200, help="最多采集多少篇官网文档")
    parser.add_argument("--laws-dir", type=str, default=None, help="Laws 开源数据目录")
    parser.add_argument(
        "--laws-repo-url",
        type=str,
        default="https://github.com/LawRefBook/Laws.git",
        help="未提供 laws-dir 时，将自动 clone 该仓库",
    )
    parser.add_argument("--laws-repo-branch", type=str, default="master")
    parser.add_argument("--laws-repo-local-dir", type=str, default=".cache/laws_repo")
    parser.add_argument("--max-laws-docs", type=int, default=5000, help="最多导入多少篇开源文档")
    parser.add_argument("--skip-clean", action="store_true", help="跳过 LLM 文本清洗")
    parser.add_argument("--force-rebuild", action="store_true", help="忽略缓存，强制重新处理")
    parser.add_argument(
        "--use-cache",
        action="store_true",
        help="法律数据流程默认不复用缓存，传此参数后会复用缓存",
    )
    parser.add_argument(
        "--cache-tag",
        default=None,
        help="缓存标签，便于区分 official/laws/both 的不同实验",
    )
    parser.add_argument(
        "--demo-query",
        default="介绍一下离车后自动上锁功能",
        help="建索引后用于快速验证召回效果的查询",
    )
    return parser.parse_args()


def _build_raw_docs(args):
    if args.source == "pdf":
        return load_pdf()
    return collect_legal_documents(
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


def _get_cache_paths(args):
    if args.source == "pdf":
        raw_path = raw_docs_path
        clean_path = clean_docs_path
        split_path = split_docs_path
    else:
        raw_path = legal_raw_docs_path
        clean_path = legal_raw_docs_path
        split_path = legal_split_docs_path

    tag = args.cache_tag
    if tag:
        raw_path = _with_tag(raw_path, tag)
        clean_path = _with_tag(clean_path, tag)
        split_path = _with_tag(split_path, tag)
    return raw_path, clean_path, split_path


def main():
    args = parse_args()
    raw_path, clean_path, split_path = _get_cache_paths(args)
    reuse_cache = args.use_cache or args.source == "pdf"

    if reuse_cache and (not args.force_rebuild) and os.path.exists(raw_path):
        raw_docs = _load_pickle(raw_path)
        print("加载原始文档数:", len(raw_docs))
    else:
        raw_docs = _build_raw_docs(args)
        print("原始文档数:", len(raw_docs))
        _save_pickle(raw_path, raw_docs)

    need_clean = args.source == "pdf" and (not args.skip_clean)
    if need_clean:
        from src.client.llm_clean_client import request_llm_clean

        if reuse_cache and (not args.force_rebuild) and os.path.exists(clean_path):
            clean_docs = _load_pickle(clean_path)
            print("加载清洗文档数:", len(clean_docs))
        else:
            clean_docs = request_llm_clean(raw_docs)
            print("清洗后文档数:", len(clean_docs))
            _save_pickle(clean_path, clean_docs)
    else:
        clean_docs = raw_docs
        print("跳过清洗，直接使用原始文档")

    if reuse_cache and (not args.force_rebuild) and os.path.exists(split_path):
        split_docs = _load_pickle(split_path)
        print("加载切分文档总数:", len(split_docs))
    else:
        if args.source == "pdf":
            split_docs = texts_split(clean_docs)
        else:
            parent_docs, split_docs = legal_texts_split(clean_docs)
            print("父文档数（章节级，存MongoDB）:", len(parent_docs))
            save_2_mongo(parent_docs)
        print("切分后文档总数:", len(split_docs))
        _save_pickle(split_path, split_docs)

    bm25_retriever = BM25(split_docs)
    bm25_docs = bm25_retriever.retrieve_topk(args.demo_query, topk=3)
    print("BM25召回样例:")
    print(bm25_docs)
    print("=" * 100)

    milvus_retriever = MilvusRetriever(split_docs)
    milvus_docs = milvus_retriever.retrieve_topk(args.demo_query, topk=3)
    print("BGE-M3召回样例:")
    print(milvus_docs)


if __name__ == "__main__":
    main()

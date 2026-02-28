#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""简化的索引重建脚本，直接使用已有的 split_docs.pkl"""

import pickle
from src.constant import split_docs_path
from src.retriever.bm25_retriever import BM25
from src.retriever.milvus_retriever import MilvusRetriever
from src.parser.pdf_parse import save_2_mongo

print("加载切分文档...")
with open(split_docs_path, 'rb') as f:
    split_docs = pickle.load(f)
print(f"文档数: {len(split_docs)}")

print("\n保存到 MongoDB...")
save_2_mongo(split_docs)

print("\n构建 BM25 索引...")
bm25_retriever = BM25(split_docs)
print("BM25 索引构建完成")

print("\n构建 Milvus 索引...")
milvus_retriever = MilvusRetriever(split_docs)
print("Milvus 索引构建完成")

print("\n测试检索...")
query = "劳动合同纠纷如何处理"
bm25_docs = bm25_retriever.retrieve_topk(query, topk=3)
print(f"\nBM25 召回 {len(bm25_docs)} 条:")
for i, doc in enumerate(bm25_docs[:2]):
    print(f"{i+1}. {doc.page_content[:80]}...")

milvus_docs = milvus_retriever.retrieve_topk(query, topk=3)
print(f"\nMilvus 召回 {len(milvus_docs)} 条:")
for i, doc in enumerate(milvus_docs[:2]):
    print(f"{i+1}. {doc.page_content[:80]}...")

print("\n✓ 索引重建完成")

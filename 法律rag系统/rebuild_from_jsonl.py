#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""从 laws.jsonl 重建法律文档索引"""

import json
from langchain_core.documents import Document
from src.retriever.bm25_retriever import BM25
from src.parser.pdf_parse import save_2_mongo

print("加载 laws.jsonl...")
docs = []
with open("data/processed_docs/laws.jsonl", 'r', encoding='utf-8') as f:
    for line in f:
        data = json.loads(line)
        doc = Document(
            page_content=data['page_content'],
            metadata=data.get('metadata', {})
        )
        docs.append(doc)

print(f"文档数: {len(docs)}")

print("\n保存到 MongoDB...")
save_2_mongo(docs)

print("\n构建 BM25 索引...")
bm25_retriever = BM25(docs)
print("BM25 索引构建完成")

print("\n测试检索...")
query = "劳动合同纠纷如何处理"
bm25_docs = bm25_retriever.retrieve_topk(query, topk=3)
print(f"\nBM25 召回 {len(bm25_docs)} 条:")
for i, doc in enumerate(bm25_docs):
    print(f"{i+1}. {doc.page_content[:100]}...")

print("\n✓ 法律文档索引重建完成")

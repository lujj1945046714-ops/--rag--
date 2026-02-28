#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""从 laws.jsonl 构建 Milvus 向量索引"""

import os
import json
from dotenv import load_dotenv

# 强制重新加载环境变量
os.environ.pop('SILICONFLOW_API_KEY', None)
load_dotenv(override=True)

from langchain_core.documents import Document
from src.retriever.milvus_retriever import MilvusRetriever

print("加载 laws.jsonl...")
docs = []
with open("data/processed_docs/laws.jsonl", 'r', encoding='utf-8') as f:
    for line in f:
        data = json.loads(line)
        # 确保 page_content 不超过 1024 字符
        content = data['page_content'][:1024]
        # 确保有 unique_id
        if 'metadata' not in data:
            data['metadata'] = {}
        if 'unique_id' not in data['metadata']:
            import hashlib
            data['metadata']['unique_id'] = hashlib.md5(content.encode()).hexdigest()

        doc = Document(
            page_content=content,
            metadata=data['metadata']
        )
        docs.append(doc)

print(f"文档数: {len(docs)}")
print(f"示例文档长度: {[len(d.page_content) for d in docs[:5]]}")

print("\n构建 Milvus 向量索引（调用 SiliconFlow API）...")
milvus_retriever = MilvusRetriever(docs)
print("Milvus 索引构建完成")

print("\n测试检索...")
query = "劳动合同纠纷如何处理"
milvus_docs = milvus_retriever.retrieve_topk(query, topk=3)
print(f"\nMilvus 召回 {len(milvus_docs)} 条:")
for i, doc in enumerate(milvus_docs):
    print(f"{i+1}. {doc.page_content[:100]}...")

print("\n✓ Milvus 向量索引构建完成")

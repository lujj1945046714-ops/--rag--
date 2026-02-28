#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""仅使用 BM25 的简化推理脚本（不依赖向量检索）"""

import time
from src.retriever.bm25_retriever import BM25
from src.client.llm_chat_client import request_chat
from src.utils import post_processing

# 初始化 BM25
print("初始化 BM25 检索器...")
bm25_retriever = BM25(docs=None, retrieve=True)

while True:
    query = input("\n输入问题（输入 q 退出）—> ")
    if query.lower() == 'q':
        break

    # BM25 召回
    t1 = time.time()
    bm25_docs = bm25_retriever.retrieve_topk(query, topk=5)
    print(f"\nBM25 召回 {len(bm25_docs)} 条（耗时 {time.time()-t1:.2f}s）:")
    for i, doc in enumerate(bm25_docs[:3]):
        print(f"  [{i+1}] {doc.page_content[:100]}...")
    print("=" * 80)

    # 生成答案
    context = "\n".join([f"【{idx+1}】{doc.page_content}" for idx, doc in enumerate(bm25_docs)])
    res_handler = request_chat(query, context, stream=True)

    print("\n答案—> ", end='')
    response = ""
    for r in res_handler:
        uttr = r.choices[0].delta.content
        if not uttr:
            continue
        response += uttr
        print(uttr, end='', flush=True)

    # 后处理
    answer = post_processing(response, bm25_docs)
    print(f"\n\n最终答案—> {answer}")
    print("=" * 80)

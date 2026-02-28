import os
import time
from src.retriever.bm25_retriever import BM25
from src.retriever.milvus_retriever import MilvusRetriever
from src.client.llm_chat_client import request_chat
from src.client.llm_router_client import route_and_rewrite
from src.reranker.qwen3_reranker import Qwen3ReRanker
from src.constant import qwen3_4b_reranker_model_path
from src.utils import merge_docs, post_processing

RERANK_SCORE_THRESHOLD = 0.3

# warmstart
bm25_retriever = BM25(docs=None, retrieve=True)
milvus_retriever = MilvusRetriever(docs=None, retrieve=True)
qwen3_reranker = Qwen3ReRanker(model_path=qwen3_4b_reranker_model_path)
milvus_retriever.retrieve_topk("这是一条测试数据", topk=3)


while True:
    query = input("输入—>")

    # 路由判断 + 查询改写
    route_result = route_and_rewrite(query)
    if not route_result["is_legal"]:
        print("该问题与法律无关，无法回答。")
        print("="*100)
        continue
    search_query = route_result["rewritten_query"]
    print(f"[改写查询] {search_query}")

    # BM25召回
    t1 = time.time()
    bm25_docs = bm25_retriever.retrieve_topk(search_query, topk=10)
    print("BM25召回样例:")
    print(bm25_docs)
    print("="*100)
    t2 = time.time()

    # BGE-M3稠密+稀疏召回+RRF初排
    milvus_docs = milvus_retriever.retrieve_topk(search_query, topk=10)
    print("BGE-M3召回样例:")
    print(milvus_docs)
    print("="*100)
    t3 = time.time()

    # 去重 + 父文档召回
    merged_docs = merge_docs(bm25_docs, milvus_docs)
    print(merged_docs)
    print("="*100)

    # 精排 + 分数过滤（top_score=None 为 API 降级，跳过阈值）
    ranked_docs, top_score = qwen3_reranker.rank(query, merged_docs, topk=5)
    print(f"[Rerank 最高分] {top_score:.4f}" if top_score is not None else "[Rerank 最高分] N/A (API降级)")
    print(ranked_docs)
    print("="*100)

    if top_score is not None and top_score < RERANK_SCORE_THRESHOLD:
        print("未检索到相关法律条文，无法回答该问题。")
        print("="*100)
        continue

    # 答案
    context = "\n".join(["【" + str(idx+1) + "】" + doc.page_content for idx, doc in enumerate(ranked_docs)])
    res_handler = request_chat(query, context, stream=True)
    response = ""
    for r in res_handler:
        uttr = r.choices[0].delta.content
        if not uttr:
            continue
        response += uttr
        print(uttr, end='')
    print("\n" + "="*100)

    # 后处理
    answer = post_processing(response, ranked_docs)
    print("\n答案—>", answer)

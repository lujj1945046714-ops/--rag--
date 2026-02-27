# -*- coding: utf-8 -*-
# Reranker via SiliconFlow API (BAAI/bge-reranker-v2-m3)
# 替代本地 Qwen3-Reranker-4B，接口保持不变

import os
import requests
from langchain_core.documents import Document


SILICONFLOW_API_KEY = os.environ.get("SILICONFLOW_API_KEY", "")
RERANK_API_URL = "https://api.siliconflow.cn/v1/rerank"
RERANK_MODEL = "BAAI/bge-reranker-v2-m3"

TASK = (
    "Given a legal question in Chinese, retrieve the most relevant Chinese legal articles, "
    "provisions, or case analyses that directly answer or apply to the question. "
    "Prefer documents that cite specific law names and article numbers. "
    "Higher-level laws (national law > administrative regulation > local regulation) "
    "should be ranked higher when content relevance is equal."
)


class Qwen3ReRanker:
    """
    调用硅基流动 Reranker API 进行精排，接口与本地版本完全兼容。
    模型：BAAI/bge-reranker-v2-m3
    """

    def __init__(self, model_path=None, max_length=4096, batch_size=32):
        # model_path 参数保留以兼容调用方，API 模式下忽略
        self.api_key = SILICONFLOW_API_KEY
        self.max_length = max_length

    def rank(self, query, candidate_docs, topk=10):
        if not candidate_docs:
            return []

        documents = [doc.page_content for doc in candidate_docs]

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": RERANK_MODEL,
            "query": query,
            "documents": documents,
            "top_n": min(topk, len(documents)),
            "return_documents": False,
        }

        try:
            resp = requests.post(RERANK_API_URL, json=payload, headers=headers, timeout=30)
            resp.raise_for_status()
            results = resp.json().get("results", [])
        except Exception as e:
            print(f"[Reranker] API 调用失败: {e}，降级为原始顺序")
            return candidate_docs[:topk]

        # results: [{"index": int, "relevance_score": float}, ...]
        ranked = sorted(results, key=lambda x: x["relevance_score"], reverse=True)
        return [candidate_docs[r["index"]] for r in ranked[:topk]]


if __name__ == "__main__":
    reranker = Qwen3ReRanker()
    query = "劳动者拒绝违法加班，用人单位能否解除劳动合同"
    docs = [
        Document(page_content="用人单位不得强迫劳动者加班", metadata={}),
        Document(page_content="今天天气不错", metadata={}),
        Document(page_content="《劳动法》第四十一条规定每月加班不超过三十六小时", metadata={}),
    ]
    print(reranker.rank(query, docs, topk=2))

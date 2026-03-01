# -*- coding: utf-8 -*-
# Reranker via SiliconFlow API (Qwen/Qwen3-Reranker-4B)
# Qwen3-Reranker 通过 chat completions + logprobs 打分，接口与旧版完全兼容

import os
import math
import concurrent.futures
from openai import OpenAI
from langchain_core.documents import Document


SILICONFLOW_API_KEY = os.environ.get("SILICONFLOW_API_KEY", "")
RERANK_MODEL = "Qwen/Qwen3-Reranker-4B"
RERANK_BASE_URL = "https://api.siliconflow.cn/v1"

TASK = (
    "Given a legal question in Chinese, retrieve the most relevant Chinese legal articles, "
    "provisions, or case analyses that directly answer or apply to the question. "
    "Prefer documents that cite specific law names and article numbers. "
    "Higher-level laws (national law > administrative regulation > local regulation) "
    "should be ranked higher when content relevance is equal."
)

_SYSTEM_PROMPT = (
    'Judge whether the Document meets the requirements based on the Query and the Instruct provided. '
    'Note that the answer can only be "yes" or "no".'
)

_client = OpenAI(api_key=SILICONFLOW_API_KEY, base_url=RERANK_BASE_URL)


def _score(query: str, doc_text: str) -> float:
    """对单个文档打分，返回 [0, 1] 相关度分数。"""
    user_content = f"<Instruct>: {TASK}\n\n<Query>: {query}\n\n<Document>: {doc_text}"
    try:
        resp = _client.chat.completions.create(
            model=RERANK_MODEL,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user",   "content": user_content},
            ],
            max_tokens=1,
            logprobs=True,
            top_logprobs=20,
            temperature=0,
            extra_body={"enable_thinking": False},
        )
        top_lp = resp.choices[0].logprobs.content[0].top_logprobs
        lp_map = {item.token.strip().lower(): item.logprob for item in top_lp}
        yes_lp = lp_map.get("yes", -100.0)
        no_lp  = lp_map.get("no",  -100.0)
        return math.exp(yes_lp) / (math.exp(yes_lp) + math.exp(no_lp))
    except Exception as e:
        print(f"[Reranker] 评分失败: {e}")
        return 0.0


class Qwen3ReRanker:
    """
    调用硅基流动 Qwen3-Reranker-4B 进行精排。
    使用 chat completions + logprobs 对每个文档打分，线程池并发加速。
    接口与旧版完全兼容：rank() 返回 (List[Document], Optional[float])
    """

    def __init__(self, model_path=None, max_length=4096, batch_size=32):
        self.max_workers = 8  # 并发线程数

    def rank(self, query: str, candidate_docs, topk: int = 10):
        if not candidate_docs:
            return [], 0.0

        # 并发对所有文档打分
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [
                executor.submit(_score, query, doc.page_content)
                for doc in candidate_docs
            ]
            scores = [f.result() for f in futures]

        ranked = sorted(zip(scores, candidate_docs), key=lambda x: x[0], reverse=True)
        top_docs  = [doc for _, doc in ranked[:topk]]
        top_score = ranked[0][0] if ranked else 0.0
        return top_docs, top_score


if __name__ == "__main__":
    reranker = Qwen3ReRanker()
    query = "劳动者拒绝违法加班，用人单位能否解除劳动合同"
    docs = [
        Document(page_content="用人单位不得强迫劳动者加班", metadata={}),
        Document(page_content="今天天气不错", metadata={}),
        Document(page_content="《劳动法》第四十一条规定每月加班不超过三十六小时", metadata={}),
    ]
    result, score = reranker.rank(query, docs, topk=2)
    print(f"top_score: {score:.4f}")
    for doc in result:
        print(doc.page_content)

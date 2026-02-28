# -*- coding: utf-8 -*-
# Milvus 向量检索器（dense-only，embedding 调用硅基流动 BGE-M3 API）
# 稀疏检索由 BM25 负责，此处只保留 dense 向量检索

import os
import time
import hashlib
import requests
import numpy as np
from pymilvus import (
    connections,
    utility,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
)
from langchain_core.documents import Document

from src.client.mongodb_config import MongoConfig


SILICONFLOW_API_KEY = os.environ.get("SILICONFLOW_API_KEY", "")
EMB_API_URL = "https://api.siliconflow.cn/v1/embeddings"
EMB_MODEL = "BAAI/bge-m3"
DENSE_DIM = 1024          # bge-m3 dense 维度
EMB_BATCH = 8             # 降低批次大小避免超过 token 限制
MAX_TEXT_LENGTH = 1024
ID_MAX_LENGTH = 100


def _truncate_to_bytes(text: str, max_bytes: int) -> str:
    """按字节截断字符串，避免超过 Milvus VARCHAR 限制。"""
    encoded = text.encode('utf-8')
    if len(encoded) <= max_bytes:
        return text
    return encoded[:max_bytes].decode('utf-8', errors='ignore')
COL_NAME = "legal_bge_m3_dense"

MILVUS_HOST = os.environ.get("MILVUS_HOST", "localhost")
MILVUS_PORT = int(os.environ.get("MILVUS_PORT", 19530))

mongo_collection = MongoConfig.get_collection("legal_text")
connections.connect(host=MILVUS_HOST, port=MILVUS_PORT)


def _embed_texts(texts: list[str]) -> list[list[float]]:
    """调用硅基流动 BGE-M3 API 获取 dense embedding。"""
    import time
    headers = {
        "Authorization": f"Bearer {SILICONFLOW_API_KEY}",
        "Content-Type": "application/json",
    }
    all_embeddings = []
    for i in range(0, len(texts), EMB_BATCH):
        batch = texts[i: i + EMB_BATCH]
        payload = {"model": EMB_MODEL, "input": batch, "encoding_format": "float"}

        # 添加重试逻辑
        for retry in range(3):
            try:
                resp = requests.post(EMB_API_URL, json=payload, headers=headers, timeout=60)
                resp.raise_for_status()
                data = resp.json()["data"]
                # data 按 index 排序
                data.sort(key=lambda x: x["index"])
                all_embeddings.extend([d["embedding"] for d in data])
                break
            except requests.exceptions.HTTPError as e:
                if retry < 2:
                    print(f"[Embedding] API 错误 {e.response.status_code}，重试 {retry+1}/3...")
                    time.sleep(2 ** retry)  # 指数退避
                else:
                    print(f"[Embedding] API 失败: {e.response.text}")
                    raise

        # 批次间延迟，避免速率限制
        if i + EMB_BATCH < len(texts):
            time.sleep(0.5)

    return all_embeddings


class MilvusRetriever:
    def __init__(self, docs, retrieve=False):
        fields = [
            FieldSchema(name="unique_id", dtype=DataType.VARCHAR,
                        is_primary=True, max_length=ID_MAX_LENGTH),
            FieldSchema(name="text", dtype=DataType.VARCHAR,
                        max_length=MAX_TEXT_LENGTH),
            FieldSchema(name="dense_vector", dtype=DataType.FLOAT_VECTOR,
                        dim=DENSE_DIM),
        ]
        schema = CollectionSchema(fields)

        if not retrieve and utility.has_collection(COL_NAME):
            Collection(COL_NAME).drop()
        self.col = Collection(COL_NAME, schema, consistency_level="Strong")

        dense_index = {"index_type": "AUTOINDEX", "metric_type": "IP"}
        self.col.create_index("dense_vector", dense_index)

        if not retrieve:
            self.save_vectorstore(docs)

        self.col.load()

    def save_vectorstore(self, docs: list):
        # 按字节截断，Milvus VARCHAR max_length 是字节数
        raw_texts = [_truncate_to_bytes(doc.page_content, MAX_TEXT_LENGTH) for doc in docs]
        unique_ids = [doc.metadata["unique_id"][:ID_MAX_LENGTH] for doc in docs]

        print(f"[Milvus] 正在调用 API 计算 {len(raw_texts)} 条 embedding...")
        embeddings = _embed_texts(raw_texts)

        for i in range(0, len(docs), EMB_BATCH):
            batch_texts = raw_texts[i: i + EMB_BATCH]
            batch_ids = unique_ids[i: i + EMB_BATCH]
            batch_embs = embeddings[i: i + EMB_BATCH]
            self.col.insert([batch_ids, batch_texts, batch_embs])

        self.col.flush()
        print(f"[Milvus] 索引构建完成，共 {self.col.num_entities} 条")

    def retrieve_topk(self, query: str, topk: int = 10):
        t1 = time.time()
        query_emb = _embed_texts([query])[0]

        search_params = {"metric_type": "IP", "params": {}}
        results = self.col.search(
            [query_emb],
            anns_field="dense_vector",
            limit=topk,
            output_fields=["unique_id", "text"],
            param=search_params,
        )[0]

        related_docs = []
        for result in results:
            search_res = mongo_collection.find_one({"unique_id": result["id"]})
            if search_res:
                doc = Document(
                    page_content=search_res["page_content"],
                    metadata=search_res["metadata"],
                )
            else:
                # MongoDB 无数据时，直接使用 Milvus 中存储的 text 字段
                doc = Document(
                    page_content=result.entity.get("text", ""),
                    metadata={"unique_id": result["id"]},
                )
            related_docs.append(doc)

        print(f"[Milvus] 检索耗时 {time.time() - t1:.2f}s，返回 {len(related_docs)} 条")
        return related_docs


if __name__ == "__main__":
    retriever = MilvusRetriever(docs=None, retrieve=True)
    results = retriever.retrieve_topk("劳动者拒绝违法加班，用人单位能否解除劳动合同", topk=3)
    for r in results:
        print(r.page_content[:100])
        print("=" * 80)

import argparse
import json
import math
import re
import statistics
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    from rank_bm25 import BM25Okapi as _RankBM25Okapi
except ModuleNotFoundError:
    _RankBM25Okapi = None


TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]")


@dataclass
class Snippet:
    file_path: str
    span: Tuple[int, int]


@dataclass
class QueryCase:
    benchmark: str
    query: str
    snippets: List[Snippet]


@dataclass
class Chunk:
    chunk_id: int
    file_path: str
    start: int
    end: int
    text: str


@dataclass
class EvalResult:
    benchmark: str
    query: str
    precision: float
    recall: float
    retrieved_spans: List[Tuple[str, Tuple[int, int]]]


class SimpleBM25Okapi:
    """A minimal BM25 fallback when rank_bm25 is unavailable."""

    def __init__(self, corpus: List[List[str]], k1: float = 1.5, b: float = 0.75):
        self.corpus = corpus
        self.k1 = k1
        self.b = b
        self.doc_freqs = [Counter(doc) for doc in corpus]
        self.doc_lens = [len(doc) for doc in corpus]
        self.avgdl = (sum(self.doc_lens) / len(self.doc_lens)) if self.doc_lens else 0.0
        self.corpus_size = len(corpus)

        df: Counter = Counter()
        for doc in corpus:
            for token in set(doc):
                df[token] += 1
        self.idf: Dict[str, float] = {}
        for token, freq in df.items():
            self.idf[token] = math.log(1.0 + (self.corpus_size - freq + 0.5) / (freq + 0.5))

    def get_scores(self, query_tokens: List[str]) -> List[float]:
        scores = [0.0] * self.corpus_size
        if not self.corpus_size:
            return scores
        for token in query_tokens:
            idf = self.idf.get(token)
            if idf is None:
                continue
            for i, freqs in enumerate(self.doc_freqs):
                tf = freqs.get(token, 0)
                if tf == 0:
                    continue
                denom = tf + self.k1 * (1.0 - self.b + self.b * self.doc_lens[i] / (self.avgdl + 1e-9))
                scores[i] += idf * (tf * (self.k1 + 1.0) / (denom + 1e-9))
        return scores


def tokenize(text: str) -> List[str]:
    tokens = TOKEN_PATTERN.findall(text.lower())
    return tokens if tokens else [text.lower()]


def read_text(path: Path) -> str:
    for encoding in ("utf-8", "utf-8-sig", "gbk"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    return path.read_text(encoding="utf-8", errors="ignore")


def merge_spans(spans: List[Tuple[int, int]], max_bridge_gap_len: int = 0) -> List[Tuple[int, int]]:
    if not spans:
        return []
    spans = sorted(spans, key=lambda x: x[0])
    merged = [spans[0]]
    for start, end in spans[1:]:
        prev_start, prev_end = merged[-1]
        if start <= prev_end + max_bridge_gap_len:
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))
    return merged


def overlap_len(span_a: Tuple[int, int], span_b: Tuple[int, int]) -> int:
    common_min = max(span_a[0], span_b[0])
    common_max = min(span_a[1], span_b[1])
    return max(0, common_max - common_min)


def load_benchmark_cases(benchmarks_dir: Path) -> List[QueryCase]:
    cases: List[QueryCase] = []
    for benchmark_file in sorted(benchmarks_dir.glob("*.json")):
        data = json.loads(read_text(benchmark_file))
        tests = data.get("tests", [])
        benchmark_name = benchmark_file.stem
        for item in tests:
            query = item.get("query", "").strip()
            snippets_data = item.get("snippets", [])
            snippets: List[Snippet] = []
            for snip in snippets_data:
                file_path = snip.get("file_path")
                span = snip.get("span")
                if (
                    isinstance(file_path, str)
                    and isinstance(span, list)
                    and len(span) == 2
                    and isinstance(span[0], int)
                    and isinstance(span[1], int)
                ):
                    snippets.append(Snippet(file_path=file_path, span=(span[0], span[1])))
            if query and snippets:
                cases.append(QueryCase(benchmark=benchmark_name, query=query, snippets=snippets))
    return cases


def chunk_documents_legal(corpus_dir: Path) -> List[Chunk]:
    """使用 legal_texts_split 切分，用于对比测试"""
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from langchain_core.documents import Document as LCDoc
    from src.parser.legal_splitter import legal_texts_split

    raw_docs = []
    for file_path in sorted(p for p in corpus_dir.rglob("*.md") if p.is_file()):
        rel_path = file_path.relative_to(corpus_dir).as_posix()
        content = read_text(file_path)
        if not content:
            continue
        raw_docs.append(LCDoc(
            page_content=content,
            metadata={
                "title": file_path.stem,
                "category": file_path.parts[-3] if len(file_path.parts) >= 3 else "",
                "source": str(file_path),
                "doc_type": "open_source_laws",
                "page": 1,
                "images_info": [],
            }
        ))

    _, child_docs = legal_texts_split(raw_docs)

    chunks = []
    for i, doc in enumerate(child_docs):
        # 从原始文件重新定位 span
        src_path = doc.metadata.get("source", "")
        try:
            src_file = Path(src_path)
            full_text = read_text(src_file)
            chunk_text = doc.page_content
            # 去掉 contextual prefix 后查找原始位置
            prefix_end = chunk_text.find("\n") + 1 if chunk_text.startswith("【") else 0
            search_text = chunk_text[prefix_end:].strip()[:100]
            start = full_text.find(search_text)
            if start < 0:
                start = 0
            end = min(start + len(chunk_text), len(full_text))
            rel_path = src_file.relative_to(corpus_dir).as_posix()
        except Exception:
            start, end = 0, len(doc.page_content)
            rel_path = doc.metadata.get("source", f"chunk_{i}")

        chunks.append(Chunk(
            chunk_id=i,
            file_path=rel_path,
            start=start,
            end=end,
            text=doc.page_content,
        ))

    return chunks


def chunk_documents(corpus_dir: Path, chunk_size: int, chunk_overlap: int) -> List[Chunk]:
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap 必须小于 chunk_size")

    step = chunk_size - chunk_overlap
    chunks: List[Chunk] = []
    chunk_id = 0
    for file_path in sorted(p for p in corpus_dir.rglob("*") if p.is_file()):
        rel_path = file_path.relative_to(corpus_dir).as_posix()
        content = read_text(file_path)
        if not content:
            continue
        for start in range(0, len(content), step):
            end = min(start + chunk_size, len(content))
            text = content[start:end]
            if not text:
                continue
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    file_path=rel_path,
                    start=start,
                    end=end,
                    text=text,
                )
            )
            chunk_id += 1
            if end >= len(content):
                break
    return chunks


def bm25_topk(bm25, chunks: List[Chunk], query: str, topk: int) -> List[Chunk]:
    query_tokens = tokenize(query)
    scores = bm25.get_scores(query_tokens)
    indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:topk]
    return [chunks[i] for i in indices]


def maybe_rerank_with_qwen3(
    query: str,
    candidate_chunks: List[Chunk],
    model_path: Optional[str],
    topk: int,
) -> List[Chunk]:
    if not model_path:
        return candidate_chunks[:topk]

    from src.reranker.qwen3_reranker import Qwen3ReRanker

    @dataclass
    class _Doc:
        page_content: str
        metadata: dict

    reranker = Qwen3ReRanker(model_path=model_path)
    docs = [_Doc(page_content=chunk.text, metadata={"chunk": chunk}) for chunk in candidate_chunks]
    ranked_docs = reranker.rank(query, docs, topk=topk)
    return [doc.metadata["chunk"] for doc in ranked_docs]


def score_query(gt_snippets: List[Snippet], retrieved_chunks: List[Chunk]) -> Tuple[float, float, List[Tuple[str, Tuple[int, int]]]]:
    gt_by_file: Dict[str, List[Tuple[int, int]]] = {}
    for snip in gt_snippets:
        gt_by_file.setdefault(snip.file_path, []).append(snip.span)
    for key in list(gt_by_file.keys()):
        gt_by_file[key] = merge_spans(gt_by_file[key], max_bridge_gap_len=0)

    ret_by_file: Dict[str, List[Tuple[int, int]]] = {}
    for chunk in retrieved_chunks:
        ret_by_file.setdefault(chunk.file_path, []).append((chunk.start, chunk.end))
    for key in list(ret_by_file.keys()):
        ret_by_file[key] = merge_spans(ret_by_file[key], max_bridge_gap_len=0)

    total_retrieved_len = sum(end - start for spans in ret_by_file.values() for start, end in spans)
    total_relevant_len = sum(end - start for spans in gt_by_file.values() for start, end in spans)

    relevant_retrieved_len = 0
    for file_path, ret_spans in ret_by_file.items():
        gt_spans = gt_by_file.get(file_path, [])
        if not gt_spans:
            continue
        for ret_span in ret_spans:
            for gt_span in gt_spans:
                relevant_retrieved_len += overlap_len(ret_span, gt_span)

    precision = (relevant_retrieved_len / total_retrieved_len) if total_retrieved_len > 0 else 0.0
    recall = (relevant_retrieved_len / total_relevant_len) if total_relevant_len > 0 else 0.0

    flattened = [(fp, sp) for fp, spans in ret_by_file.items() for sp in spans]
    return precision, recall, flattened


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate retrieval on LegalBench-RAG")
    parser.add_argument(
        "--legalbench-root",
        type=str,
        required=True,
        help="LegalBench-RAG 数据根目录，目录下应包含 corpus/ 和 benchmarks/",
    )
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--chunk-overlap", type=int, default=0)
    parser.add_argument("--bm25-topk", type=int, default=20, help="BM25 初召回数量")
    parser.add_argument("--final-topk", type=int, default=10, help="最终评测用片段数")
    parser.add_argument(
        "--qwen3-reranker-model-path",
        type=str,
        default=None,
        help="可选，传入后启用 Qwen3-Reranker 对 BM25 结果重排",
    )
    parser.add_argument("--save-json", type=str, default=None, help="结果输出文件路径")
    parser.add_argument(
        "--use-legal-splitter",
        action="store_true",
        help="使用 legal_texts_split 切分（法律专用），替代固定 chunk_size 切分",
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        default=None,
        help="只评测指定 benchmark（文件名不含.json），不传则评测全部",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    root = Path(args.legalbench_root)
    corpus_dir = root / "corpus"
    benchmarks_dir = root / "benchmarks"
    if not corpus_dir.exists():
        raise FileNotFoundError(f"未找到目录: {corpus_dir}")
    if not benchmarks_dir.exists():
        raise FileNotFoundError(f"未找到目录: {benchmarks_dir}")

    print("加载测试集...")
    cases = load_benchmark_cases(benchmarks_dir)
    if args.benchmark:
        cases = [c for c in cases if c.benchmark == args.benchmark]
    print(f"测试问题数: {len(cases)}")

    print("构建分块...")
    if args.use_legal_splitter:
        chunks = chunk_documents_legal(corpus_dir)
        print(f"分块数（legal_texts_split）: {len(chunks)}")
    else:
        chunks = chunk_documents(corpus_dir, chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)
        print(f"分块数（固定切割 chunk_size={args.chunk_size}）: {len(chunks)}")

    print("构建 BM25 索引...")
    tokenized_chunks = [tokenize(chunk.text) for chunk in chunks]
    bm25 = _RankBM25Okapi(tokenized_chunks) if _RankBM25Okapi is not None else SimpleBM25Okapi(tokenized_chunks)

    print("开始评测...")
    results: List[EvalResult] = []
    for i, case in enumerate(cases, start=1):
        candidates = bm25_topk(bm25, chunks, case.query, topk=args.bm25_topk)
        retrieved_chunks = maybe_rerank_with_qwen3(
            case.query,
            candidates,
            model_path=args.qwen3_reranker_model_path,
            topk=args.final_topk,
        )
        precision, recall, retrieved_spans = score_query(case.snippets, retrieved_chunks)
        results.append(
            EvalResult(
                benchmark=case.benchmark,
                query=case.query,
                precision=precision,
                recall=recall,
                retrieved_spans=retrieved_spans,
            )
        )
        if i % 50 == 0:
            print(f"已完成: {i}/{len(cases)}")

    per_benchmark: Dict[str, Dict[str, float]] = {}
    grouped: Dict[str, List[EvalResult]] = {}
    for r in results:
        grouped.setdefault(r.benchmark, []).append(r)

    for benchmark, items in grouped.items():
        per_benchmark[benchmark] = {
            "count": len(items),
            "precision": statistics.fmean([x.precision for x in items]) if items else 0.0,
            "recall": statistics.fmean([x.recall for x in items]) if items else 0.0,
        }

    overall_precision = statistics.fmean([x.precision for x in results]) if results else 0.0
    overall_recall = statistics.fmean([x.recall for x in results]) if results else 0.0

    benchmark_weighted_precision = 0.0
    benchmark_weighted_recall = 0.0
    if per_benchmark:
        weight = 1.0 / len(per_benchmark)
        benchmark_weighted_precision = sum(v["precision"] * weight for v in per_benchmark.values())
        benchmark_weighted_recall = sum(v["recall"] * weight for v in per_benchmark.values())

    report = {
        "config": {
            "legalbench_root": str(root),
            "chunk_size": args.chunk_size,
            "chunk_overlap": args.chunk_overlap,
            "bm25_topk": args.bm25_topk,
            "final_topk": args.final_topk,
            "qwen3_reranker_model_path": args.qwen3_reranker_model_path,
        },
        "summary": {
            "query_count": len(results),
            "overall_precision": overall_precision,
            "overall_recall": overall_recall,
            "benchmark_weighted_precision": benchmark_weighted_precision,
            "benchmark_weighted_recall": benchmark_weighted_recall,
        },
        "per_benchmark": per_benchmark,
    }

    save_json = args.save_json
    if not save_json:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        save_json = f"data/eval/legalbenchrag_eval_{timestamp}.json"
    output_path = Path(save_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 80)
    print(f"总体 Precision: {overall_precision:.4f}")
    print(f"总体 Recall:    {overall_recall:.4f}")
    print(f"基准均权 Precision: {benchmark_weighted_precision:.4f}")
    print(f"基准均权 Recall:    {benchmark_weighted_recall:.4f}")
    print(f"结果已保存: {output_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()

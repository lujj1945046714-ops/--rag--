FROM python:3.10-slim

WORKDIR /app

# 系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ git curl \
    && rm -rf /var/lib/apt/lists/*

# 先安装依赖（利用 Docker 层缓存）
COPY requirements-docker.txt .
RUN pip install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple \
    -r requirements-docker.txt

# 复制源码
COPY src/ ./src/
COPY build_index.py infer.py final_score.py \
     generate_legal_qa_pairs.py collect_external_benchmarks.py \
     evaluate_legalbench_rag.py generate_legal_benchmark.py \
     collect_legal_docs.py ./

# 数据目录（挂载点）
RUN mkdir -p data/qa_pairs data/saved_index data/mongodb models .cache

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

CMD ["python", "infer.py"]

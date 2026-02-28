# 中文法律知识问答系统 Legal RAG

> 基于检索增强生成（RAG）技术的中文法律领域智能问答系统，针对法律文本的层级结构、条文引用关系和法理推理逻辑进行了深度定制优化。

---

## 目录

- [项目简介](#项目简介)
- [数据集来源](#数据集来源)
- [系统架构](#系统架构)
- [核心亮点](#核心亮点)
- [技术栈](#技术栈)
- [目录结构](#目录结构)
- [环境配置](#环境配置)
- [快速开始](#快速开始)
- [评测](#评测)

---

## 项目简介

本项目是一个面向中文法律领域的检索增强生成（RAG）问答系统。用户输入法律问题，系统从 3500+ 篇法律文件中精准检索相关条文，结合大语言模型生成引用条文原文、说明适用层级和时效的专业法律回答。

**与通用 RAG 系统的核心差异：**

| 维度 | 通用 RAG | 本系统 |
|------|---------|--------|
| 分块策略 | 固定 token 数切割 | 按条文边界 + 章节结构切割 |
| 索引结构 | 单层 flat 索引 | Parent-Child 双层索引 |
| 上下文 | 无结构前缀 | 法律名·编·章 上下文前缀 |
| 引用处理 | 不处理跨条文引用 | 自动扩展被引用条文 |
| 案例处理 | 随机切割 | 整体保留完整推理链 |
| 生成约束 | 通用 prompt | 强制引用条文原文 + 层级说明 |

---

## 数据集来源

### 法律语料

**[LawRefBook/Laws](https://github.com/LawRefBook/Laws)**（开源中文法律库）

| 分类 | 文件数 | 说明 |
|------|--------|------|
| 地方法规（DLC） | 1,918 | 山东、河南、浙江、上海等省市地方性法规 |
| 行政法规 | 711 | 国务院行政法规 |
| 司法解释 | 440 | 最高人民法院、最高人民检察院司法解释 |
| 民法典 | 7 | 民法典各编 |
| 刑法 | 若干 | 刑法及修正案 |
| 案例 | 52 | 劳动人事、民法典、消费购物、行政协议诉讼典型案例 |
| 其他 | 若干 | 宪法、经济法、社会法、部门规章等 |
| **合计** | **3,500+** | 时间跨度 1979—2024 年 |

数据格式统一为 Markdown，命名规范为 `法律名称(YYYY-MM-DD).md`，包含完整章节和条文结构。

### 评测数据集

自动从案例文件提取构建，共 **52 条**测试问题，覆盖 4 个法律领域：

- 劳动人事（10条）
- 民法典（14条）
- 消费购物（18条）
- 行政协议诉讼（10条）

---

## 系统架构

```
用户问题
    │
    ▼
┌─────────────────────────────────────────┐
│           问题路由 + 查询改写            │
│  route_and_rewrite()                    │
│  · 非法律问题 → 直接返回"无答案"        │
│  · 法律问题 → 改写为检索友好查询        │
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│              检索层                      │
│  BM25 召回（topk=10）                   │
│       +                                 │
│  BGE-M3 向量召回（dense+sparse+RRF）    │
│       ↓                                 │
│  merge_docs（去重 + Parent-Child 召回） │
│       ↓                                 │
│  BGE-Reranker-v2-m3 精排（topk=8）     │
│       ↓                                 │
│  分数过滤（top_score < 0.3 → 无答案）  │
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│              生成层                      │
│  法律专用 Prompt（引用条文 + 层级说明） │
│  无关内容处理指令（防幻觉）             │
│  大语言模型（豆包/DeepSeek/GPT-4）      │
└─────────────────────────────────────────┘
    │
    ▼
答案（含条文引用编号 + 来源页码）
```

### Parent-Child 双层索引

```
父层（章节级，存 MongoDB）
  《民法典》侵权责任编·第三章 责任主体的特殊规定
  第1188条 ~ 第1201条（整章，约800字）

子层（滑动窗口，存 Milvus + BM25）
  chunk_A: 第1188~1191条  ←── 检索命中
  chunk_B: 第1191~1194条
  chunk_C: 第1194~1197条
       ↓ parent_id 关联
  返回整章父 doc 给 LLM
```

---

## 核心亮点

### 1. 法律专用分块策略

针对三种文档类型分别处理：

**法律条文**（民法典、刑法、行政法规、司法解释）
- 按 `## 第X章` 切分章节，章节整体作为父 doc
- 章节内按 `第X条` 边界滑动窗口（window=4, overlap=1）生成子 doc
- 短于 50 字的条文自动合并到相邻条文，避免碎片化

**案例文档**（基本案情 + 案例分析 + 典型意义）
- 整体保留为单个 chunk，不切分
- 保留完整法律推理链：事实 → 条文引用 → 推理 → 结论

**司法解释 / 地方法规**
- 前言段落独立成 chunk
- 条文部分同法律条文处理方式

### 2. Contextual Prefix（上下文前缀）

每个子 chunk 入库前自动添加结构化前缀，提升 embedding 质量：

```
【中华人民共和国民法典·侵权责任编·第三章 责任主体的特殊规定】
第一千一百九十一条 用人单位的工作人员因执行工作任务造成他人损害的...
```

### 3. 条文内引用自动扩展

检测 chunk 中的跨条文引用（如"适用本法第X条"），自动追加被引用条文：

```
第一千一百七十六条 ...活动组织者的责任适用本法第一千一百九十八条至第一千二百零一条的规定。

【引用条文】
第一千一百九十八条 宾馆、商场...经营者未尽到安全保障义务...
第一千一百九十九条 无民事行为能力人在幼儿园...
```

### 4. 法律专用 Reranker 指令

Qwen3-Reranker 的 task 指令针对法律场景定制：

```
Given a legal question in Chinese, retrieve the most relevant Chinese legal articles,
provisions, or case analyses that directly answer or apply to the question.
Prefer documents that cite specific law names and article numbers.
Higher-level laws (national law > administrative regulation > local regulation)
should be ranked higher when content relevance is equal.
```

### 5. 问题路由 + 查询改写

一次 LLM 调用同时完成两个任务：

1. **路由判断**：识别非法律问题（天气、数学等），直接返回"无答案"，不进入检索流程，节省资源
2. **查询改写**：将口语化问题改写为检索友好的法律关键词查询（≤20字），提升 BM25 和向量检索的召回质量

```python
# 示例
输入：  "我被公司开除了怎么办"
改写后："劳动合同违法解除赔偿金"
```

### 6. Rerank 分数过滤

精排后若最高相关分 < 0.3，直接返回"无答案"，避免将低质量检索结果送入 LLM 生成错误答案。API 降级时自动跳过过滤，保证可用性。

### 7. 法理约束生成

LLM 生成阶段强制要求：
1. 引用相关条文原文（格式：《法律名称》第X条）
2. 说明条文适用层级（国家法律 / 行政法规 / 地方法规）和生效时间
3. 如多条法律存在冲突，说明适用优先级
4. 给出简明结论

---

## 技术栈

| 模块 | 技术选型 |
|------|---------|
| 向量检索 | Milvus-Lite + BGE-M3（dense + sparse + RRF） |
| 关键词检索 | BM25（rank-bm25） |
| 父子文档存储 | MongoDB 7.0 |
| 精排模型 | Qwen3-Reranker-4B |
| 嵌入模型 | BGE-M3 / Qwen3-Embedding-0.6B |
| 大语言模型 | 豆包 / DeepSeek / GPT-4（OpenAI 兼容接口） |
| RAG 评测 | RAGAS（LLMContextRecall + LLMContextPrecisionWithReference） |
| 语义评测 | text2vec + 关键词 Jaccard 加权 |
| 框架 | LangChain + FastAPI |
| 语义切分服务 | FastAPI（本地微服务） |

---

## 目录结构

```
法律rag系统/
├── build_index.py              # 构建检索索引（BM25 + Milvus）
├── infer.py                    # 交互式问答推理
├── final_score.py              # 完整评测（语义相似度 + RAGAS）
├── evaluate_legalbench_rag.py  # 检索层评测（Precision / Recall）
├── generate_legal_benchmark.py # 自动生成中文法律评测集
├── generate_sft_data.py        # 生成 SFT 微调训练数据
├── collect_legal_docs.py       # 采集法律文档
├── config.ini                  # 环境变量配置模板
├── requirements.txt            # 依赖列表
│
├── src/
│   ├── parser/
│   │   ├── legal_splitter.py       # 法律专用分块器（核心）
│   │   ├── legal_source_ingest.py  # 法律文档加载（Laws 仓库）
│   │   └── pdf_parse.py            # PDF 文档加载
│   ├── retriever/
│   │   ├── bm25_retriever.py       # BM25 检索器
│   │   ├── milvus_retriever.py     # Milvus 向量检索器
│   │   └── faiss_retriever.py      # Faiss 检索器
│   ├── reranker/
│   │   └── qwen3_reranker.py       # Qwen3 精排模型
│   ├── client/
│   │   ├── llm_chat_client.py      # LLM 问答客户端（法律专用 prompt）
│   │   ├── llm_router_client.py    # 问题路由 + 查询改写（route_and_rewrite）
│   │   └── llm_hyde_client.py      # HyDE 假设文档生成
│   └── utils.py                    # merge_docs（Parent-Child 召回）
│
├── data/
│   ├── qa_pairs/                   # 测试问答对
│   ├── eval/                       # 评测结果
│   └── mongodb/                    # MongoDB 数据目录
│
└── LegalBench-RAG/
    ├── benchmarks/
    │   └── chinese_legal_cases.json  # 中文法律评测集（52条）
    └── corpus/
        └── 案例/                     # 评测语料
```

---

## 环境配置

### 1. 基础环境

```bash
# Python 3.10+
conda create -n rag python=3.10
conda activate rag

pip install -r requirements.txt
```

### 2. 环境变量

```bash
export PYTHONPATH=$PYTHONPATH:$PWD

# LLM API（豆包 / DeepSeek / 任意 OpenAI 兼容接口）
export DOUBAO_API_KEY="your_api_key"
export DOUBAO_BASE_URL="https://ark.cn-beijing.volces.com/api/v3"
export DOUBAO_MODEL_NAME="your_model_endpoint"

# 硅基流动 API（embedding + reranker，注册：https://siliconflow.cn）
export SILICONFLOW_API_KEY="your_siliconflow_api_key"

# HuggingFace 镜像（国内加速）
export HF_ENDPOINT=https://hf-mirror.com
```

### 3. 启动依赖服务

```bash
# 启动 MongoDB
mongod --port=27017 --dbpath=data/mongodb/data \
       --logpath=data/mongodb/log/mongodb.log \
       --bind_ip=0.0.0.0 --fork

# 启动语义切分服务
nohup python src/client/semantic_chunk.py > log/semantic_chunk.log 2>&1 &
sleep 10
```

### 4. 模型说明

BGE-M3（embedding）和 Reranker 均通过**硅基流动 API** 调用，无需本地下载大模型。

只需下载用于评测的 text2vec 模型（约 400MB）：

```bash
# text2vec 语义相似度评测模型（仅 final_score.py 需要）
python -c "
import os; os.environ['HF_ENDPOINT']='https://hf-mirror.com'
from huggingface_hub import snapshot_download
snapshot_download('shibing624/text2vec-base-chinese', local_dir='models/text2vec-base-chinese')
"
```

---

## 快速开始

### Step 1：准备法律语料

```bash
# 方式一：使用本地已有的 Laws 仓库
python build_index.py --source laws --laws-dir /path/to/Laws

# 方式二：自动 clone Laws 仓库
python build_index.py --source laws
```

### Step 2：构建检索索引

```bash
# 构建 BM25 + Milvus 索引（法律专用切分）
python build_index.py \
    --source laws \
    --laws-dir .cache/laws_repo \
    --demo-query "劳动者拒绝违法加班，用人单位能否解除劳动合同"
```

输出示例：
```
原始文档数: 3502
父文档数（章节级，存MongoDB）: 12847
切分后文档总数: 48632
BM25召回样例: [...]
BGE-M3召回样例: [...]
```

### Step 3：启动问答

```bash
python infer.py
```

```
输入—> 劳动者拒绝违法超时加班，用人单位能否以不符合录用条件为由解除劳动合同？

BM25召回样例: [...]
BGE-M3召回样例: [...]
精排结果: [...]

根据《中华人民共和国劳动法》第四十一条规定，用人单位延长工作时间每日不得超过三小时，
每月不得超过三十六小时。某快递公司规章制度中"早9时至晚9时，每周工作6天"严重违反
法律规定，应认定为无效。

劳动者拒绝违法超时加班系维护自身合法权益，不能据此认定其不符合录用条件，
用人单位解除劳动合同属于违法解除，应支付赔偿金。【1, 3】
```

---

## 评测

### 检索层评测（Precision / Recall）

```bash
# 生成中文法律评测集
python generate_legal_benchmark.py \
    --laws-dir .cache/laws_repo \
    --output-dir LegalBench-RAG

# 运行检索评测
python evaluate_legalbench_rag.py \
    --legalbench-root LegalBench-RAG \
    --benchmark chinese_legal_cases \
    --use-legal-splitter \
    --bm25-topk 20 \
    --final-topk 5
```

### 端到端评测（RAGAS）

评测集默认使用项目内置的 52 条案例问答对，也可接入外部开源数据集扩充。

**生成内置评测集：**
```bash
python generate_legal_qa_pairs.py
```

**接入外部数据集（可选）：**

| 数据集 | HuggingFace ID | 任务 |
|--------|---------------|------|
| LawBench | `doolayer/LawBench` | 法条适用（3-1）+ 罪名预测（3-3） |
| JEC-QA | `hails/agieval-jec-qa-kd` | 司法考试单选题 |

两个数据集均自动从 HuggingFace 下载，无需本地文件：

```bash
# 下载 LawBench + JEC-QA，各取 100 条，追加到评测集
python collect_external_benchmarks.py --lawbench --jecqa --max-per-source 100

# 覆盖原评测集（不追加）
python collect_external_benchmarks.py --lawbench --jecqa --overwrite
```

**运行评测：**
```bash
python final_score.py
```

评测指标：
- **语义相似度 + 关键词 Jaccard 加权得分**（text2vec）
- **LLMContextRecall**：上下文召回率（RAGAS）
- **LLMContextPrecisionWithReference**：上下文精确率（RAGAS）

---

## 参考资料

- [LawRefBook/Laws](https://github.com/LawRefBook/Laws) — 开源中文法律库
- [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3) — 多功能向量模型
- [Qwen3-Reranker](https://huggingface.co/Qwen/Qwen3-Reranker-4B) — 精排模型
- [RAGAS](https://docs.ragas.io/) — RAG 评测框架
- [LegalBench-RAG](https://github.com/ZeroEntropy-AI/legalbench-rag) — 法律 RAG 评测基准

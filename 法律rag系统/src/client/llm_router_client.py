import os
import json
import src.constant  # 触发 .env 加载
from openai import OpenAI

ROUTER_REWRITE_PROMPT = """你是一个中国法律助手。请完成以下两个任务：
1. 判断问题是否与中国法律、法规、司法实践或法律考试相关
2. 如果相关，将问题改写为适合检索法律条文的简洁查询（20字以内，突出法律关键词）

以JSON格式返回，不要有其他内容：
{{"is_legal": true, "rewritten_query": "改写后的查询"}}
或
{{"is_legal": false, "rewritten_query": ""}}

问题：{query}"""

_api_key  = os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("DOUBAO_API_KEY")
_base_url = os.environ.get("DEEPSEEK_BASE_URL") or os.environ.get("DOUBAO_BASE_URL")
_model    = os.environ.get("DEEPSEEK_MODEL_NAME") or os.environ.get("DOUBAO_MODEL_NAME")
_client = OpenAI(api_key=_api_key, base_url=_base_url)


def route_and_rewrite(query: str) -> dict:
    """
    返回 {"is_legal": bool, "rewritten_query": str}
    一次 LLM 调用同时完成路由判断和查询改写
    """
    prompt = ROUTER_REWRITE_PROMPT.format(query=query)
    try:
        resp = _client.chat.completions.create(
            model=_model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=100,
            temperature=0,
        )
        content = resp.choices[0].message.content.strip()
        # 用 raw_decode 做平衡括号提取，兼容嵌套 JSON 和 markdown 包裹
        start = content.find('{')
        if start == -1:
            return {"is_legal": True, "rewritten_query": query}
        result, _ = json.JSONDecoder().raw_decode(content, start)
        # 严格 bool 解析：仅接受 bool 或白名单字符串
        raw = result.get("is_legal")
        _STR_MAP = {"true": True, "false": False, "1": True, "0": False, "yes": True, "no": False, "是": True, "否": False}
        if isinstance(raw, bool):
            is_legal = raw
        elif isinstance(raw, str) and raw.strip().lower() in _STR_MAP:
            is_legal = _STR_MAP[raw.strip().lower()]
        else:
            # 未知值：保守地视为法律问题（fail-open）
            is_legal = True
        return {
            "is_legal": is_legal,
            "rewritten_query": result.get("rewritten_query", query) or query
        }
    except Exception:
        # 解析失败时默认视为法律问题，使用原始查询
        return {"is_legal": True, "rewritten_query": query}

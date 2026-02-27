import os
import json
import re
from openai import OpenAI
from langchain_core.documents import Document
import src.constant  # 触发 .env 加载

LLM_HYDE_PROMPT = """
你是一位Tesla汽车专家，现在请你结合Model 3车辆和新能源电动汽车相关知识回答下列问题.
请给出用户问题的使用方法，详细分析问题原因，返回有用的内容。
{query}
最终的回答请尽可能的精简, 不超过100字:
"""

_api_key  = os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("DOUBAO_API_KEY")
_base_url = os.environ.get("DEEPSEEK_BASE_URL") or os.environ.get("DOUBAO_BASE_URL")
_model    = os.environ.get("DEEPSEEK_MODEL_NAME") or os.environ.get("DOUBAO_MODEL_NAME")

llm_client = OpenAI(api_key=_api_key, base_url=_base_url)


def request_hyde(query):

    prompt = LLM_HYDE_PROMPT.format(query=query)

    completion = llm_client.chat.completions.create(
        model=_model,
        messages=[
            {"role": "system", "content": "你是一个有用的人工智能助手."},
            {"role": "user", "content": prompt}
        ],
        top_p=0,
        temperature=0.001
    )
    return completion.choices[0].message.content


if __name__ == "__main__":
    query = "劳动者拒绝违法加班，用人单位能否解除劳动合同"
    res = request_hyde(query)
    print(res)

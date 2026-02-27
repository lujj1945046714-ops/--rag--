import os
import src.constant  # 触发 .env 加载
from openai import OpenAI


LLM_CHAT_PROMPT = """
### 参考法律条文
{context}

### 任务
你是一个专业的中国法律问答助手，请根据上方【参考法律条文】回答以下问题：

"{query}"

回答要求：
1. 先引用相关条文原文（格式：《法律名称》第X条）
2. 说明该条文的适用层级（国家法律/行政法规/地方法规）和生效时间（如已知）
3. 如多条法律存在冲突，说明适用优先级
4. 最后给出简明结论

输出格式：{{答案}}【{{引用编号1}}, {{引用编号2}}, ...】
如果参考条文中无法找到答案，请说"无答案"，不允许编造法律条文。
"""

_api_key  = os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("DOUBAO_API_KEY")
_base_url = os.environ.get("DEEPSEEK_BASE_URL") or os.environ.get("DOUBAO_BASE_URL")
_model    = os.environ.get("DEEPSEEK_MODEL_NAME") or os.environ.get("DOUBAO_MODEL_NAME")

llm_client = OpenAI(api_key=_api_key, base_url=_base_url)


def request_chat(query, context, stream=False):

    prompt = LLM_CHAT_PROMPT.format(context=context, query=query) 

    completion = llm_client.chat.completions.create(
        model=_model,
        messages=[
            {"role": "system", "content": "你是一个专业的中国法律助手，熟悉中华人民共和国各类法律法规、司法解释及地方性法规。回答时必须以法律条文为依据，不得编造法律内容。"},
            {"role": "user", "content": prompt}
        ],
        max_tokens=4096,
        stream=stream
    )
    if stream:
        return completion

    return completion.choices[0].message.content



if __name__ == "__main__":

    context = """
    【1】### 离车后自动上锁
    带着手机钥匙或配对的遥控钥匙离开时，车门和行李箱可以自动锁定（如果订购日期是在大约 2019 年 10 月 1 日之后）。要打开或关闭此功能，可点击控制 > 车锁 > 离车后自动上锁。
    **注**：如果已将 Apple 手表认证为钥匙，也可以将该手表用于离车后自动上锁功能。
    【2】车门锁闭时，外部车灯闪烁一次，后视镜折叠（如果折叠后视镜开启）。要在 Model 3 锁定时听到提示音，可点击控制 > 车锁 > 锁定提示音。
    【3】### 大灯延时照明
    停止驾驶并将 Model 3 停在照明较差的环境中时，外部车灯会短暂亮起。它们会在一分钟后或您锁闭 Model 3 时（以较早者为准）自动关闭。当您使用 Tesla 手机应用程序锁定 Model 3 时，大灯将立即熄灭。但是，如果车辆因启用了“离车后自动上锁”功能而锁定（请参阅离车后自动上锁 页码 7），则大灯将在一分钟后自动熄灭。要打开或关闭此功能，请点击控制 > 车灯 > 大灯延时照明。关闭大灯延时照明后，当换入驻车挡并打开车门时，大灯会立即熄灭。"""

    query = "介绍一下离车后自动上锁功能"

    res = request_chat(query, context, stream=True)
    for chunk in res:
        delta = chunk.choices[0].delta.content
        if delta:
            print(delta, end="")
    print()

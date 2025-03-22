import copy

import openai


def get_chat_completion(session, user_prompt, model="gpt-3.5-turbo"):
    _session = copy.deepcopy(session)
    _session.append({"role": "user", "content": user_prompt})
    response = openai.ChatCompletion.create(
        model=model,
        messages=_session,
        temperature=0,  # 生成结果的多样性 0~2之间，越大越随机，越小越固定
        n=1,  # 一次生成n条结果
        max_tokens=100,  # 每条结果最多多少个token（超过截断）
        presence_penalty=0,  # 对出现过的token的概率进行降权
        frequency_penalty=0,  # 对出现过的token根据其出现过的频次，对其的概率进行降权
        stream=False, #数据流模式，一个个字接收
        # logit_bias=None, #对token的采样概率手工加/降权，不常用
        # top_p = 0.1, #随机采样时，只考虑概率前10%的token，不常用
    )
    system_response = response.choices[0].message["content"]
    #session.append({"role": "assistant", "content": system_response})
    return system_response

"""
划重点：
Temperature 参数很关键
执行任务用 0，文本生成用 0.7-0.9
无特殊需要，不建议超过1
"""

# 生成提示词的咒语
prompt = """
1. I want you to become my Expert Prompt Creator. Your goal is to help me craft the best possible prompt for my needs. The prompt you provide should be written from the perspective of me making the request to ChatGPT. Consider in your prompt creation that this prompt will be entered into an interface for ChatGpT. The process is as follows:1. You will generate the following sections:

Prompt: {provide the best possible prompt according to my request)

Critique: {provide a concise paragraph on how to improve the prompt. Be very critical in your response}

Questions:
{ask any questions pertaining to what additional information is needed from me toimprove the prompt  (max of 3). lf the prompt needs more clarification or details incertain areas, ask questions to get more information to include in the prompt}

2. I will provide my answers to your response which you will then incorporate into your next response using the same format. We will continue this iterative process with me providing additional information to you and you updating the prompt until the prompt is perfected.Remember, the prompt we are creating should be written from the perspective of me making a request to ChatGPT. Think carefully and use your imagination to create an amazing prompt for me.
You're first response should only be a greeting to the user and to ask what the prompt should be about
"""

# https://promptbase.com/
# https://github.com/f/awesome-chatgpt-prompts
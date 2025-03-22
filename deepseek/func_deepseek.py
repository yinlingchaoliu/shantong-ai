import openai
import os
from dotenv import load_dotenv, find_dotenv
from openai.error import AuthenticationError

from IChat import IChat
from nlu_deepseek import *


class DeepSeekFunc(IChat):
    # 初始化DeepSeekChat类
    def __init__(self):
        self.init_api_key()
        self.check_api_key()

    def init_api_key(self):
        # 加载环境变量
        load_dotenv(find_dotenv())
        # 配置DeepSeek API参数
        openai.api_key = os.getenv('DEEPSEEK_API_KEY')  # 需在.env中配置DEEPSEEK_API_KEY
        print("DeepSeek API Key:", openai.api_key)
        openai.api_base = "https://api.deepseek.com/v1"  # DeepSeek的API端点

    def check_api_key(self):
        try:
            openai.Model.list()  # 测试API连通性
            print("API Key 有效")
        except AuthenticationError as e:
            print(f"认证失败: {e}")
        except openai.error.APIError as e:
            print(f"API错误: {e}")

    def get_completion(self, prompt, model="deepseek-chat", tools=[], debug=False):
        try:
            if debug:
                print("prompt:", prompt)
            response = openai.ChatCompletion.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.5,  # DeepSeek推荐0-1之间的随机性
                tools=tools
            )
            # DeepSeek的响应结构与OpenAI存在差异，需要特殊处理
            # tool_call = response['choices'][0]['message']['tool_calls'][0]
            # resp = tool_call['function']['arguments'].strip()
            resp = response.choices[0].message
            print(response)
            if debug:
                print("模型回复：", resp)
            return resp
        except openai.error.APIError as e:
            if "Insufficient Balance" in str(e):
                # 发送邮件/短信通知
                print("DeepSeek账户余额不足，请及时充值")
            else:
                print(f"API请求失败: {e}")
        except Exception as e:
            print(f"发生未知错误: {e}")
            return ""

    def get_completion_messages(self, messages, model="deepseek-chat", tools=[], debug=False):
        try:
            if debug:
                print("prompt:", messages)
            response = openai.ChatCompletion.create(
                model=model,
                messages=messages,
                temperature=0.5,  # DeepSeek推荐0-1之间的随机性
                tools=tools
            )
            # DeepSeek的响应结构与OpenAI存在差异，需要特殊处理
            resp = response.choices[0].message
            # tool_call = response['choices'][0]['message']['tool_calls'][0]
            # resp = tool_call['function']['arguments'].strip()
            print(response)
            if debug:
                print("模型回复：", resp)
            return resp
        except openai.error.APIError as e:
            if "Insufficient Balance" in str(e):
                # 发送邮件/短信通知
                print("DeepSeek账户余额不足，请及时充值")
            else:
                print(f"API请求失败: {e}")
        except Exception as e:
            print(f"发生未知错误: {e}")
            return ""

    def moderation(self, text, debug=False):
        try:
            if debug:
                print("prompt:", text)
            response = openai.Moderation.create(
                input=text
            )
            # DeepSeek的响应结构与OpenAI存在差异，需要特殊处理
            resp = response.results[0]
            print("模型回复：", resp)
            return resp.flagged
        except openai.error.APIError as e:
            print(f"API请求失败: {e}")


# 使用示例
if __name__ == "__main__":
    deepseek = DeepSeekFunc()

    functions_sum = [
        {
            "type": "function",
            "function": {  # 用 JSON 描述函数。可以定义多个，但是只有一个会被调用，也可能都不会被调用
                "name": "sum",
                "description": "计算数组中所有数字的和",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "numbers": {
                            "type": "array",
                            "items": {
                                "type": "number"
                            }
                        }
                    }
                },
            }
        }
    ]

    prompt_sum = "桌上有 2 个苹果，四个桃子和 3 本书，一共有几个水果？"

    messages_sum = [
        {"role": "system", "content": "你是一个小学数学老师，你要教学生加法"},
        {"role": "user", "content": prompt_sum}
    ]

    result = deepseek.get_completion_messages(messages_sum, tools=functions_sum, debug=True)
    content = result.content.strip()
    if result.tool_calls:
        tool_call = result.tool_calls[0]
        id = tool_call.id
        function_name = tool_call.function.name
        arguments = tool_call.function.arguments
        print("id:", id)
        print("function_name:", function_name)
        print("arguments:", arguments)
        print("content:", content)
        print("result:", eval(arguments)["numbers"])
    else:
        print("content:", content)
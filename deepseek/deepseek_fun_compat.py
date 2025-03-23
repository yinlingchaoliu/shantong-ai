import json
from typing import Union

import openai
import os
from dotenv import load_dotenv, find_dotenv
from openai import AuthenticationError , OpenAI

from IChat import IChat

from importlib.metadata import version

openai_version = version("openai")
# 版本导入兼容
if openai_version >= "1.0.0":
    from openai import APIError
else:
    from openai.error import APIError


class DeepSeekFuncCompat(IChat):

    # 初始化DeepSeekChat类
    def __init__(self):
        self._model_name = "deepseek-chat"
        self._openai_api_base = "https://api.deepseek.com/v1"
        self._openai_api_beta = "https://api.deepseek.com/beta"
        self._openai_api_key = None
        self.init_api_key()
        # 初始化OpenAI客户端 根据openai版本
        self.client:OpenAI = self._get_openai_client()
        self.check_api_key()

    def init_api_key(self):
        # 加载环境变量
        load_dotenv(find_dotenv())
        # 配置DeepSeek API参数
        self._openai_api_key = os.getenv('DEEPSEEK_API_KEY')  # 需在.env中配置DEEPSEEK_API_KEY
        print("DeepSeek API Key:", self._openai_api_key)

    def check_api_key(self):
        try:
            if self.client:
                self.openai = OpenAI(api_key=self._openai_api_key, base_url="https://api.deepseek.com/v1")
                self.openai.models.list()  # 测试API连通性
            else:
                openai.Model.list()  # 测试API连通性
            print("API Key 有效")
        except AuthenticationError as e:
            print(f"认证失败: {e}")
        except Exception as e:
            self._handle_openai_error(e)

    def get_completion(self, prompt, model="deepseek-chat", tools=[], debug=False):
        try:
            if debug:
                print("prompt:", prompt)
            messages = [{"role": "user", "content": prompt}]
            if self.client:
                response = self.client.chat.completions.create(model=model, messages=messages, tools=tools)
            else:
                response = openai.ChatCompletion.create(model=model, messages=messages, tools=tools)
            # DeepSeek的响应结构与OpenAI存在差异，需要特殊处理
            # tool_call = response['choices'][0]['message']['tool_calls'][0]
            # resp = tool_call['function']['arguments'].strip()
            resp = response.choices[0].message
            if debug:
                print("模型回复：", resp)
            return resp
        except APIError as e:
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
                formatted_json = json.dumps(messages, indent=4, ensure_ascii=False)
                print("prompt:", formatted_json)
            if self.client:
                response = self.client.chat.completions.create(model=model, messages=messages, tools=tools)
            else:
                response = openai.ChatCompletion.create(model=model, messages=messages, tools=tools)

            # DeepSeek的响应结构与OpenAI存在差异，需要特殊处理
            resp = response.choices[0].message

            if debug:
                print("模型回复：", resp)
            return resp
        except APIError as e:
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
        except APIError as e:
            print(f"API请求失败: {e}")

    def _get_openai_client(self, api_key: str = None, base_url: str = None) -> Union[OpenAI, None]:
        openai_version = version("openai")
        # 新版 SDK (≥1.0.0)
        if tuple(map(int, openai_version.split('.')[:2])) >= (1, 0):
            from openai import OpenAI
            return OpenAI(
                api_key=api_key or self._openai_api_key,
                base_url=base_url or self._openai_api_beta,
                timeout=30.0
            )
        # 旧版 SDK (<1.0.0)
        else:
            openai.api_key = api_key or self._openai_api_key
            openai.api_base = base_url or self._openai_api_beta
            return None  # 旧版无需返回客户端实例

    """
    统一异常处理
    """
    def _handle_openai_error(self,error: Exception):
        """处理不同版本的API错误"""
        error_info = {
            "error_type": type(error).__name__,
            "message": str(error)
        }

        # 新版错误结构
        if hasattr(error, "status_code"):
            error_info["status_code"] = error.status_code
            error_info["code"] = getattr(error, "code", None)
        # 旧版错误结构
        elif hasattr(error, "http_status"):
            error_info["status_code"] = error.http_status
            error_info["code"] = error.code

        print(f"OpenAI API Error: {error_info}")


# 使用示例
if __name__ == "__main__":
    deepseek = DeepSeekFuncCompat()

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

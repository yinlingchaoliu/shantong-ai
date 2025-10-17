import json
from typing import Union

import openai
import os
from dotenv import load_dotenv, find_dotenv
from openai import AuthenticationError, OpenAI

from IChat import IChat
from nlu_deepseek import *

from importlib.metadata  import version
openai_version = version("openai")
#版本导入兼容
if openai_version >= "1.0.0":
    from openai import APIError
else:
    from openai.error import APIError

"""
deepseek兼容版本
"""
class DeepSeekChatCompat(IChat):
    # 初始化DeepSeekChat类
    def __init__(self):
        self._model_name = "deepseek-chat"
        self._openai_api_beta = "https://api.deepseek.com/beta"
        self._openai_api_key = None
        self.init_api_key()
        # 初始化OpenAI客户端 根据openai版本
        self.client = self._get_openai_client()
        self.check_api_key()

    """
    根据 OpenAI SDK 版本返回兼容的客户端对象
    :param api_key: API密钥
    :param base_url: 自定义API端点（用于国内代理或私有化部署）
    :return: 客户端对象或旧版全局配置
    """

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

    def get_completion(self, prompt, model="deepseek-chat", debug=False):
        try:
            if debug:
                print("prompt:", prompt)
            if self.client:
                messages = [{"role": "user", "content": prompt}]
                response = self.client.chat.completions.create(model=model, messages=messages)
            else:
                response = openai.ChatCompletion.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                )
            resp = response.choices[0].message.content.strip()
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
            self._handle_openai_error(e)
            return ""

    def get_completion_messages(self, messages, model="deepseek-chat", debug=False):
        try:
            if debug:
                formatted_json = json.dumps(messages, indent=4, ensure_ascii=False)
                print("prompt:", formatted_json)

            if self.client:
                response = self.client.chat.completions.create(model=model, messages=messages)
            else:
                response = openai.ChatCompletion.create(
                    model=model,
                    messages=messages,
                    temperature=0.5,  # DeepSeek推荐0-1之间的随机性
                    max_tokens=512,  # 控制生成文本的最大长度
                )

            resp = response.choices[0].message.content.strip()
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
    deepseek = DeepSeekChatCompat()
    prompt = json_prompt_v4()
    # result = deepseek.get_completion(prompt, debug=True)
    messages =  [{"role": "user", "content": prompt}]
    result_v1 = deepseek.get_completion_messages(messages, debug=True)

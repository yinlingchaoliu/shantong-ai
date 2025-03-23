import json

import openai
import os
from dotenv import load_dotenv, find_dotenv
from openai import AuthenticationError, OpenAI, APIError

from IChat import IChat
from nlu_deepseek import *

"""
openai 低版本的api > 0.1.0
"""
class DeepSeekChatV2(IChat):
    # 初始化DeepSeekChat类
    def __init__(self):
        self._model_name = "deepseek-chat"
        self._openai_api_key = None
        self._openai_api_base = "https://api.deepseek.com/beta"  # DeepSeek的API端点
        self.init_api_key()
        self.openai = OpenAI(api_key=self._openai_api_key, base_url=self._openai_api_base)
        self.check_api_key()

    def init_api_key(self):
        # 加载环境变量
        load_dotenv(find_dotenv())
        # 配置DeepSeek API参数
        self._openai_api_key = os.getenv('DEEPSEEK_API_KEY')  # 需在.env中配置DEEPSEEK_API_KEY
        print("DeepSeek API Key:", self._openai_api_key)

    def check_api_key(self):
        try:
            openai = OpenAI(api_key=self._openai_api_key, base_url="https://api.deepseek.com/v1")
            openai.models.list()  # 测试API连通性
            print("API Key 有效")
        except AuthenticationError as e:
            print(f"认证失败: {e}")
        except APIError as e:
            print(f"API错误: {e}")

    def get_completion(self, prompt, model="deepseek-chat",debug=False):
        messages = [{"role": "user", "content": prompt}]
        prompt_json = json.dumps(messages)
        try:
            if debug:
                print("prompt:", prompt)
            response = self.openai.completions.create(
                model=model,
                prompt=prompt_json,
                temperature=0.5,  # DeepSeek推荐0-1之间的随机性
                max_tokens=512,  # 控制生成文本的最大长度
            )
            # DeepSeek的响应结构与OpenAI存在差异，需要特殊处理
            print("模型回复：", response)
            resp = response.choices[0].text.strip()
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

    def get_completion_messages(self, messages, model="deepseek-chat",debug=False):
        try:
            prompt_json = json.dumps(messages)
            if debug:
                formatted_json = json.dumps(messages, indent=4, ensure_ascii=False)
                print("prompt:", formatted_json)
            response = self.openai.completions.create(
                model=model,
                prompt=prompt_json,
                temperature=0.5,  # DeepSeek推荐0-1之间的随机性
                max_tokens=512,  # 控制生成文本的最大长度
            )
            # DeepSeek的响应结构与OpenAI存在差异，需要特殊处理
            resp = response.choices[0].text.strip()
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

    def moderation(self, text,debug=False):
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

# 使用示例
if __name__ == "__main__":
    deepseek = DeepSeekChatV2()
    prompt = json_prompt_v4()
    # result = deepseek.get_completion(prompt,debug=True)
    messages =  [{"role": "user", "content": prompt}]
    result_v1 = deepseek.get_completion_messages(messages, debug=True)
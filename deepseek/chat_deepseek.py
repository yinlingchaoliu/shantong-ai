from abc import ABC

import openai
import os
from dotenv import load_dotenv, find_dotenv
from openai.error import AuthenticationError

from IChat import IChat
from nlu_deepseek import *

class DeepSeekChat(IChat):
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

    def get_completion(self, prompt, model="deepseek-chat"):
        try:
            response = openai.ChatCompletion.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.5,  # DeepSeek推荐0-1之间的随机性
                max_tokens=512,  # 控制生成文本的最大长度
            )
            # DeepSeek的响应结构与OpenAI存在差异，需要特殊处理
            return response.choices[0].message.content.strip()
        except openai.error.APIError as e:
            if "Insufficient Balance" in str(e):
                # 发送邮件/短信通知
                print("DeepSeek账户余额不足，请及时充值")
            else:
                print(f"API请求失败: {e}")
        except Exception as e:
            print(f"发生未知错误: {e}")
            return ""

    def get_completion_messages(self, messages, model="deepseek-chat"):
        try:
            response = openai.ChatCompletion.create(
                model=model,
                messages=messages,
                temperature=0.5,  # DeepSeek推荐0-1之间的随机性
                max_tokens=512,  # 控制生成文本的最大长度
            )
            # DeepSeek的响应结构与OpenAI存在差异，需要特殊处理
            return response.choices[0].message.content.strip()
        except openai.error.APIError as e:
            if "Insufficient Balance" in str(e):
                # 发送邮件/短信通知
                print("DeepSeek账户余额不足，请及时充值")
            else:
                print(f"API请求失败: {e}")
        except Exception as e:
            print(f"发生未知错误: {e}")
            return ""
# 使用示例
if __name__ == "__main__":
    deepseek = DeepSeekChat()
    prompt = json_prompt_v4()
    print(prompt)
    result = deepseek.get_completion(prompt)
    print("模型回复：", result)






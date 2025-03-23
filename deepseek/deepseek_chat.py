from abc import ABC

from IChat import IChat
from deepseek_chat_compat import DeepSeekChatCompat
from nlu_deepseek import *

class DeepSeekChat(IChat):
    # 初始化DeepSeekChat类
    def __init__(self):
        self._deepseek = DeepSeekChatCompat()

    def init_api_key(self):
        self._deepseek.init_api_key()

    def check_api_key(self):
        self._deepseek.check_api_key()

    def get_completion(self, prompt, model="deepseek-chat",debug=False):
        self._deepseek.get_completion(prompt, model=model,debug=debug)

    def get_completion_messages(self, messages, model="deepseek-chat",debug=False):
        self._deepseek.get_completion_messages(messages, model=model,debug=debug)

    def moderation(self, text,debug=False):
        self._deepseek.moderation(text, debug=debug)

# 使用示例
if __name__ == "__main__":
    deepseek = DeepSeekChat()
    prompt = json_prompt_v4()
    # deepseek.get_completion(prompt,debug=True)
    messages =  [{"role": "user", "content": prompt}]
    deepseek.get_completion_messages(messages, debug=True)

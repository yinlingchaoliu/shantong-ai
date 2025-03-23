# pip3 install langchain_openai
# python3 deepseek_langchain.py
from langchain_core.language_models import BaseChatModel
import openai
import os
from dotenv import load_dotenv, find_dotenv
from langchain_openai import ChatOpenAI
from openai import AuthenticationError, OpenAI


# pip install langchain langchain-openai openai>=1.0
# pip install -r requirements.txt
# pip show openai
# pip install langchain-openai

class DeepSeekLangChain():
    # 初始化DeepSeekChat类
    def __init__(self):
        self.openai:OpenAI = None
        self._model_name = "deepseek-chat"
        self._openai_api_base = None
        self._openai_api_key = None
        self._llm_instance = None  # 使用不同名称避免冲突
        self.init_api_key()
        self.check_api_key()

    def init_api_key(self):
        # 加载环境变量
        load_dotenv(find_dotenv())
        # 配置DeepSeek API参数
        self._openai_api_key = os.getenv('DEEPSEEK_API_KEY') # 需在.env中配置DEEPSEEK_API_KEY
        self._openai_api_base = "https://api.deepseek.com/v1"
        openai.api_key = self._openai_api_key  # 需在.env中配置DEEPSEEK_API_KEY
        print("DeepSeek API Key:", openai.api_key)
        openai.api_base = self._openai_api_base  # DeepSeek的API端点
        #兼容方案 升级新版openai sdk
        self.openai = OpenAI(api_key=self._openai_api_key, base_url=self._openai_api_base)

    def check_api_key(self):
        try:
            self.openai.models.list()
#            openai.Model.list()  # 测试API连通性
            print("API Key 有效")
        except AuthenticationError as e:
            print(f"认证失败: {e}")
        except openai.APIError as e:
            print(f"API错误: {e}")

    def llm(self)->BaseChatModel:
        if not self._llm_instance:
            self._llm_instance = ChatOpenAI(
                model=self._model_name,
                api_key=self._openai_api_key,
                base_url=self._openai_api_base,
                max_tokens=1024
            )
        return self._llm_instance

if __name__ == '__main__':
    langchain = DeepSeekLangChain()
    response = langchain.llm().invoke("你好欢迎")
    print(response)
# pip install langchain_openai
# python deepseek_langchain.py
import langchain_openai
import openai
from langchain_core.language_models import BaseChatModel
import os
from dotenv import load_dotenv, find_dotenv
from langchain_openai import ChatOpenAI,OpenAI

from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)

from langchain.prompts import PromptTemplate
# from langchain_community.llms import OpenAI
from langchain.prompts import PromptTemplate, ChatPromptTemplate, HumanMessagePromptTemplate

from langchain.output_parsers import PydanticOutputParser
from openai import AuthenticationError, APIError
from pydantic import BaseModel, Field, validator
from typing import List, Dict
import json

# pip install langchain langchain-openai openai>=1.0
# pip install -r requirements.txt
# pip show openai
# pip install langchain-openai

"""
langchain 是对openai高级封装
对原有sdk能力增强
"""
class DeepSeekLangChain():
    # 初始化DeepSeekChat类
    def __init__(self):
        self._model_name = "deepseek-chat"
        self._openai_api_base = "https://api.deepseek.com/v1"
        self._openai_api_key = None
        self.init_api_key()
        self._openai_instance :OpenAI= None
        self._llm_instance = None
        # 兼容方案 升级新版openai sdk
        self.check_api_key()

    def init_api_key(self):
        # 加载环境变量
        load_dotenv(find_dotenv())
        # 配置DeepSeek API参数
        self._openai_api_key = os.getenv('DEEPSEEK_API_KEY')  # 需在.env中配置DEEPSEEK_API_KEY
        print("DeepSeek API Key:", self._openai_api_key)

    def check_api_key(self):
        try:
            _openai = openai.OpenAI(api_key=self._openai_api_key, base_url=self._openai_api_base)
            _openai.models.list()  # 测试API连通性
            print("API Key 有效")
        except AuthenticationError as e:
            print(f"认证失败: {e}")
        except APIError as e:
            print(f"API错误: {e}")

    def llm(self,temperature=None)->BaseChatModel:
        if not self._llm_instance:
            self._llm_instance = ChatOpenAI(
                model=self._model_name,
                api_key=self._openai_api_key,
                base_url=self._openai_api_base,
                max_tokens=1024,
                temperature = temperature
            )
        return self._llm_instance

    def client(self)->langchain_openai.OpenAI:
        if not self._openai_instance:
            self._openai_instance = langchain_openai.OpenAI(
                api_key=self._openai_api_key,
                base_url=self._openai_api_base,
                max_tokens=1024,
                temperature=0.7
            )
        return self._openai_instance

if __name__ == '__main__':
    langchain = DeepSeekLangChain()
    chat_model = langchain.llm()
    client = langchain.client()
    response= chat_model.invoke("你好，欢迎")
    print(response.content)

    messages = [
        SystemMessage(content="你是AGIClass的课程助理。"),
        HumanMessage(content="我来上课了")
    ]

    response = chat_model.invoke(messages)
    print(response.content)

    template = PromptTemplate.from_template("给我讲个关于{subject}的笑话,说给{other}听")
    print(template.input_variables)
    print(template.format(subject='小明', other='小红'))

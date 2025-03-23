#!pip install langchain

from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI

llm = OpenAI()  # 默认是text-davinci-003模型
llm.predict("你好，欢迎")

from langchain.chat_models import ChatOpenAI

# 只需修改以下参数即可切换至 DeepSeek
llm = ChatOpenAI(
    openai_api_base="https://api.deepseek.com/v1",  # DeepSeek API 地址
    model_name="deepseek-chat",                    # DeepSeek 模型名称
    openai_api_key="YOUR_DEEPSEEK_API_KEY"         # DeepSeek API 密钥
)

# 原代码无需修改，直接复用
response = llm.predict("你好，欢迎")
print(response)
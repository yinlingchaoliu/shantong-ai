from openai.error import AuthenticationError
import openai
import os
from dotenv import load_dotenv, find_dotenv

# 加载环境变量
_ = load_dotenv(find_dotenv())

# 配置DeepSeek API参数
openai.api_key = os.getenv('DEEPSEEK_API_KEY')  # 需在.env中配置DEEPSEEK_API_KEY
print("DeepSeek API Key:", openai.api_key)
openai.api_base = "https://api.deepseek.com/v1"  # DeepSeek的API端点

def check_api_key():
    try:
        openai.Model.list()  # 测试API连通性
        print("API Key 有效")
    except AuthenticationError as e:
        print(f"认证失败: {e}")
    except openai.error.APIError as e:
        print(f"API错误: {e}")

check_api_key()
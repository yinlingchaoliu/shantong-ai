#!pip install ollama

import ollama
from ollama import Client

def list_ollama_models():
    # 关键修改点1：显式指定客户端配置
    client = Client(host="http://localhost:11434")  # 确保端口匹配

    try:
        # 关键修改点2：使用正确的方法名
        models = client.list()["models"]  # 新版API使用.list()

        if not models:
            print("⚠️ 没有找到可用模型，请先下载模型")
            print("执行命令：ollama pull llama2")
            return

        print(f"找到 {len(models)} 个模型：")
        for model in models:
            # 关键修改点3：访问正确的字段
            print(f"▪ 名称: {model['name']}")
            print(f"  模型ID: {model['model']}")
            print(f"  最后修改: {model['modified_at']}")
            print("-" * 40)

    except Exception as e:
        print(f"❌ 连接失败: {str(e)}")
        print("请检查：1. Ollama服务是否启动 2. 防火墙设置 3. 端口是否暴露")

if __name__ == "__main__":
    list_ollama_models()
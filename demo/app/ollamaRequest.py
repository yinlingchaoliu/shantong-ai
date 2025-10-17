import openai
import os
import time
import requests
import time

# 配置 Ollama 服务的地址
OLLAMA_BASE_URL = "http://localhost:11434"

def generate_text(prompt, model="deepseek-r1:1.5b"):
    """
    调用 Ollama 服务生成文本
    :param prompt: 输入的提示信息
    :param model: 使用的模型，默认为 llama2
    :return: 生成的文本结果
    """
    url = f"{OLLAMA_BASE_URL}/api/generate"
    data = {
        "model": model,
        "prompt": prompt
    }
    try:
        response = requests.post(url, json=data, stream=True)
        response.raise_for_status()
        result = ""
        for line in response.iter_lines():
            if line:
                # 解析 JSON 数据
                import json
                line_data = json.loads(line)
                if 'response' in line_data:
                    result += line_data['response']
        return result
    except requests.RequestException as e:
        print(f"请求出错: {e}")
        return None


if __name__ == "__main__":
    prompt = "今天我很"
    start_time = time.time()
    generated_text = generate_text(prompt)
    end_time = time.time()

    if generated_text:
        print("生成的文本:")
        print(generated_text)
        print(f"耗时: {end_time - start_time:.2f} 秒")

# 概率生成
# 学习 权重 生成

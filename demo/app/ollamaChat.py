import ollama
from ollama import Client

# 配置 Ollama 客户端
client = Client(host="http://localhost:11434")  # 默认地址

# 消息格式（与 OpenAI 兼容）
messages = [
    {
        "role": "system",
        "content": "你是AI助手小瓜.你是AGIClass的助教。这门课每周二、四上课。"
    },
    {
        "role": "user",
        "content": "你是干什么的?什么时间上课"
    }
]

try:
    # 调用本地模型（需提前下载）
    response = client.chat(
        model="deepseek-r1:1.5b",  # 或其他已下载模型
        messages=messages,
        options={
            "temperature": 0.7,
            "num_predict": 512  # 最大生成token数
        }
    )

    # 输出回复
    print("小瓜回复：", response['message']['content'])

    print(response)

except Exception as e:
    print(f"请求失败：{str(e)}")
    print("请检查：1. Ollama服务是否运行 2. 模型是否下载（执行 ollama pull llama2）")
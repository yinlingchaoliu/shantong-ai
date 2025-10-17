from prometheus_client import Counter, start_http_server
import time
import sqlite3
from datetime import datetime
import openai
import os
from dotenv import load_dotenv, find_dotenv

# ================= Prometheus监控 =================
# API调用成本计数器（单位：元）
api_cost_counter = Counter(
    'deepseek_api_cost_rmb',
    'DeepSeek API调用总成本',
    ['model', 'status']
)


# ================= 持久化存储 =================
def init_db():
    """初始化SQLite数据库"""
    conn = sqlite3.connect('api_cost.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS api_calls
                 (timestamp DATETIME, model TEXT, prompt_tokens INT, 
                 completion_tokens INT, total_tokens INT, cost REAL)''')
    conn.commit()
    conn.close()


def record_cost_to_db(model, usage, cost):
    """记录到数据库"""
    conn = sqlite3.connect('api_cost.db')
    c = conn.cursor()
    c.execute('''INSERT INTO api_calls VALUES 
                 (?, ?, ?, ?, ?, ?)''',
              (datetime.now(), model,
               usage.get('prompt_tokens', 0),
               usage.get('completion_tokens', 0),
               usage.get('total_tokens', 0),
               cost))
    conn.commit()
    conn.close()


# ================= 核心调用封装 =================
DEEPSEEK_PRICE_RATE = {
    # 假设定价（实际需根据官方价格调整）
    'deepseek-chat': 0.01,  # 元/千tokens
    'deepseek-coder': 0.02,
}

def init_api_key():
    # 加载环境变量
    load_dotenv(find_dotenv())
    # 配置DeepSeek API参数
    openai.api_key = os.getenv('DEEPSEEK_API_KEY')  # 需在.env中配置DEEPSEEK_API_KEY
    print("DeepSeek API Key:", openai.api_key)
    openai.api_base = "https://api.deepseek.com/v1"  # DeepSeek的API端点

def calculate_cost(usage, model):
    """计算调用成本"""
    rate = DEEPSEEK_PRICE_RATE.get(model, 0.01)
    total_tokens = usage.get('total_tokens', 0)
    return (total_tokens / 1000) * rate

def get_deepseek_completion(prompt, model="deepseek-chat"):
    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
            max_tokens=512
        )

        # 解析使用量
        usage = response.usage.to_dict()  # 假设响应包含usage字段
        cost = calculate_cost(usage, model)

        # Prometheus记录
        api_cost_counter.labels(model=model, status='success').inc(cost)

        # 持久化存储
        record_cost_to_db(model, usage, cost)

        return response.choices[0].message.content.strip()

    except openai.error.APIError as e:
        api_cost_counter.labels(model=model, status='failed').inc(0)
        raise


# ================= 使用示例 =================
if __name__ == "__main__":
    init_api_key()
    # 初始化监控系统（暴露9100端口）
    start_http_server(9100)
    init_db()

    # 示例调用
    while True:
        try:
            result = get_deepseek_completion("你好", model="deepseek-chat")
            print(f"API调用成功，结果长度：{len(result)}")
        except Exception as e:
            print(f"调用失败：{str(e)}")

        time.sleep(60)  # 每分钟调用一次

# -- 按小时统计成本
# SELECT strftime('%Y-%m-%d %H:00', timestamp) AS hour,
#        SUM(cost) AS total_cost
# FROM api_calls
# GROUP BY hour;

# curl http://localhost:9100/metrics

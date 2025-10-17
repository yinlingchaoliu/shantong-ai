import openai
import os
import time

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())

# 未找到api_key 未调通 练手使用ollama 平行替换
openai.api_key  = os.getenv('OPENAI_API_KEY')

prompt='今天我很'
response = openai.Completion.create(
  model="text-davinci-003",
  prompt=prompt,
  max_tokens=40,
  temperature=0,
  stream=True
)

#print(response.choices[0].text)

for chunk in response:
    print(chunk.choices[0].text, end='')
    time.sleep(0.2)

# 模型列表
models = openai.Model.list()

for model in models.data:
    print(model.id)

messages = [
    {
        "role": "system",
        "content": "你是AI助手小瓜.你是AGIClass的助教。这门课每周二、四上课。"
    },
    {
        "role": "user",
        "content": "你是干什么的?什么时间上课"
    },

]

# 调用ChatGPT-3.5
chat_completion = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=messages)

# 输出回复
print(chat_completion.choices[0].message.content)
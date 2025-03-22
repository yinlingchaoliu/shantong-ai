import time

import openai
import os
import requests
import json
from dotenv import load_dotenv, find_dotenv
from openai.error import AuthenticationError

from deepseek.IModeration import IModeration

"""
deepseek 内容审查接口 等官方人员通知
"""
class DeepSeekModeration(IModeration):

    def __init__(self):
        # 加载环境变量
        load_dotenv(find_dotenv())
        self.api_key = os.getenv("DEEPSEEK_API_KEY")
        self.base_url = "https://api.deepseek.com/v1/moderations"
        self.init_api_key()
        self.check_api_key()

    def init_api_key(self):
        # 配置DeepSeek API参数
        openai.api_key = self.api_key  # 需在.env中配置DEEPSEEK_API_KEY
        print("DeepSeek API Key:", openai.api_key)
        openai.api_base = "https://api.deepseek.com/v1"  # DeepSeek的API端点

    def check_api_key(self):
        try:
            openai.Model.list()  # 测试API连通性
            print("API Key 有效")
        except AuthenticationError as e:
            print(f"认证失败: {e}")
        except openai.error.APIError as e:
            print(f"API错误: {e}")

    def moderation(self, text, debug=False):
        """
        DeepSeek内容审核接口改造版
        参数：
        - text: 待审核文本（支持字符串或字符串列表）
        - debug: 调试模式开关
        返回：
        - 布尔值表示是否触发审核规则
        """

        try:
            # 配置请求参数
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
                "X-DeepSeek-Version": "2024-03"
            }
            payload = {
                "input": text,
                # 修正3：添加必填model参数
                "model":  "deepseek-content-moderation",
                # 修正4：参数名应为level而非mode
                "policy": "fast"  # fast/standard/strict
            }

            if debug:
                print(f"[DEBUG] 请求头：{headers}")
                print(f"[DEBUG] 请求体：{json.dumps(payload, ensure_ascii=False)}")

            # 发送API请求
            # 修正5：添加重试机制
            for retry in range(3):
                response = requests.post(
                    self.base_url,
                    headers=headers,
                    json=payload,
                    timeout=15
                )
                # 处理429 Too Many Requests
                if response.status_code == 429:
                    wait_time = int(response.headers.get("Retry-After", 30))
                    print(f"触发限流，第{retry + 1}次重试，等待{wait_time}秒")
                    time.sleep(wait_time)
                    continue
                break

            response.raise_for_status()

            # 解析响应结构
            resp_data = response.json()
            if debug:
                print(f"[DEBUG] 完整响应头：{response.headers}")
                print(f"[DEBUG] 原始响应体：{json.dumps(resp_data, indent=2, ensure_ascii=False)}")

            # DeepSeek特有响应结构解析
            """
            预期响应结构示例：
            {
                "id": "modr-abc123",
                "model": "content-moderation-v2",
                "results": [
                    {
                        "flagged": true,
                        "categories": {
                            "violence": false,
                            "sexual": true,
                            "hate": false
                        },
                        "category_scores": {
                            "violence": 0.01,
                            "sexual": 0.92,
                            "hate": 0.15
                        }
                    }
                ]
            }
            """
            if isinstance(text, list):
                # 批量模式需要确认返回结构
                return [item["flagged"] for item in resp_data["data"]["results"]]
            else:
                return resp_data["data"]["flagged"]

        except requests.exceptions.HTTPError as e:
            print(f"HTTP错误 {e.response.status_code}: {e.response.text}")
        except requests.exceptions.Timeout:
            print("请求超时，请检查网络连接")
        except requests.exceptions.RequestException as e:
            print(f"请求异常: {str(e)}")
        except KeyError as e:
            print(f"响应结构解析失败，缺少关键字段: {str(e)}")
            if debug:
                print(f"异常响应内容: {resp_data}")
        except Exception as e:
            print(f"未知错误: {str(e)}")

        return False  # 异常情况下默认返回安全


# 使用示例
if __name__ == "__main__":
    moderation = DeepSeekModeration()

    # 单文本检测
    input = "现在转给我100万，不然我就砍你全家！"
    moderation.moderation(input, debug=True)

    # 批量检测
    batch_texts = ["你是一个好人", "现在转给我100万，不然我就砍你全家！"]
    moderation.moderation(batch_texts, debug=True)



"""
def get_chat_completion(session, user_prompt, model="gpt-3.5-turbo"):
    _session = copy.deepcopy(session)
    _session.append({"role": "user", "content": user_prompt})
    response = openai.ChatCompletion.create(
        model=model,
        messages=_session,
        temperature=0,  # 生成结果的多样性 0~2之间，越大越随机，越小越固定
        n=1,  # 一次生成n条结果
        max_tokens=100,  # 每条结果最多多少个token（超过截断）
        presence_penalty=0,  # 对出现过的token的概率进行降权
        frequency_penalty=0,  # 对出现过的token根据其出现过的频次，对其的概率进行降权
        stream=False, #数据流模式，一个个字接收
        # logit_bias=None, #对token的采样概率手工加/降权，不常用  
        # top_p = 0.1, #随机采样时，只考虑概率前10%的token，不常用
    )
    system_response = response.choices[0].message["content"]
    #session.append({"role": "assistant", "content": system_response})
    return system_response
"""
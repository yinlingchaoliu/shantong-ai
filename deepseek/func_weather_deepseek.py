"""
function calling 演示调用
1. NLU 分析语义
2. 解析deepseek返回 调用本地API
3. 结果上传deepseek, 友好语言返回
"""
import json

from deepseek_chat import DeepSeekChat
from deepseek_fun import DeepSeekFunc

"""
 模拟接口调用
"""

"""
messages 原有提示词
tool_calls 调用工具
new_data 接口获得数据
"""

class WeatherFunc():

    def __init__(self):
        self.deepfunc = DeepSeekFunc()
        self.deepchat = DeepSeekChat()
        self.messages = []

    def chat(self,input,debug=True):
        self.messages.append({"role": "user", "content": input})
        result = self._deepseek_func(debug=debug)
        content = result.content.strip()
        if result.tool_calls:
            tool_call = result.tool_calls[0]
            function_name = tool_call.function.name
            arguments = tool_call.function.arguments
            print("function_name:", function_name)
            print("arguments:", arguments.strip())
            print("content:", content)
            print("city:", eval(arguments)["city"].strip())
            print("unit:", eval(arguments)["unit"])

            # 调用API 查询资料
            city = eval(arguments)["city"].strip()
            unit = eval(arguments)["unit"].strip()
            weather_data = self._get_current_weather(city,unit)
            self._deepseek_func_warp(tool_call,weather_data)
        else:
            print("content:", content)
        pass

    """
        第一次使用function calling 进行语义分析
    """
    def _deepseek_func(self,debug=True):
        functions_weather = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "获取指定城市的实时天气信息",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "city": {
                                "type": "string",
                                "description": "城市名称，例如：北京、上海"
                            },
                            "unit": {
                                "type": "string",
                                "enum": ["celsius", "fahrenheit"],
                                "description": "温度单位，默认为摄氏度（celsius）"
                            }
                        },
                        "required": ["city"]
                    }
                }

            }
        ]
        return self.deepfunc.get_completion_messages(self.messages, tools=functions_weather, debug=debug)

    """
        请求第三方api, 获得真实数据
        实际应调用如 OpenWeatherMap 的 API（此处为模拟数据）
    """
    def _get_current_weather(self, city, unit="celsius"):
        return {
            "city": city,
            "temperature": 28.5 if unit == "celsius" else 83.3,
            "unit": unit,
            "condition": "晴",
            "humidity": 65
        }

    """
    组装数据 让deepseek 重新回答
    """
    def _deepseek_func_warp(self, tool_call, new_data, debug=True):
        function_args = tool_call['function']['arguments']
        self.messages.append(
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": tool_call['id'],
                    "type": "function",
                    "function": {
                        "name": "get_current_weather",
                        "arguments": function_args
                    }
                }]
            }
        )

        self.messages.append(
            {
                "role": "tool",
                "content": json.dumps(new_data),
                "tool_call_id": tool_call['id']
            }
        )

        return self.deepchat.get_completion_messages(self.messages, debug=debug)


if __name__ == '__main__':

    deepfunc = WeatherFunc()
    deepfunc.chat("今天北京天气怎么样？温度用摄氏度显示。",debug=True)

    #
    # messages = []
    # prompt_weather = "今天北京天气怎么样？温度用摄氏度显示。"
    # messages.append({"role": "user", "content": prompt_weather})
    # # 联网无法恢复
    # functions_weather = [
    #     {
    #         "type": "function",
    #         "function": {
    #             "name": "get_weather",
    #             "description": "获取指定城市的实时天气信息",
    #             "parameters": {
    #                 "type": "object",
    #                 "properties": {
    #                     "city": {
    #                         "type": "string",
    #                         "description": "城市名称，例如：北京、上海"
    #                     },
    #                     "unit": {
    #                         "type": "string",
    #                         "enum": ["celsius", "fahrenheit"],
    #                         "description": "温度单位，默认为摄氏度（celsius）"
    #                     }
    #                 },
    #                 "required": ["city"]
    #             }
    #         }
    #
    #     }
    # ]
    #
    # result = deepfunc.get_completion_messages(messages, tools=functions_weather, debug=True)
    # content = result.content.strip()
    # if result.tool_calls:
    #     tool_call = result.tool_calls[0]
    #     function_name = tool_call.function.name
    #     arguments = tool_call.function.arguments
    #     print("function_name:", function_name)
    #     print("arguments:", arguments.strip())
    #     print("content:", content)
    #     print("result:", eval(arguments)["city"].strip())
    #     print("unit:", eval(arguments)["unit"])
    #
    #     # 调用API 查询资料
    #     args = json.loads(arguments)
    #
    # else:
    #     print("content:", content)
    #
    # # 结果给deepseek

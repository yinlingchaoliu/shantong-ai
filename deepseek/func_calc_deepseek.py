import json

from deepseek_chat import DeepSeekChat
from deepseek.Switch import Switch
from deepseek_fun import DeepSeekFunc

"""
计算器
"""


class calcDeepSeek():
    def __init__(self):
        self.deepfunc = DeepSeekFunc()
        self.deepchat = DeepSeekChat()
        self.messages = []
        # function_calling 函数
        self.functions = [
            {
                "type": "function",
                "function": {  # 用 JSON 描述函数。可以定义多个，但是只有一个会被调用，也可能都不会被调用
                    "name": "sum",
                    "description": "计算数组中所有数字的和",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "numbers": {
                                "type": "array",
                                "items": {
                                    "type": "number",
                                    "description": "必须是数值类型"
                                }
                            }
                        }
                    },
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "subtract",
                    "description": "计算 a - b 的值",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "a": {
                                "type": "number",
                                "description": "被减数，必须是数值类型"
                            },
                            "b": {
                                "type": "number",
                                "description": "减数，必须是数值类型"
                            }
                        }
                    },
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "multiply",
                    "description": "计算数组中所有数字的积",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "numbers": {
                                "type": "array",
                                "items": {
                                    "type": "number",
                                    "description": "必须是数值类型"
                                }
                            }
                        }
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "divide",
                    "description": "计算 a/b 的值",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "a": {
                                "type": "number",
                                "description": "被除数，必须是数值类型"
                            },
                            "b": {
                                "type": "number",
                                "description": "除数，必须是数值类型"
                            }
                        }
                    },
                }
            }

        ]

    def calc(self, prompt,debug=True):
        self.messages = [
            {"role": "system", "content": "你是一个小学数学老师，你要教学生四则混合运算"},
            {"role": "user", "content": prompt}
        ]

        result = self.deepfunc.get_completion_messages(self.messages, tools=self.functions, debug=debug)
        tool_calls = result.tool_calls
        if tool_calls:
            for tool_call in tool_calls:
                function_name = tool_call.function.name
                arguments = tool_call.function.arguments
                id = tool_call.id
                print("id:", id)
                print("function_name:", function_name)
                print("arguments:", arguments)
                args = json.loads(arguments)
                # 后续采用if改写
                result = (Switch(function_name)
                .case("sum", lambda: sum(args["numbers"]))
                .case("subtract", lambda: args["a"] - args["b"])
                .case("multiply", lambda: args["numbers"][0] * args["numbers"][1])
                .case("divide", lambda: args["a"] / args["b"])
                .default({"Unknown function"}))
                print(result)
                self._deepseek_func_warp(tool_call, result, debug=True)
                # self.messages.append({"role": "function", "name": function_name, "content": str(result)})
            self.deepchat.get_completion_messages(self.messages, debug=debug)

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


if __name__ == '__main__':
    calc = calcDeepSeek()
    # calc.calc("请计算: 1+2*3-4/5")
    calc.calc("请计算: 6 * 3 / (4+2) = ?")
    pass

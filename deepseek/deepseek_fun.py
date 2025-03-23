
from IChat import IChat
from deepseek.deepseek_fun_compat import DeepSeekFuncCompat

class DeepSeekFunc(IChat):
    # 初始化DeepSeekChat类
    def __init__(self):
        self._deepseek= DeepSeekFuncCompat()

    def init_api_key(self):
        self._deepseek.init_api_key()

    def check_api_key(self):
        self._deepseek.check_api_key()

    def get_completion(self, prompt, model="deepseek-chat", tools=[], debug=False):
        return self._deepseek.get_completion(prompt, model, tools, debug)

    def get_completion_messages(self, messages, model="deepseek-chat", tools=[], debug=False):
        return self._deepseek.get_completion_messages(messages, model, tools, debug)

    def moderation(self, text, debug=False):
        self._deepseek.moderation(text, debug)

# 使用示例
if __name__ == "__main__":
    deepseek = DeepSeekFunc()

    functions_sum = [
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
                                "type": "number"
                            }
                        }
                    }
                },
            }
        }
    ]

    prompt_sum = "桌上有 2 个苹果，四个桃子和 3 本书，一共有几个水果？"

    messages_sum = [
        {"role": "system", "content": "你是一个小学数学老师，你要教学生加法"},
        {"role": "user", "content": prompt_sum}
    ]

    result = deepseek.get_completion_messages(messages_sum, tools=functions_sum, debug=True)
    content = result.content.strip()
    if result.tool_calls:
        tool_call = result.tool_calls[0]
        id = tool_call.id
        function_name = tool_call.function.name
        arguments = tool_call.function.arguments
        print("id:", id)
        print("function_name:", function_name)
        print("arguments:", arguments)
        print("content:", content)
        print("result:", eval(arguments)["numbers"])
    else:
        print("content:", content)
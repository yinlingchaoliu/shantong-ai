import json

from chat_deepseek import DeepSeekChat
from sqlite_company import init_database
from func_deepseek import DeepSeekFunc

"""
用户界面
  │
  ↓ 自然语言查询
DeepSeek API
  │
  ↓ 函数调用请求
业务服务器
  │
  ↓ SQL查询
SQLite 数据库
  │
  ↑ 查询结果
DeepSeek API
  │
  ↓ 自然语言回复
用户界面
"""


class AddContactFunc():

    def __init__(self):
        self.deepfunc = DeepSeekFunc()
        self.deepchat = DeepSeekChat()
        self.messages = []
        self.function_config = {
            "name": "add_contact",
            "description": "添加联系人",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "联系人姓名"
                    },
                    "address": {
                        "type": "string",
                        "description": "联系人地址"
                    },
                    "tel": {
                        "type": "string",
                        "description": "联系人电话"
                    },
                }
            }
        }
        # tools工具
        self.tools = [
            {
                "type": "function",
                "function": self.function_config
            }
        ]

    def chat(self, input, debug=True):
        self.messages.append({"role": "user", "content": input})
        self.messages.append({"role": "system",
                              "content": "你是一个联系人录入员。遇到以“啥啥”、“某某”、“什么”等模糊称谓指代的人名，只记录姓氏"})
        result = self._deepseek_func(debug=debug)
        content = result.content.strip()
        if result.tool_calls:
            tool_call = result.tool_calls[0]
            function_name = tool_call.function.name
            arguments = tool_call.function.arguments
            id = tool_call.id
            print("id:", id)
            print("function_name:", function_name)
            print("arguments:", arguments.strip())
            print("content:", content)
            # args = json.loads(arguments)
            # api_data = self._get_api(**args)
            # self._deepseek_func_warp(tool_call, api_data)
        else:
            print("content:", content)
        pass

    """
        第一次使用function calling 进行语义分析
    """

    def _deepseek_func(self, debug=True):
        return self.deepfunc.get_completion_messages(self.messages, tools=self.tools, debug=debug)

    """
        请求第三方api, 获得真实数据
        此处查询数据库
    """

    def _get_api(self, **arguments):
        return {}

    def _validate_params(self, params: dict):
        allowed_departments = ['技术部', '市场部', '财务部']
        if 'department' in params and params['department'] not in allowed_departments:
            raise ValueError("无效的部门名称")
        if 'min_salary' in params and params['min_salary'] < 0:
            raise ValueError("工资不能为负数")

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
                        "name": "query_employees",
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
    # 初始化数据库
    init_database()
    deepfunc = AddContactFunc()
    # 查询数据库
    deepfunc.chat("帮我寄给上周认识那个王啥啥来着那个女的，地址是北京市朝阳区亮马桥外交办公大楼，电话13012345678",
                  debug=True)

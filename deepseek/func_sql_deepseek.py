import json
import sqlite3

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
class SqlFunc():

    def __init__(self):
        self.deepfunc = DeepSeekFunc()
        self.deepchat = DeepSeekChat()
        self.messages = []
        self.function_config = {
            "name": "query_employees",
            "description": "根据条件查询员工信息",
            "parameters": {
                "type": "object",
                "properties": {
                    "department": {
                        "type": "string",
                        "description": "部门名称，例如：技术部、市场部"
                    },
                    "min_salary": {
                        "type": "number",
                        "description": "最低工资阈值"
                    },
                    "is_active": {
                        "type": "boolean",
                        "description": "是否在职员工"
                    }
                },
                # 至少需要指定一个查询条件
                "anyOf": [
                    {"required": ["department"]},
                    {"required": ["min_salary"]},
                    {"required": ["is_active"]}
                ]
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
            args = json.loads(arguments)
            api_data = self._get_api(**args)
            self._deepseek_func_warp(tool_call, api_data)
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
        department: str = arguments["department"].strip()
        min_salary: float = arguments["min_salary"]
        is_active: bool = arguments["is_active"]

        #数据校验nlu
        self._validate_params(arguments)

        """执行数据库查询并返回结果"""
        conn = sqlite3.connect('company.db')
        cursor = conn.cursor()

        # 构建查询条件
        conditions = []
        params = []

        if department:
            conditions.append("department = ?")
            params.append(department)
        if min_salary:
            conditions.append("salary >= ?")
            params.append(min_salary)
        if is_active is not None:
            conditions.append("is_active = ?")
            params.append(1 if is_active else 0)

        # 组合SQL语句
        query = "SELECT name, department, salary FROM employees"
        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        # 执行安全查询（防止SQL注入）
        cursor.execute(query, params)
        results = cursor.fetchall()

        # 格式化结果
        output = []
        for name, dept, salary in results:
            output.append(f"{name} ({dept}): ¥{salary:.2f}")

        conn.close()
        return json.dumps({"count": len(results), "employees": output}, ensure_ascii=False)

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
    deepfunc = SqlFunc()
    #查询数据库
    deepfunc.chat("列出技术部工资超过8000的在职员工", debug=True)

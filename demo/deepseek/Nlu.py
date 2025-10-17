
import json, copy

from deepseek_chat import DeepSeekChat
from nlu_deepseek import *

"""
NLU->DSL->DST->ASK
NLU: 自然语言理解 (语音结构化)
DST: 对话状态追踪 (整理回复)

典型工作流程
用户输入 → NLU解析 → DST更新状态 → 策略模块决策 → 生成回复
"""

# Nlu 通过deepseek 生成指定json语义
"""
角色
任务描述
输出格式
例子
用户输入
"""
class NLU:
    def __init__(self):
        self.prompt_template = f"{instruction_v1}\n\n{output_format_v2}\n\n{examples}\n\n用户输入：\n__INPUT__"
        self.chat = DeepSeekChat()

    # 调用deepseek的接口
    def _get_completion(self, prompt, model="deepseek-chat"):
        print(f"NLU prompt: {prompt}")
        response = self.chat.get_completion(prompt, model)
        print(f"NLU response: {response}")
        return response

    # 调用NLU 接口
    def parse(self, user_input):
        prompt = self.prompt_template.replace("__INPUT__" ,user_input)
        response = self._get_completion(prompt)
        semantics = json.loads(response)
        return {k: v for k, v in semantics.items() if v}

class DST:
    def __init__(self):
        pass

    def update(self, state, nlu_semantics):
        if "name" in nlu_semantics:
            state.clear()
        if "sort" in nlu_semantics:
            slot = nlu_semantics["sort"]["value"]
            if slot in state and state[slot]["operator"] == "==":
                del state[slot]
        for k, v in nlu_semantics.items():
            state[k] = v
        return state


class MockedDB:
    def __init__(self):
        self.data = [
            {"name": "经济套餐", "price": 50, "data": 10, "requirement": None},
            {"name": "畅游套餐", "price": 180, "data": 100, "requirement": None},
            {"name": "无限套餐", "price": 300, "data": 1000, "requirement": None},
            {"name": "校园套餐", "price": 150, "data": 200, "requirement": "在校生"},
        ]

    def retrieve(self, **kwargs):
        records = []
        for r in self.data:
            select = True
            if r["requirement"]:
                if "status" not in kwargs or kwargs["status"] != r["requirement"]:
                    continue
            for k, v in kwargs.items():
                if k == "sort":
                    continue
                if k == "data" and v["value"] == "无上限":
                    if r[k] != 1000:
                        select = False
                        break
                if "operator" in v:
                    if not eval(str(r[k]) + v["operator"] + str(v["value"])):
                        select = False
                        break
                elif str(r[k]) != str(v):
                    select = False
                    break
            if select:
                records.append(r)
        if len(records) <= 1:
            return records
        key = "price"
        reverse = False
        if "sort" in kwargs:
            key = kwargs["sort"]["value"]
            reverse = kwargs["sort"]["ordering"] == "descend"
        return sorted(records, key=lambda x: x[key], reverse=reverse)


class DialogManager:
    def __init__(self, prompt_templates):
        self.state = {}
        self.session = [
            {
                "role": "system", # (system/assistant/user) (系统内置角色/助手角色/用户角色)
                "content": "你是一个手机流量套餐的客服代表，你叫小瓜。可以帮助用户选择最合适的流量套餐产品。"
            }
        ]
        #自定义聊天
        self.chat = DeepSeekChat()
        self.nlu = NLU()
        self.dst = DST()
        self.db = MockedDB()
        self.prompt_templates = prompt_templates

    def _wrap(self, user_input, records):
        if records:
            prompt = self.prompt_templates["recommand"].replace("__INPUT__", user_input)
            r = records[0]
            for k, v in r.items():
                prompt = prompt.replace(f"__{k.upper()}__", str(v))
        else:
            prompt = self.prompt_templates["not_found"].replace("__INPUT__", user_input)
            for k, v in self.state.items():
                if "operator" in v:
                    prompt = prompt.replace(f"__{k.upper()}__", v["operator"] + str(v["value"]))
                else:
                    prompt = prompt.replace(f"__{k.upper()}__", str(v))
        return prompt

    def _call_chatgpt(self, prompt, model="deepseek-chat"):
        session = copy.deepcopy(self.session)
        session.append({"role": "user", "content": prompt})
        print(f"session: {session}")
        response = self.chat.get_completion_messages(session, model)
        print(f"response: {response}")
        return response

    def run(self, user_input):
        # 调用NLU获得语义解析
        semantics = self.nlu.parse(user_input)
        print(f'semantics: {semantics}' )
        # 调用DST更新多轮状态
        self.state = self.dst.update(self.state, semantics)
        print(f'state: {self.state}' )
        # 根据状态检索DB，获得满足条件的候选
        records = self.db.retrieve(**self.state)
        print(f'records: {records}')
        # 拼装prompt调用chatgpt
        prompt_for_chatgpt = self._wrap(user_input, records)
        # 调用chatgpt获得回复
        response = self._call_chatgpt(prompt_for_chatgpt)
        # 将当前用户输入和系统回复维护入chatgpt的session
        self.session.append({"role": "user", "content": user_input})
        self.session.append({"role": "assistant", "content": response})
        print(f'add memory session: {self.session}')
        return response

if __name__ == '__main__':
    prompt_templates = {
        "recommand": "用户说：__INPUT__ \n\n向用户介绍如下产品：__NAME__，月费__PRICE__元，每月流量__DATA__G。",
        "not_found": "用户说：__INPUT__ \n\n没有找到满足__PRICE__元价位__DATA__G流量的产品，询问用户是否有其他选择倾向。"
    }

    # 修改语气
    # ext = "很口语，亲切一些。不用说“抱歉”。直接给出回答，不用在前面加“小瓜说：”。NO COMMENTS. NO ACKNOWLEDGEMENTS."
    ext = "\n\n遇到类似问题，请参照以下回答：\n你们流量包太贵了\n亲，我们都是全省统一价哦。"
    prompt_templates = {k: v + ext for k, v in prompt_templates.items()}

    dm = DialogManager(prompt_templates)
    # response = dm.run("流量大的")
    # response = dm.run("300太贵了，200元以内有吗")
    response = dm.run("这流量包太贵了")
    print("===response===")
    print(response)


"""
思维链 

思维树
"""
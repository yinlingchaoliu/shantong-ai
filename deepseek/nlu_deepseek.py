
# 流量包产品
package = """
流量包产品:
| 名称 | 流量（G/月） | 价格（元/月） | 适用人群 |
| :---: | :---: | :---: | :---: | 
| 经济套餐 | 10 | 50 | 无限制 |
| 畅游套餐 | 100 | 180 | 无限制 |
| 无限套餐 | 1000 | 300 | 无限制 |
| 校园套餐 | 200 | 150 | 在校生 |
"""

# 任务描述
instruction = """
你的任务是识别用户对手机流量套餐产品的选择条件。
每种流量套餐产品包含三个属性：名称，月费价格，月流量。
根据用户输入，识别用户在上述三种属性上的倾向。
"""

instruction_v1 = """
你的任务是识别用户对手机流量套餐产品的选择条件。
每种流量套餐产品包含三个属性：名称(name)，月费价格(price)，月流量(data)。
根据用户输入，识别用户在上述三种属性上的倾向。
"""

# 用户输入
input_text = """
办个100G的套餐。
"""

input_text_v1="哪个便宜"

# 输出描述
output_format = """
以JSON格式输出
"""

# 输出描述
output_format_v1 = """
以JSON格式输出。
1. name字段的取值为string类型，取值必须为以下之一：经济套餐、畅游套餐、无限套餐、校园套餐 或 null；

2. price字段的取值为一个结构体 或 null，包含两个字段：
(1) operator, string类型，取值范围：'<='（小于等于）, '>=' (大于等于), '=='（等于）
(2) value, int类型

3. data字段的取值为取值为一个结构体 或 null，包含两个字段：
(1) operator, string类型，取值范围：'<='（小于等于）, '>=' (大于等于), '=='（等于）
(2) value, int类型或string类型，string类型只能是'无上限'

4. 用户的意图可以包含按price或data排序，以sort字段标识，取值为一个结构体：
(1) 结构体中以"ordering"="descend"表示按降序排序，以"value"字段存储待排序的字段
(2) 结构体中以"ordering"="ascend"表示按升序排序，以"value"字段存储待排序的字段

只输出中只包含用户提及的字段，不要猜测任何用户未直接提及的字段，不输出值为null的字段。
"""

output_format_v2 = """
以JSON格式输出。
1. name字段的取值为string类型，取值必须为以下之一：经济套餐、畅游套餐、无限套餐、校园套餐 或 null；

2. price字段的取值为一个结构体 或 null，包含两个字段：
(1) operator, string类型，取值范围：'<='（小于等于）, '>=' (大于等于), '=='（等于）
(2) value, int类型

3. data字段的取值为取值为一个结构体 或 null，包含两个字段：
(1) operator, string类型，取值范围：'<='（小于等于）, '>=' (大于等于), '=='（等于）
(2) value, int类型或string类型，string类型只能是'无上限'

4. 用户的意图可以包含按price或data排序，以sort字段标识，取值为一个结构体：
(1) 结构体中以"ordering"="descend"表示按降序排序，以"value"字段存储待排序的字段
(2) 结构体中以"ordering"="ascend"表示按升序排序，以"value"字段存储待排序的字段

只输出中只包含用户提及的字段，不要猜测任何用户未直接提及的字段。
DO NOT OUTPUT NULL-VALUED FIELD! 确保输出能被json.loads加载。
不要使用```json等代码块标记
"""

examples = """
便宜的套餐：{"sort":{"ordering"="ascend","value"="price"}}
有没有不限流量的：{"data":{"operator":"==","value":"无上限"}}
流量大的：{"sort":{"ordering"="descend","value"="data"}}
100G以上流量的套餐最便宜的是哪个：{"sort":{"ordering"="ascend","value"="price"},"data":{"operator":">=","value":100}}
月费不超过200的：{"price":{"operator":"<=","value":200}}
就要月费180那个套餐：{"price":{"operator":"==","value":180}}
经济套餐：{"name":"经济套餐"}
"""

examples_v1 = """
客服：有什么可以帮您
用户：100G套餐有什么

{"data":{"operator":">=","value":100}}

客服：有什么可以帮您
用户：100G套餐有什么
客服：我们现在有无限套餐，不限流量，月费300元
用户：太贵了，有200元以内的不

{"data":{"operator":">=","value":100},"price":{"operator":"<=","value":200}}

客服：有什么可以帮您
用户：便宜的套餐有什么
客服：我们现在有经济套餐，每月50元，10G流量
用户：100G以上的有什么

{"data":{"operator":">=","value":100},"sort":{"ordering"="ascend","value"="price"}}

客服：有什么可以帮您
用户：100G以上的套餐有什么
客服：我们现在有畅游套餐，流量100G，月费180元
用户：流量最多的呢

{"sort":{"ordering"="descend","value"="data"},"data":{"operator":">=","value":100}}
"""

def context(input_text):
    context = f"""
    客服：有什么可以帮您
    用户：有什么100G以上的套餐推荐
    客服：我们有畅游套餐和无限套餐，您有什么价格倾向吗
    用户：{input_text}
    """
    return context



def nlu_prompt():
    # 这是系统预置的 prompt。魔法咒语的秘密都在这里
    prompt = f"""
    {package}
    
    {instruction}

    用户输入：
    {input_text}
    """
    return prompt

def json_prompt():
    # 这是系统预置的 prompt。魔法咒语的秘密都在这里
    prompt = f"""
    {package}

    {instruction}

    {output_format}

    用户输入:
    {input_text}
    """
    return prompt

def json_prompt_v1():
    # 这是系统预置的 prompt。魔法咒语的秘密都在这里
    prompt = f"""
    {instruction_v1}

    {package}

    {output_format_v1}

    用户输入:
    {input_text}
    """
    return prompt

def json_prompt_v2():
    # 这是系统预置的 prompt。魔法咒语的秘密都在这里
    prompt = f"""
    {instruction_v1}
    {package}
    {output_format_v1}
    例如：
    {examples}
    用户输入：
    {input_text}
    """
    return prompt

def json_prompt_v3():
    # 这是系统预置的 prompt。魔法咒语的秘密都在这里
    prompt = f"""
    {instruction_v1}
    {package}
    {output_format_v1}
    例如：
    {examples_v1}
    {context(input_text)}
    """
    return prompt

def json_prompt_v4():
    # 这是系统预置的 prompt。魔法咒语的秘密都在这里
    prompt = f"""
    {instruction_v1}
    {package}
    {output_format_v2}
    例如:
    {examples_v1}
    {context(input_text_v1)}
    """
    return prompt
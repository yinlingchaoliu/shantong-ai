import json
from langchain.prompts import PromptTemplate, ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_community.llms import OpenAI
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field, validator, field_validator
from typing import List, Dict
import json


from deepseek_langchain import DeepSeekLangChain

"""
conda install -c conda-forge pydantic 
"""
def chinese_friendly(string):
    lines = string.split('\n')
    for i, line in enumerate(lines):
        if line.startswith('{') and line.endswith('}'):
            try:
                lines[i] = json.dumps(json.loads(line), ensure_ascii=False)
            except:
                pass
    return '\n'.join(lines)

class Command(BaseModel):
    command: str = Field(description="linux shell命令名")
    arguments: Dict[str, str] = Field(description="命令的参数 (name:value)")

    # 你可以添加自定义的校验机制
    @field_validator('command')
    def no_space(cls, field):
        if " " in field or "\t" in field or "\n" in field:
            raise ValueError("命令名中不能包含空格或回车!")
        return field


if __name__ == '__main__':
    langchain = DeepSeekLangChain()
    model= langchain.llm()

    parser = PydanticOutputParser(pydantic_object=Command)
    prompt = PromptTemplate(
        template="将用户的指令转换成linux命令.\n{format_instructions}\n{query}\n以JSON格式输出\n不要使用```json等代码块标记",
        input_variables=["query"],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )
    query = "将系统日期设为2023-04-01"
    model_input = prompt.format_prompt(query=query)

    print("Prompt:\n",model_input.to_string())
    output = model.invoke(model_input.to_string())
    print("Output:\n", output)
    cmd = parser.parse(output.content)
    print("cmd:\n",cmd)
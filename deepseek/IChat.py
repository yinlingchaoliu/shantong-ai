from abc import ABC, abstractmethod

"""
    IChat 接口类
"""


class IChat(ABC):

    """
        初始化
    """
    @abstractmethod
    def init_api_key(self):
        pass

    """
        检查连通性
    """
    @abstractmethod
    def check_api_key(self):
        pass

    """
        获取 completions
    """
    @abstractmethod
    def get_completion(self, prompt, model):
        pass

    @abstractmethod
    def get_completion_messages(self, messages, model):
        pass
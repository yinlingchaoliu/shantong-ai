from abc import ABC, abstractmethod

"""
    IModeration 内容审核 接口类
"""
class IModeration(ABC):

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
    def moderation(self, prompt, model):
        pass
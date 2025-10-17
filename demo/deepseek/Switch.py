"""
switch-case
"""
class Switch:
    def __init__(self, case_key):
        self.case_key = case_key
        self.result = None

    def case(self, key, func):
        if self.case_key == key:
            self.result = func()
        return self

    def default(self, func):
        if self.result is None:
            self.result = func()
        return self.result

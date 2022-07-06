class BasePowerFunction():
    def name():
        pass

    def calculate():
        pass


class SimpleLScore(BasePowerFunction):
    def name(self):
        return "Simple Score MSG Count"

    def calculate(self, s_l, s_r):
        return s_l

class SimpleRScore(BasePowerFunction):
    def name(self):
        return "Simple Score Feature"

    def calculate(self, s_l, s_r):
        return s_r

class PowerFuncScore(BasePowerFunction):
    def name(self):
        return "MSG COUNT To The Power of Score Feature"

    def calculate(self, s_l, s_r):
        return s_l.pow(1 + s_r)


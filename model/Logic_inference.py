

import torch
import torch.nn as nn

ONE = torch.Tensor([1]).cuda()
ZERO = torch.Tensor([0]).cuda()


# basic
class LogicInferenceBase(nn.Module):
    def __init__(self, args):
        super(LogicInferenceBase, self).__init__()
        self.args = args
        self.in_channels = args.in_channels
        self.out_channels = args.out_channels
        self.device = args.device
        self.to(self.device)
        
    @staticmethod
    def generalized_softmax(x, y, alpha=20):
        numerator = x * torch.exp(alpha * x) + y * torch.exp(alpha * y)
        denominator = torch.exp(alpha * x) + torch.exp(alpha * y)
        return numerator / denominator 

    @staticmethod
    def generalized_softmin(x, y, alpha=20):
        return -LogicInferenceBase.generalized_softmax(-x, -y, alpha=alpha)

    @staticmethod
    def implication(x, y):
        return LogicInferenceBase.generalized_softmin(ONE, ONE - x + y)

    @staticmethod
    def equivalence(x, y):
        return ONE - torch.abs(x - y)

    @staticmethod
    def negation(x):
        return ONE - x

    @staticmethod
    def weak_conjunction(x, y):
        return LogicInferenceBase.generalized_softmin(x, y)

    @staticmethod
    def weak_disjunction(x, y):
        return LogicInferenceBase.generalized_softmax(x, y)

    @staticmethod
    def strong_conjunction(x, y):
        return LogicInferenceBase.generalized_softmax(ZERO, x + y - 1)

    @staticmethod
    def strong_disjunction(x, y):
        return LogicInferenceBase.generalized_softmin(ONE, x + y)
    def test_forward(self):
        test_input = torch.randn(2, self.in_channels).to(self.device)
        output = self.forward(test_input)
        assert output.shape == (2, self.out_channels), f"\
        input shape is {test_input.shape}, \n\
        Output shape is {output.shape}, \n\
        expected {(2, self.out_channels)}"
        
    def forward(self, x):
        raise NotImplementedError("This method should be implemented by subclass.")

# 2元操作

class LogicInferenceBase2Arity(LogicInferenceBase):
    def __init__(self, args):
        super(LogicInferenceBase2Arity, self).__init__(args)
        
    def split_input(self, x):
        # 拆分输入信号
        half_channels = self.in_channels // 2
        x1 = x[:, :half_channels]
        x2 = x[:, half_channels:]
        return x1, x2
    def repeat_input(self, x):
        return torch.cat([x, x], dim=-1)
    def forward(self, x):
        x1, x2 = self.split_input(x)
        x = self.operation(x1, x2)
        x = self.repeat_input(x)
        return 

    def operation(self, x1, x2):
        raise NotImplementedError("This method should be implemented by subclass.")
    
    def test_forward(self):
        test_input = torch.randn(2, self.in_channels).to(self.device)
        output = self.forward(test_input)
        assert output.shape == (2, self.out_channels), f"\
        input shape is {test_input.shape}, \n\
        Output shape is {output.shape}, \n\
        expected {(2, self.out_channels)}"

# %%
class ImplicationOperation(LogicInferenceBase2Arity):
    def __init__(self, args):
        super(ImplicationOperation, self).__init__(args)
        self.name = "implication"

    def operation(self, x1, x2):
        return LogicInferenceBase.implication(x1, x2)
# %%
class EquivalenceOperation(LogicInferenceBase2Arity):
    def __init__(self, args):
        super(EquivalenceOperation, self).__init__(args)
        self.name = "equivalence"

    def operation(self, x1, x2):
        return LogicInferenceBase.equivalence(x1, x2)
# %%
class NegationOperation(LogicInferenceBase):
    def __init__(self, args):
        super(NegationOperation, self).__init__(args)
        self.name = "negation"

    def forward(self, x):
        # 对于 negation，只使用 x1
        return LogicInferenceBase.negation(x)
# %%
class WeakConjunctionOperation(LogicInferenceBase2Arity):
    def __init__(self, args):
        super(WeakConjunctionOperation, self).__init__(args)
        self.name = "weak_conjunction"

    def operation(self, x1, x2):
        return LogicInferenceBase.weak_conjunction(x1, x2)
# %%
class WeakDisjunctionOperation(LogicInferenceBase2Arity):
    def __init__(self, args):
        super(WeakDisjunctionOperation, self).__init__(args)
        self.name = "weak_disjunction"

    def operation(self, x1, x2):
        return LogicInferenceBase.weak_disjunction(x1, x2)

class StrongConjunctionOperation(LogicInferenceBase2Arity):
    def __init__(self, args):
        super(StrongConjunctionOperation, self).__init__(args)
        self.name = "strong_conjunction"

    def operation(self, x1, x2):
        return LogicInferenceBase.strong_conjunction(x1, x2)
# %%
class StrongDisjunctionOperation(LogicInferenceBase2Arity):
    def __init__(self, args):
        super(StrongDisjunctionOperation, self).__init__(args)
        self.name = "strong_disjunction"

    def operation(self, x1, x2):
        return LogicInferenceBase.strong_disjunction(x1, x2)

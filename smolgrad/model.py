import math
from pprint import pprint


class Wrapper:
    def __init__(self, val, parents=[]):
        self.val = val
        self.grad = 0
        self._backward = lambda: None
        self._parents = set(parents)

    def __add__(self, arg2):
        arg2 = arg2 if isinstance(arg2, Wrapper) else Wrapper(arg2)
        ret = Wrapper(self.val + arg2.val, [self, arg2])

        def _backward():
            self.grad += ret.grad
            arg2.grad += ret.grad

        ret._backward = _backward
        return ret

    def __mul__(self, arg2):
        arg2 = arg2 if isinstance(arg2, Wrapper) else Wrapper(arg2)
        ret = Wrapper(self.val * arg2.val, [self, arg2])

        def _backward():
            self.grad += arg2.val * ret.grad
            arg2.grad += self.val * ret.grad

        ret._backward = _backward
        return ret

    def __pow__(self, arg2):
        arg2 = arg2 if isinstance(arg2, Wrapper) else Wrapper(arg2)
        ret = Wrapper(self.val ** arg2.val, [self, arg2])

        def _backward():
            self.grad += (arg2.val * (self.val ** (arg2.val - 1))) * ret.grad
            arg2.grad += (math.log(self.val) * (self.val ** arg2.val)) * ret.grad

        ret._backward = _backward
        return ret

    def tanh(self):
        ret = Wrapper(math.tanh(self.val), [self])

        def _backward():
            self.grad += (1 - (math.tanh(self.val) ** 2)) * ret.grad

        ret._backward = _backward
        return ret

    def backward(self):
        # implementation of topological sort does not work
        # L ← Empty list that will contain the sorted nodes
        # while exists nodes without a permanent mark do
        #     select an unmarked node n
        #     visit(n)

        # function visit(node n)
        #     if n has a permanent mark then
        #         return
        #     if n has a temporary mark then
        #         stop   (graph has at least one cycle)

        #     mark n with a temporary mark

        #     for each node m with an edge from n to m do
        #         visit(m)

        #     remove temporary mark from n
        #     mark n with a permanent mark
        #     add n to head of L
        L = []
        permanent_mark = set()

        def _visit(node):
            if node in permanent_mark:
                return
            for parent in node._parents:
                _visit(parent)
            permanent_mark.add(node)
            L.insert(0, node)

        _visit(self)
        self.grad = 1
        for node in L:
            node._backward()
        pprint(L)

    def __repr__(self):
        return f"<Wrapper object with val: {self.val}, grad: {self.grad}>"

    def __neg__(self):
        return self * -1

    def __radd__(self, arg2):
        return self + arg2

    def __sub__(self, arg2):
        return self + (-arg2)

    def __rsub__(self, arg2):
        return arg2 + (-self)

    def __rmul__(self, arg2):
        return self * arg2


import torch

x = Wrapper(4.0)
z = 2 * x + 2 + x
q = z + z * x
h = (z * z).tanh()
y = h + q + q * x
y.backward()
xmg, ymg = x, y

x1 = torch.Tensor([4.0]).double()
x1.requires_grad = True
z1 = 2 * x1 + 2 + x1
q1 = z1 + z1 * x1
h1 = (z1 * z1).tanh()
y1 = h1 + q1 + q1 * x1
y1.backward()
xpt, ypt = x1, y1

# print(z.grad,z1.grad.item())
# print(q.grad,q1.grad.item())
# print(h.grad,h1.grad.item())
# print(y.grad,y1.grad.item())
print(x)
print(xmg.grad, xpt.grad.item())
# forward pass went well
assert ymg.val == ypt.data.item()
# backward pass went well
assert xmg.grad == xpt.grad.item()

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
        ret = Wrapper(self.val**arg2.val, [self, arg2])

        def _backward():
            self.grad += (arg2.val * (self.val ** (arg2.val - 1))) * ret.grad
            arg2.grad += (math.log(self.val) * (self.val**arg2.val)) * ret.grad

        ret._backward = _backward
        return ret

    def tanh(self):
        ret = Wrapper(math.tanh(self.val), [self])

        def _backward():
            self.grad += (1 - (math.tanh(self.val) ** 2)) * ret.grad

        ret._backward = _backward
        return ret

    def relu(self):
        ret = Wrapper(0 if self.val < 0 else self.val, [self])

        def _backward():
            self.grad += (ret.val > 0) * ret.grad
        ret._backward = _backward

        return ret
    
    def log(self):
        try:
            ret = Wrapper(math.log(self.val), [self])
        except:
            print(self)
            raise RuntimeError('log is not working')

        def _backward():
            self.grad += (1 / self.val) * ret.grad

        ret._backward = _backward
        return ret

    def softmax(self, all_args):
        try:
            numerator = math.exp(self.val)
            denominator = sum([math.exp(x.val) for x in all_args])
        except:
            print(all_args)
            raise RuntimeError('softmax failed')
        ret = Wrapper(numerator / denominator, all_args)

        def _backward():
            for node in all_args:
                # j is the current node (self)
                if node is self:
                    # i == j
                    node.grad += (numerator * (1 - numerator)) * ret.grad
                else:
                    # i != j
                    pj = math.exp(node.val)
                    node.grad += (-(pj*numerator)) * ret.grad

        ret._backward = _backward
        if ret.val < 0:
            print(all_args)
            raise RuntimeError('wrong')
        return ret

    def backward(self):
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
        temporary_mark = set()

        def _visit(node):
            if node in permanent_mark:
                return
            if node in temporary_mark:
                raise RuntimeError('at least one cycle detected')
            temporary_mark.add(node)
            for parent in node._parents:
                _visit(parent)
            temporary_mark.remove(node)
            permanent_mark.add(node)
            L.insert(0, node)

        _visit(self)
        self.grad = 1
        for node in L:
            node._backward()
        #pprint(L)

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

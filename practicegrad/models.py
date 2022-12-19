import math
import operator as op
import random
from typing import List
from practicegrad.wrapper import Wrapper


class Neuron:
    def __init__(self, num_ingress: int):
        self.weight = [
            Wrapper(self._xavier_initialization(num_ingress))
            for _ in range(num_ingress)
        ]
        self.bias = Wrapper(self._xavier_initialization(num_ingress + 1))

    def forward(self, input: List[float]):
        # multiply our weights with the inputs plus bias
        # and apply non-linearity
        out = sum(map(op.mul, input, self.weight), self.bias)
        return out.tanh()

    def _xavier_initialization(self, num_ingress: int):
        lower = -(1.0 / math.sqrt(num_ingress))
        upper = 1.0 / math.sqrt(num_ingress)
        return random.uniform(lower, upper)


class Layer:
    def __init__(self, num_ingress: int, num_egress: int):
        self.neurons = [Neuron(num_ingress) for _ in range(num_egress)]

    def forward(self, input: List[float]):
        out = [neuron.forward(input) for neuron in self.neurons]
        return out


class MLP:
    def __init__(self, dims: List[int]):
        self.layers = [Layer(dims[i], dims[i + 1]) for i in range(len(dims) - 1)]

    def forward(self, input: List[float]):
        last_out = None
        for layer in self.layers:
            if last_out is None:
                last_out = layer.forward(input)
            else:
                last_out = layer.forward(last_out)
        assert last_out is not None
        return last_out



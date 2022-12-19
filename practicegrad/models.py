import math
import operator as op
import random
from typing import List
from practicegrad.wrapper import Wrapper

class Module:

    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0

    def parameters(self):
        return []

class Neuron(Module):
    def __init__(self, num_ingress: int, neuron_nonlin: str = 'relu'):
        self.weight = [
            Wrapper(self._xavier_initialization(num_ingress))
            for _ in range(num_ingress)
        ]
        self.bias = Wrapper(self._xavier_initialization(num_ingress + 1))
        self.neuron_nonlin = neuron_nonlin

    def forward(self, input: List[Wrapper]):
        # multiply our weights with the inputs plus bias
        # and apply non-linearity
        out = sum(map(op.mul, input, self.weight), self.bias)
        if self.neuron_nonlin == 'tanh':
            return out.tanh()
        elif self.neuron_nonlin == 'relu':
            return out.relu()
        elif self.neuron_nonlin == 'linear':
            return out
        else:
            RuntimeError(f'nonlin {self.neuron_nonlin} unrecognized.')

    def _xavier_initialization(self, num_ingress: int):
        lower = -(1.0 / math.sqrt(num_ingress))
        upper = 1.0 / math.sqrt(num_ingress)
        return random.uniform(lower, upper)
    
    def get_parameters(self):
        return self.weight + [self.bias]


class Layer(Module):
    def __init__(
        self,
        num_ingress: int,
        num_egress: int,
        neuron_nonlin: str = 'tanh',
        layer_nonlin: str = None,
    ):
        self.neurons = [Neuron(num_ingress, neuron_nonlin) for _ in range(num_egress)]
        self.layer_nonlin = layer_nonlin

    def forward(self, input: List[Wrapper]):
        out = [neuron.forward(input) for neuron in self.neurons]
        if self.layer_nonlin == 'softmax':
            out =  [x.softmax(out) for x in out]
        return out

    def get_parameters(self):
        return [p for neuron in self.neurons for p in neuron.get_parameters()]

class MLP(Module):
    def __init__(self, dims: List[int]):
        self.layers = [Layer(dims[i], dims[i + 1]) for i in range(len(dims) - 2)]
        self.layers.append(Layer(dims[-2], dims[-1], neuron_nonlin='linear')) #, layer_nonlin='softmax'))

    def forward(self, input: List[Wrapper]):
        last_out = None
        for layer in self.layers:
            if last_out is None:
                last_out = layer.forward(input)
            else:
                last_out = layer.forward(last_out)
        assert last_out is not None
        return last_out if len(last_out) > 1 else last_out[0]

    def get_parameters(self):
        return [p for layer in self.layers for p in layer.get_parameters()]
    

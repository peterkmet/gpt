import random
from grad import Value

class Neuron:
    def __init__(self, nin, name):
        self.w = [Value(random.uniform(-1,1), label=f'{name}_w{i}') for i in range(nin)]
        self.b = Value(random.uniform(-1,1), label=f'{name}_b')

    def __call__(self,x):
        act = sum((wi*xi for wi, xi in zip(self.w, x)), self.b)
        self.out = act.tanh()
        return self.out

class Layer:
    def __init__(self, nin, nout, name):
        self.neurons = [Neuron(nin, f'{name}_n{i}') for i in range(nout)]

    def __call__(self,x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs

class MLP:
    def __init__(self, nin, nouts):
        sz = [nin] + nouts # MLP(3, [4, 4, 1]) => [3, 4, 4, 1]
        self.layers = [Layer(sz[i], sz[i+1], f'L{i}') for i in range(len(nouts))]

    def __call__(self,x):
        self.x = x
        for layer in self.layers:
            x = layer(x)
        return x



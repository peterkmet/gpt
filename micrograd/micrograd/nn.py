import random
from .grad import Value

class Neuron:
    def __init__(self, nin, name, activation=None):
        self.w = [Value(random.uniform(-1,1), label=f'{name}_w{i}') for i in range(nin)]
        self.b = Value(random.uniform(-1,1), label=f'{name}_b')
        self.activation = activation  # 'relu' | 'tanh' | None

    def __call__(self, x):
        act = sum((wi*xi for wi, xi in zip(self.w, x)), self.b)
        self.out = getattr(act, self.activation)() if self.activation else act
        return self.out

    def parameters(self):
        return self.w + [self.b] 

class Layer:
    def __init__(self, nin, nout, name, activation=None):
        self.neurons = [Neuron(nin, f'{name}_n{i}', activation=activation) for i in range(nout)]

    def __call__(self,x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs

    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]

class MLP:
    def __init__(self, nin, layers):
        self.layers = []
        prev = nin
        for i, (nout, activation) in enumerate(layers):
            self.layers.append(Layer(prev, nout, f'L{i}', activation=activation))
            prev = nout
        
    def __call__(self,x):
        self.x = x
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0.0


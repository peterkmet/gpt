from grad import Value
from nn import MLP
from visualize import draw_mlp

x = [
    Value(2.0, label='x0'), 
    Value(3.0, label='x1'), 
    Value(-1.0, label='x2')
]
n = MLP(3, [4, 4, 1])

out = n(x)
out.label = 'out'

out.backward()

dot = draw_mlp(n)
dot.render('neural_network', format='svg', cleanup=True)
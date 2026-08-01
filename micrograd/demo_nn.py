from grad import Value
from nn import MLP
from visualize import draw_mlp, draw_dot

network: MLP = MLP(3, [4, 4, 1]) # 3 inputs, 2 layers of 4 neurons 1 layer of 1 neuron

inputs = [
    Value(2.0, label='x0'), 
    Value(3.0, label='x1'), 
    Value(-1.0, label='x2')
]

out: Value = network(inputs) 
out.label = 'out'
out.backward()
draw_mlp(network).render('neural_network', format='svg', cleanup=True)
draw_dot(out).render('neural_network_computation', format='svg', cleanup=True)

from micrograd.grad import Value
from micrograd.nn import MLP
from micrograd.visualize import draw_mlp, draw_dot

# initialize 3,4,4,1 network with random weights
network: MLP = MLP(3, [4, 4, 1]) # 3 inputs, 2 layers of 4 neurons 1 layer of 1 neuron

# 4 inputs
xs = [
    [2.0, 3.0, -1.0],
    [3.0, -1.0, 0.5],
    [0.5, 1.0, 1.0],
    [1.0, 1.0, -1.0],
]

# expected outputs
ys = [1.0, -1.0, -1.0, 1.0]

# run network for predictions
ypred = [network(x) for x in xs]
print(ypred)

# loss of network predictions vs expected outputs
loss = sum([(yout-ygt)**2 for ygt, yout in zip(ys, ypred)]) # ygt - y_grandtruth
print(loss)

# bacward propagation for all 4 passes!
loss.backward() # PURE GENIUS

draw_mlp(network).render('svg/neural_network', format='svg', cleanup=True)
draw_dot(loss).render('svg/neural_network_computation', format='svg', cleanup=True)

#out.backward()
#draw_dot(out).render('neural_network_computation', format='svg', cleanup=True)

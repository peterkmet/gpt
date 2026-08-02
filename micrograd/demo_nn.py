from micrograd.nn import MLP
from micrograd.visualize import draw_mlp, draw_dot

# initialize 3,4,4,1 network with random weights
network: MLP = MLP(3, [(4, 'tanh'), (4,'tanh'), (1, 'tanh')]) # 3 inputs, 2 layers of 4 neurons 1 layer of 1 neuron

# 4 inputs
xs = [
    [2.0, 3.0, -1.0],
    [3.0, -1.0, 0.5],
    [0.5, 1.0, 1.0],
    [1.0, 1.0, -1.0],
]
# expected outputs
ys = [1.0, -1.0, -1.0, 1.0]

def loss():
    ypred = [network(x) for x in xs] # run network to get predictions 
    return sum([(yout-ygt)**2 for ygt, yout in zip(ys, ypred)]) # loss of network - predictions vs expected outputs

for k in range(40):
    # forward
    total_loss = loss() # return full graph of all instances erorrs aggregated

    # backward
    network.zero_grad()
    total_loss.backward() # PURE GENIUS

    # update (sgd)
    learning_rate = 0.1 - 0.09*k/40
    for p in network.parameters():
        p.data -= learning_rate * p.grad

    print(k, total_loss.data)

draw_mlp(network).render('svg/neural_network', format='svg', cleanup=True)
draw_dot(total_loss).render('svg/neural_network_computation', format='svg', cleanup=True)
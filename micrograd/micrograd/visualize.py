import numpy as np
from graphviz import Digraph
import matplotlib.pyplot as plt

def trace(root):
    nodes, edges = set(), set()
    def build(v):
        if v not in nodes:
            nodes.add(v)
            for child in v._prev:
                edges.add((child,v))
                build(child)
    build(root)
    return nodes, edges

def draw_dot(root):
    dot = Digraph(format='svg', graph_attr={'rankdir': 'LR', 'splines': 'line'})

    nodes, edges = trace(root)
    for n in nodes:
        uid = str(id(n))
        dot.node(name = uid, label = "{ %s | d %.2f | g %.2f}" % (n.label, n.data, n.grad), shape='record')
        if n._op:
            dot.node(name = uid + n._op, label = n._op)
            dot.edge(uid + n._op, uid + ':w')

    for n1, n2 in edges:
        dot.edge(f'{id(n1)}:e', f'{id(n2)}{n2._op}:w')

    return dot

def draw_mlp(mlp):
    dot = Digraph(format='svg', graph_attr={'rankdir': 'LR', 'splines': 'line'})

    prev = []
    # net inputs
    for i, xi in enumerate(mlp.x):
        uid = f'x{i}'
        dot.node(uid, label=f'{{ {uid} | <out> {getattr(xi, "data", xi):.2f} }}', shape='record')
        prev.append(uid)

    # net layers
    for li, layer in enumerate(mlp.layers):
        ids = []
        for ni, neuron in enumerate(layer.neurons):
            uid = f'L{li}n{ni}'
            ws = ' | '.join(f'<w{j}> {w.data:.2f}' for j, w in enumerate(neuron.w))
            label = f'{uid} | {{ {{ {ws} }} | {neuron.b.data:.2f} | <out> {neuron.out.data:.2f} }}'
            dot.node(uid, label=label, shape='record')
            for j, p in enumerate(prev):
                dot.edge(f'{p}:out:e', f'{uid}:w{j}:w')
            ids.append(uid)
        prev = ids

    return dot

def draw_digit(imgs):
    imgs = np.asarray(imgs)
    if imgs.ndim == 2:          # single 8x8 digit
        imgs = imgs[None]       # -> (1, 8, 8)
    N = int(np.ceil(np.sqrt(len(imgs))))
    pad = np.zeros((N*N - len(imgs), 8, 8))
    grid = np.concatenate([imgs, pad]).reshape(N, N, 8, 8)
    big = grid.transpose(0, 2, 1, 3).reshape(N*8, N*8)

    fig = plt.figure(figsize=(12, 12))
    plt.imshow(big, cmap=plt.get_cmap('gray', 16))
    plt.axis('off')
    return fig
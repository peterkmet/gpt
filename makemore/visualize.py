from graphviz import Digraph
from data import itos

def draw_model(itos):
    V = len(itos)
    dot = Digraph(format='svg')
    dot.attr(rankdir='LR', splines='line', nodesep='0.12', ranksep='3.5')

    def column(prefix, labels, width):
        with dot.subgraph() as c:
            c.attr(rank='same')
            c.attr('node', shape='circle', fixedsize='true', width=width, fontsize='10')
            for i in range(V):
                c.node(f'{prefix}{i}', labels[i] if labels else '')
            for i in range(V - 1):
                c.edge(f'{prefix}{i}', f'{prefix}{i+1}', style='invis')

    column('x', itos, '0.35')   # inputs: current char (one-hot)
    column('n', None, '0.28')   # neurons: plain units, no letter identity
    column('y', itos, '0.35')   # outputs: P(next char)

    for i in range(V):          # W: every input to every neuron
        for j in range(V):
            dot.edge(f'x{i}', f'n{j}', color='gray75', penwidth='0.3', arrowsize='0.3')
    for j in range(V):          # each neuron's logit is one next-char score
        dot.edge(f'n{j}', f'y{j}', color='gray40', penwidth='0.6', arrowsize='0.4')
    return dot

if __name__ == '__main__':
    draw_model(itos).render('model', cleanup=True)  # -> model.svg
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 15 20:51:30 2024

@author: junga1
"""

import matplotlib.pyplot as plt
import networkx as nx
from networkx.algorithms import bipartite

# Initialize the multilayer network (using Graph for simplicity here)
G = nx.Graph()

# Adding nodes for layer 1
layer1 = ['A1', 'B1', 'C1', 'D1']
G.add_nodes_from(layer1, layer='Employee')

# Adding nodes for layer 2
layer2 = ['A2', 'B2', 'C2', 'D2']
G.add_nodes_from(layer2, layer='Manager')

# Adding edges within layer 1
G.add_edges_from([('A1', 'B1'), ('B1', 'C2'), ('C1', 'D1')])

# Adding edges between layers
G.add_edges_from([('A1', 'A2'), ('B1', 'B2'), ('C1', 'C2'), ('D1', 'D2')])

# Plotting the network
pos = dict()
pos.update((node, (1, index * 10)) for index, node in enumerate(layer1))  # Layer 1 at x=1
pos.update((node, (2, index * 10)) for index, node in enumerate(layer2))  # Layer 2 at x=2

# Nodes
nx.draw_networkx_nodes(G, pos, nodelist=layer1, node_color='lightblue')
nx.draw_networkx_nodes(G, pos, nodelist=layer2, node_color='lightgreen')

# Edges
nx.draw_networkx_edges(G, pos, edgelist=G.edges, style='dotted', alpha=0.5)

# Labels
labels = {node: node for node in G.nodes()}
nx.draw_networkx_labels(G, pos, labels=labels)

# Layer labels
plt.text(1, max(pos[node][1] for node in layer1) + 5, 'Layer 1', horizontalalignment='center')
plt.text(2, max(pos[node][1] for node in layer2) + 5, 'Layer 2', horizontalalignment='center')


plt.title("Simple Multilayer Network")
plt.axis('off')  # Turn off the axis
plt.show()

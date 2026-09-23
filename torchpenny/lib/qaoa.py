from pennylane.pauli import PauliSentence
import pennylane as qml


from networkx import Graph

from torchpenny import QLayer, QSubLayer
from torchpenny.lib.vqe import VQE 


class QAOA(VQE): # VQE Layer
    def __init__(self, ops:PauliSentence, ansatz:QSubLayer, *args, **kwargs):
        super(QAOA, self).__init__(ops, ansatz, *args, **kwargs)
    @classmethod
    def from_nx_graph(cls, graph:Graph, ansatz:QSubLayer, *args, **kwargs):
        edge_list = graph.edges
        pauliZ_words = sum([0.5*qml.PauliZ(i)@qml.PauliZ(j) for i, j in edge_list])
        Is = -0.5*len(edge_list)*qml.I(wires=[i for i in range(len(graph.nodes))])
        H = Is + pauliZ_words
        return cls(H.pauli_rep, ansatz, *args, **kwargs)
    
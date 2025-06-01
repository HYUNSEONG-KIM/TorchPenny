
from pennylane.pauli import PauliSentence
import pennylane as qml

from torch.nn import Parameter
import torch

from networkx import Graph

from torchpenny import QLayer, QSubLayer
from torchpenny.lib.trotter import TrotterAnsatz
from torchpenny.lib.vqe import VQE


class QAOAAnsatz(TrotterAnsatz):
    def __init__(self, wires, ops:PauliSentence, p=1):
        self.p = p
        super(QAOAAnsatz, self).__init__(wires, ops, reps=self.p, order=1)
        
        self.params_mixer = self._init_weights_mixer()
        
    def _init_weights_mixer(self): # Mixer weight
        n = self.p * self.wires
        return Parameter(torch.rand(n))
    
    def forward(self, wires=[]):
        for o in range(self.p):
            p_o = o*self.wires
            for i, p in enumerate(self.params_mixer[p_o: p_o + self.wires]):
                qml.RX(p, wires=i)
            super().forward(wires, on=o)
            qml.Barrier(wires=range(self.wires))
            

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
    
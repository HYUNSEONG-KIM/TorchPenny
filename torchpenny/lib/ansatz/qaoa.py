
from pennylane.pauli import PauliSentence
import pennylane as qml

from torch.nn import Parameter
import torch

from torchpenny.lib.ansatz.trotter import TrotterAnsatz


class QAOAAnsatz(TrotterAnsatz):
    def __init__(self, wires, ops:PauliSentence, p=1):
        self.p = p # QAOA layer repeatition.
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
            


import pennylane as qml
from pennylane.pauli import PauliWord
import torch


from torchpenny import QSubLayer

class TrotterAnsatz(QSubLayer):
    def __init__(self, wires, ops, reps=1, order=1):
        self.ops = ops
        self.reps = reps
        self.order = order # For 2nd order Suzuki-Trotter circuit.
        super(TrotterAnsatz, self).__init__(wires, param_received=False)
    def init_weights(self):
        n = len(self.ops)
        return torch.rand(self.reps*n)

    def forward(self, wires=[], on=None):
        assert len(wires) == (self.wires), "Applied wires must be matched with the defined qubit dimension."
        
        range_wires = range(self.wires)
        n = len(self.ops)
        if on is None:
            for i in range(self.reps):
                if i!=0:
                    qml.Barrier(wires=range_wires)
                ni = n*i
                for p, pauli_word in zip(self.params[ni: ni+n], self.ops):
                    self._pauil2circuit(p, pauli_word, range_wires)
        else: # For QAOA Ansatz interface
            i = on
            ni = n*i
            for p, pauli_word in zip(self.params[ni: ni+n], self.ops):
                self._pauil2circuit(p, pauli_word, range_wires)
    
    def _pauil2circuit(self, weight, pauli:PauliWord, wires=[]):
        n = len(pauli)
        cx_map = []
        assert len(wires) >= n, f"Given Pauli word requires at least {n} wires."
        
        # Basis transformation
        for i, v in pauli.items():
            if v == "I": continue
            if v == "X":
                qml.Hadamard(i)
            elif v =="Y":
                qml.adjoint(qml.S(i))
                qml.Hadamard(i)
            cx_map.append(i)
        if len(cx_map) ==0: return None # III...III case
        w_last = cx_map[-1]
        
        del(cx_map[n-1])
        
        for cx_i in cx_map:
            qml.CNOT(wires=[wires[cx_i], w_last])
        
        qml.RZ(weight, wires=wires[w_last])
        
        for cx_i in reversed(cx_map):
            qml.CNOT(wires=[wires[cx_i], w_last])
        
        # Basis restoration
        for i, v in pauli.items():
            if v == "X":
                qml.Hadamard(i)
            elif v =="Y":
                qml.adjoint(qml.S(i))
                qml.Hadamard(i)
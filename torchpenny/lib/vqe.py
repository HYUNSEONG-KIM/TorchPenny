from pennylane.pauli import PauliSentence
import pennylane as qml
import numpy as np
#from tp import tpd

from torchpenny import QLayer, QSubLayer

class VQE(QLayer): # VQE Layer
    def __init__(self, ops:PauliSentence, ansatz:QSubLayer, *args, **kwargs):
        self.ops = ops
        super(VQE, self).__init__(*args, **kwargs)
        self.ansatz = ansatz
        self._get_loss = True
    @classmethod
    def from_hamiltonian(cls, mat:np.matrix, ansatz:QSubLayer, *args, **kwargs):
        # Hamiltonian -> Paulilist
        H_pauli_sentence = qml.pauli_decompose(mat, pauli=True)
        return cls(H_pauli_sentence , ansatz)

    @property
    def measurement_value(self):
        return "loss" if self._get_loss else "sample"
    def set_measure(self, loss=False):
        self._get_loss = loss
    def expval(self):
        self._get_loss = True
    def sample(self, shots=300):
        if self._get_loss:
            self.set_measure(loss=False)
            if self.q_device.shots.total_shots is None:
                self.update_qdevice("default.qubit", q_device_kwargs={"wires": self.wires, "shots": shots},
                                    qnode_kwargs={"diff_method": "best"}) # Finite-shot sampling cannot use backprop.
        
        return self()
    def inner_gates(self):
        self.ansatz(wires=range(self.wires))
    def measurement(self):
        return qml.expval(self.ops.operation()) if self._get_loss else qml.sample()


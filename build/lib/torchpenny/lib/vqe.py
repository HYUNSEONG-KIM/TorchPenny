from pennylane.pauli import PauliSentence
import pennylane as qml


from torchpenny import QLayer, QSubLayer

class VQE(QLayer): # VQE Layer
    def __init__(self, ops:PauliSentence, ansatz:QSubLayer, *args, **kwargs):
        self.ops = ops
        super(VQE, self).__init__(*args, **kwargs)
        self.ansatz = ansatz
        self._get_loss = True
    @property
    def measurement_value(self):
        return "loss" if self._get_loss else "sample"
    def set_measure(self, loss=True):
        self._get_loss = loss
    def expval(self):
        self._get_loss = True
    def sample(self, shots=300):
        if self._get_loss:
            self.set_measure()
            if self.q_device.shots.total_shots is None:
                self.update_qdevice("default.qubit", q_device_kwargs={"wires": self.wires, "shots": shots}) # Update qdevice
        return self()
    def inner_gates(self, x = None):
        self.ansatz(wires=range(self.wires))
    def measurement(self):
        return qml.expval(self.ops.operation()) if self._get_loss else qml.sample()

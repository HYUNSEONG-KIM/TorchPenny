from torchpenny.module import QSubLayer

import pennylane as qml
import torch
from torch.nn.parameter import Parameter

class ZfeatureMap(QSubLayer):
    def init_weights(self):
        data = torch.empty(self.wires, requires_grad = False) if self.param_received else torch.rand(self.wires)
        return data.unsqueeze(0) if self.param_received else Parameter(data.unsqueeze(0)) 
    
    def forward(self, x=None, wires=[]):
        assert len(wires) == (self.wires), "Applied wires must be matched with the defined qubit dimension."
        params = x if self.param_received and x is not None else self.params

        for i, w in enumerate(wires):
            qml.Hadamard(wires=w)
            qml.RZ(params[:, i], w) # batch input을 위해 항상 [:, .] 형태로 두는 것을 권장.

class ZZfeatureMap(QSubLayer):
    entanglement_structure= [
        "full", "linear", "reverse_linear", "circular", "sca"
    ]
    def __init__(self, 
            wires:int, 
            param_received=False, 
            entanglement_structure ="linear", 
            only_zz=False):
        self.entanglement_structure  = entanglement_structure 
        self.only_zz = only_zz

        super(ZZfeatureMap, self).__init__(wires=wires, param_received=param_received)
        

    def init_weights(self):
        n = self.wires-1 if self.only_zz else 2*self.wires -1
        data = torch.empty(n, requires_grad = False) if self.param_received else torch.rand(n)
        return data.unsqueeze(0) if self.param_received else Parameter(data.unsqueeze(0)) 
    
    def forward(self, x=None, wires=[]):
        assert len(wires) == (self.wires), "Applied wires must be matched with the defined qubit dimension."
        if x is not None:
            assert x.shape[-1] == self.num_params, "The given data was not matched with the layer input."
            params = x
        else:
            params = self.params

        if not self.only_zz:
            for i, w in enumerate(wires):
                qml.Hadamard(wires=w)
                qml.RZ(params[:, i], w)

            params_entangle = params[:, len(wires):]
            entangle_dim = len(params[0]) - len(wires)
        else:
            params_entangle = params
            entangle_dim = len(wires)-1
        qml.Barrier(wires=range(self.wires))

        for i in range(entangle_dim):
            p = params_entangle[:, i]
            wi = wires[i]
            wj = wires[i+1]

            qml.CNOT([wi, wj])
            qml.RZ(p, wj)
            qml.CNOT([wi, wj])

        if self.entanglement_structure == "circular":
            qml.CNOT([wj, wires[0]])
        

            


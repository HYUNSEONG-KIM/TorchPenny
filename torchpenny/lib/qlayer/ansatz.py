from typing import Union, Tuple
from itertools import combinations

import pennylane as qml
import torch
from torch.nn.parameter import Parameter

from torchpenny.module import QSubLayer
from torchpenny.lib.qlayer.utils import ENTANGLEMTNS, ROTATION_GATES, get_entanglement_map


class TwoLocalAnsatz(QSubLayer):
    def __init__(self, 
                 wires:int,
                 rotations:Union[str, Tuple[str, ...]] = 'rx',
                 entanglments:str = "cx",
                 entanglement_structure = "full",
                 except_initial = False
                 ):
        self.entanglement_structure = entanglement_structure
        self.entanglements = ENTANGLEMTNS[entanglments]
        self.except_initial = except_initial 
        self.rotations =  [ROTATION_GATES[rotations]] if isinstance(rotations, str) else [ROTATION_GATES[r] for r in (rotations)]

        super(TwoLocalAnsatz, self).__init__(wires=wires, param_received=False)
    
    def init_weights(self):
        k = len(self.rotations)
        l = 1 if self.except_initial  else 2
        #n = self.wires * k * l

        data = torch.rand((l, self.wires, k))
        return Parameter(data) 
    
    def forward(self, wires=[]):
        assert len(wires) == (self.wires), "Applied wires must be matched with the defined qubit dimension."
        params = self.params
        entanglement_map = get_entanglement_map(self.wires, self.entanglement_structure)
        layer, _, sub_layer = self.params.shape
        
        init = 1
        if self.except_initial:
            init = 0
        else:
            for i, w in enumerate(wires):
                for sl in range(sub_layer):
                    self.rotations[sl](params[0, i, sl], w)
        # Fix
        for l in range(init, layer):
            # Entanglement
            for i, j in entanglement_map:
                self.entanglements((wires[i],wires[j]))
            for i, w in enumerate(wires):
                for sl in range(sub_layer):
                    self.rotations[sl](params[l, i, sl], w)
                
            
        
class RealAmplitude(TwoLocalAnsatz):
    def __init__(self, wires=int, entanglement_structure="full", only_entangle=False):
        super(RealAmplitude, self).__init__(wires, ["ry"], "cx", entanglement_structure, only_entangle)


                


        
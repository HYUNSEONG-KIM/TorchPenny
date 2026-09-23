from typing import Union, Tuple
from itertools import combinations

import pennylane as qml
import torch
from torch.nn.parameter import Parameter

from torchpenny.module import QSubLayer
from torchpenny.lib.utils import ENTANGLEMTNS, ROTATION_GATES, get_entanglement_map


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
        params = self.get_params()
        
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
    """Trainable RY/CNOT ansatz inspired by Qiskit's RealAmplitudes.

    The circuit repeats RY -> CNOT ``reps`` times, then applies a final
    RY layer unless ``skip_final_rotation_layer`` is True. Starting from
    |0>, these gates prepare a state with real amplitudes.

    ``only_entangle`` keeps its legacy meaning: skip the first RY layer,
    not all rotation layers. Defaults remain ``full`` and one repetition
    for compatibility with TorchPenny, rather than Qiskit's defaults.
    Parameters have shape (rotation_layers, wires, 1).

    Supported structures are full, linear, reverse_linear, circular,
    pairwise and sca. Circular gates follow Qiskit's closing-edge-first
    order, unlike the legacy TwoLocalAnsatz helper. Custom entangler maps
    and Qiskit-specific circuit options are not supported.
    """
    def __init__(self,
                 wires:int,
                 entanglement_structure="full",
                 only_entangle=False,
                 *,
                 reps:int=1,
                 skip_final_rotation_layer=False):
        if type(reps) is not int:
            raise TypeError("`reps` must be an integer.")
        if reps < 0:
            raise ValueError("`reps` must be a non-negative integer.")
        if type(only_entangle) is not bool or type(skip_final_rotation_layer) is not bool:
            raise TypeError("Rotation layer options must be bool values.")
        if not isinstance(entanglement_structure, str):
            raise TypeError("`entanglement_structure` must be a string.")
        if entanglement_structure not in ("full", "linear", "reverse_linear", "circular", "pairwise", "sca"):
            raise ValueError("Unsupported entanglement structure.")

        self.reps = reps
        self.skip_final_rotation_layer = skip_final_rotation_layer
        super(RealAmplitude, self).__init__(wires, ["ry"], "cx", entanglement_structure, only_entangle)

    def init_weights(self):
        l = max(0, self.reps + 1 - self.except_initial - self.skip_final_rotation_layer)
        return torch.rand((l, self.wires, 1))

    def _entanglement_map(self, layer):
        if self.wires == 1:
            return []
        if self.entanglement_structure == "full":
            return list(combinations(range(self.wires), 2))

        entanglement_map = [(i, i+1) for i in range(self.wires-1)]
        if self.entanglement_structure == "linear":
            return entanglement_map
        if self.entanglement_structure == "reverse_linear":
            return entanglement_map[::-1]
        if self.entanglement_structure == "pairwise":
            return entanglement_map[::2] + entanglement_map[1::2]

        if self.wires > 2:
            entanglement_map = [(self.wires-1, 0)] + entanglement_map
        if self.entanglement_structure == "sca":
            shift = layer % len(entanglement_map)
            entanglement_map = entanglement_map[-shift:] + entanglement_map[:-shift]
            if layer % 2:
                entanglement_map = [(j, i) for i, j in entanglement_map]
        return entanglement_map

    def forward(self, wires=None):
        wires = qml.wires.Wires(range(self.wires) if wires is None else wires)
        if len(wires) != self.wires:
            raise ValueError("Applied wires must be matched with the defined qubit dimension.")
        params = self.get_params()
        l = 0
        for layer in range(self.reps):
            if not (self.except_initial and layer == 0):
                for i, w in enumerate(wires):
                    self.rotations[0](params[l, i, 0], wires=w)
                l += 1
            for i, j in self._entanglement_map(layer):
                self.entanglements(wires=[wires[i], wires[j]])

        if not self.skip_final_rotation_layer and not (self.reps == 0 and self.except_initial):
            for i, w in enumerate(wires):
                self.rotations[0](params[l, i, 0], wires=w)


                


        

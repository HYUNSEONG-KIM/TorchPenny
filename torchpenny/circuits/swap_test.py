from typing import Tuple
from functools import partial

from pennylane.operation import Operation, AnyWires 
from pennylane.wires import Wires
from pennylane import Hadamard, CSWAP


class SWAPtest(Operation):
    num_wires = AnyWires

    def __init__(self, input_a:Tuple[int], input_b:Tuple[int], ancilla:int,  id=None):
        assert len(input_a) == len(input_b), "The given two wire ranges must have same length."
        assert len(set(input_a).intersection(set(input_b))) ==0, "The given two wire ranges must not be intersected."
        assert ancilla not in input_a and ancilla not in input_b, "Ancilla wire must not be in test wire range."

        all_wires = Wires(input_a) + Wires(input_b) + Wires(ancilla)

        self.input_a = input_a
        self.input_b = input_b
        self.ancilla = ancilla
         
        super().__init__(wires=all_wires, id=id)
    @property
    def num_params(self):
        return 0
    def adjoint(self):
        return SWAPtest(self.input_a, self.input_b, self.ancilla)
    @staticmethod
    def compute_decomposition(*params, wires = None, **hyperparameters):
        ancilla = wires[-1]
        sep_index = int(len(wires[:-1])/2)
        input_a = wires[:sep_index]
        input_b = wires[sep_index:-1]


        ops = []
        ops.append(Hadamard(wires = ancilla))
        for a, b in zip(input_a, input_b):
            ops.append(CSWAP((ancilla, a, b)))
            
        ops.append(Hadamard(wires = ancilla))
        return ops
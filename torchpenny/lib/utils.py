from itertools import combinations

import pennylane as qml

ROTATION_GATES = {
    "rx": qml.RX,
    "ry": qml.RY,
    "rz": qml.RZ
}
ENTANGLEMTNS = {
    "cx" : qml.CNOT,
    "cz" : qml.CZ
}

def get_entanglement_map(qubit:int, structure:str):
    if structure == "full":
        return combinations(range(qubit),2)
    elif structure=="linear":
        return [(i, i+1) for i in range(qubit-1)]
    elif structure == "circular":
        return [(i, i+1) for i in range(qubit-1)] + [(qubit-1, 0)]

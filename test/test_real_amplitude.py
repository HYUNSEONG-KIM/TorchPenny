"""Run from the project root: python -m unittest discover -s test -v."""
import importlib.util
import unittest

import pennylane as qml
import torch

from torchpenny import QLayer
from torchpenny.lib import VQE
from torchpenny.lib.ansatz import RealAmplitude, TwoLocalAnsatz


STRUCTURES = ("full", "linear", "reverse_linear", "circular", "pairwise", "sca")


def operations(layer, wires=None):
    with qml.queuing.AnnotatedQueue() as queue:
        layer(wires=wires)
    return qml.tape.QuantumScript.from_queue(queue).operations


def state(layer):
    @qml.qnode(qml.device("default.qubit", wires=layer.wires), interface="torch")
    def circuit():
        layer()
        return qml.state()
    return circuit()


class RealAmplitudeTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)

    def test_legacy_default_and_initial_skip(self):
        for structure in ("full", "linear"):
            for skip_initial in (False, True):
                layer = RealAmplitude(3, structure, skip_initial)
                legacy = TwoLocalAnsatz(3, ["ry"], "cx", structure, skip_initial)
                legacy.set_parameters(layer.params)
                actual = operations(layer)
                expected = operations(legacy, range(3))
                self.assertEqual(layer.params.shape, legacy.params.shape)
                self.assertEqual(len(actual), len(expected))
                for a, b in zip(actual, expected):
                    self.assertEqual((a.name, a.wires), (b.name, b.wires))
                    for x, y in zip(a.data, b.data):
                        torch.testing.assert_close(x, y)

    def test_parameter_counts_and_gate_counts(self):
        for wires in (1, 2, 4):
            for reps in (0, 1, 3):
                for skip_initial in (False, True):
                    for skip_final in (False, True):
                        with self.subTest(wires=wires, reps=reps,
                                          initial=skip_initial, final=skip_final):
                            layer = RealAmplitude(wires, only_entangle=skip_initial,
                                                  reps=reps, skip_final_rotation_layer=skip_final)
                            rotations = max(0, reps + 1 - skip_initial - skip_final)
                            self.assertEqual(layer.parameter_shape, (rotations, wires, 1))
                            self.assertEqual(layer.num_params, rotations * wires)
                            self.assertIsNone(layer.input_dim)
                            ops = operations(layer)
                            self.assertEqual(sum(op.name == "RY" for op in ops), layer.num_params)
                            self.assertEqual(sum(op.name == "CNOT" for op in ops),
                                             reps * wires * (wires-1) // 2)
                            self.assertTrue(torch.isfinite(state(layer)).all())

    def test_entanglement_order(self):
        expected = {
            "full": [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)],
            "linear": [(0, 1), (1, 2), (2, 3)],
            "reverse_linear": [(2, 3), (1, 2), (0, 1)],
            "circular": [(3, 0), (0, 1), (1, 2), (2, 3)],
            "pairwise": [(0, 1), (2, 3), (1, 2)],
            "sca": [(3, 0), (0, 1), (1, 2), (2, 3)],
        }
        for structure, pairs in expected.items():
            layer = RealAmplitude(4, structure, reps=2)
            actual = [tuple(op.wires) for op in operations(layer) if op.name == "CNOT"]
            second = [(3, 2), (0, 3), (1, 0), (2, 1)] if structure == "sca" else pairs
            self.assertEqual(actual, pairs + second)

    def test_sca_shift_cycles(self):
        layer = RealAmplitude(4, "sca", reps=5)
        expected = [
            [(3, 0), (0, 1), (1, 2), (2, 3)],
            [(3, 2), (0, 3), (1, 0), (2, 1)],
            [(1, 2), (2, 3), (3, 0), (0, 1)],
            [(1, 0), (2, 1), (3, 2), (0, 3)],
            [(3, 0), (0, 1), (1, 2), (2, 3)],
        ]
        actual = [tuple(op.wires) for op in operations(layer) if op.name == "CNOT"]
        self.assertEqual(actual, [pair for pairs in expected for pair in pairs])

    def test_one_and_two_qubit_structures(self):
        for structure in STRUCTURES:
            for wires in (1, 2):
                layer = RealAmplitude(wires, structure, reps=3)
                entanglers = [op for op in operations(layer) if op.name == "CNOT"]
                self.assertEqual(len(entanglers), 0 if wires == 1 else 3)
                self.assertTrue(all(len(set(op.wires)) == 2 for op in entanglers))
                self.assertTrue(torch.isfinite(state(layer)).all())

    def test_rotation_parameter_order(self):
        layer = RealAmplitude(3, reps=2)
        layer.set_parameters(torch.arange(9, dtype=torch.float32).reshape(3, 3, 1))
        angles = [op.data[0] for op in operations(layer) if op.name == "RY"]
        torch.testing.assert_close(torch.stack(angles), torch.arange(9, dtype=torch.float32))

    def test_real_normalized_state(self):
        for structure in STRUCTURES:
            output = state(RealAmplitude(4, structure, reps=3).double())
            torch.testing.assert_close(output.imag, torch.zeros_like(output.imag))
            torch.testing.assert_close(output.abs().square().sum(), torch.tensor(1., dtype=torch.float64))

    def test_matches_explicit_reference_circuit(self):
        layer = RealAmplitude(3, "reverse_linear", reps=2).double()

        @qml.qnode(qml.device("default.qubit", wires=3), interface="torch")
        def reference(params):
            for l in range(3):
                for i in range(3):
                    qml.RY(params[l, i, 0], wires=i)
                if l < 2:
                    qml.CNOT(wires=[1, 2])
                    qml.CNOT(wires=[0, 1])
            return qml.state()

        torch.testing.assert_close(state(layer), reference(layer.params))

    def test_vqe_gradient_and_optimizer(self):
        hamiltonian = -qml.Z(0) @ qml.Z(1) - 0.7 * (qml.X(0) + qml.X(1))
        model = VQE(hamiltonian.pauli_rep, RealAmplitude(2, reps=2), wires=2).double()
        params = model.ansatz.params
        original = params.detach().clone()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        energy = model().sum()
        energy.backward()
        gradient = params.grad.clone()
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertGreater(gradient.norm().item(), 0)

        numerical = torch.empty_like(params)
        epsilon = 1e-6
        for i in range(params.numel()):
            values = []
            for sign in (1, -1):
                perturbed = original.clone()
                perturbed.reshape(-1)[i] += sign * epsilon
                model.ansatz.set_parameters(perturbed)
                values.append(model().sum().detach())
            numerical.reshape(-1)[i] = (values[0] - values[1]) / (2 * epsilon)
        model.ansatz.set_parameters(original)
        torch.testing.assert_close(gradient, numerical, atol=1e-6, rtol=1e-5)
        optimizer.step()
        self.assertIs(model.ansatz.params, params)
        self.assertLess(model().sum().item(), energy.item())

    def test_batched_qml_and_upstream_gradient(self):
        class QuantumModel(QLayer):
            def __init__(self):
                super(QuantumModel, self).__init__(wires=2)
                self.ansatz = RealAmplitude(2, reps=2)

            def inner_gates(self, x):
                for i in range(2):
                    qml.RY(x[:, i], wires=i)
                self.ansatz()

            def measurement(self):
                return [qml.expval(qml.Z(i)) for i in range(2)]

        model = QuantumModel().double()
        x = torch.randn(4, 2, dtype=torch.float64, requires_grad=True)
        output = model(x)
        self.assertEqual(output.shape, (4, 2))
        output.square().sum().backward()
        for gradient in (x.grad, model.ansatz.params.grad):
            self.assertTrue(torch.isfinite(gradient).all())
            self.assertGreater(gradient.norm().item(), 0)

    def test_state_dict_and_conversion(self):
        layer = RealAmplitude(3, reps=2).double()
        restored = RealAmplitude(3, reps=2).double()
        restored.load_state_dict(layer.state_dict())
        self.assertEqual(list(dict(layer.named_parameters())), ["params"])
        self.assertEqual(layer.params.dtype, torch.float64)
        self.assertEqual(layer.parameter_shape, (3, 3, 1))
        torch.testing.assert_close(state(layer), state(restored))
        layer.requires_grad_(False)
        self.assertFalse(layer.params.requires_grad)
        layer.requires_grad_(True)
        self.assertTrue(layer.params.requires_grad)

    def test_wire_labels_and_invalid_wires(self):
        layer = RealAmplitude(3, "linear")
        ops = operations(layer, ["a", "c", "b"])
        self.assertEqual([tuple(op.wires) for op in ops if op.name == "CNOT"],
                         [("a", "c"), ("c", "b")])
        for wires in ([], [0, 1], [0, 1, 2, 3]):
            with self.subTest(wires=wires):
                with self.assertRaises(ValueError):
                    operations(layer, wires)
        with self.assertRaises(qml.wires.WireError):
            operations(layer, [0, 0, 1])

    def test_invalid_configuration(self):
        for value in (True, 1.5, "2", None):
            with self.assertRaises(TypeError):
                RealAmplitude(2, reps=value)
        with self.assertRaises(ValueError):
            RealAmplitude(2, reps=-1)
        with self.assertRaises(ValueError):
            RealAmplitude(2, "unknown")
        with self.assertRaises(TypeError):
            RealAmplitude(2, [(0, 1)])
        for option in ("only_entangle", "skip_final_rotation_layer"):
            with self.assertRaises(TypeError):
                RealAmplitude(2, **{option: 1})
        for wires in (True, 1.5, 0, -1):
            with self.assertRaises(AssertionError):
                RealAmplitude(wires)

    @unittest.skipUnless(importlib.util.find_spec("qiskit"), "Optional Qiskit dependency is not installed")
    def test_qiskit_statevector_parity(self):
        from qiskit.circuit.library import real_amplitudes
        from qiskit.quantum_info import Statevector

        for wires in (1, 2, 3, 4):
            for structure in STRUCTURES:
                for reps in (0, 1, 2, 5):
                    for skip_final in (False, True):
                        with self.subTest(wires=wires, structure=structure, reps=reps, final=skip_final):
                            layer = RealAmplitude(wires, structure, reps=reps,
                                                  skip_final_rotation_layer=skip_final).double()
                            reference = real_amplitudes(wires, entanglement=structure, reps=reps,
                                                        skip_final_rotation_layer=skip_final)
                            reference = reference.assign_parameters(layer.params.detach().reshape(-1).tolist())
                            # Qiskit and PennyLane use opposite statevector bit orders.
                            expected = Statevector.from_instruction(reference.reverse_bits()).data.copy()
                            torch.testing.assert_close(state(layer), torch.as_tensor(expected))


if __name__ == "__main__":
    unittest.main()

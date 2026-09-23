"""Regression tests for independent QLayer copies, including previously executed models."""
import copy
import unittest

import pennylane as qml
import torch
from torch import nn

from torchpenny import QLayer, QSubLayer


class Rotation(QSubLayer):
    def init_weights(self):
        return torch.full((1,), 0.3)

    def forward(self, x=None, wires=None):
        qml.RY(self.params[0], wires=0 if wires is None else wires[0])


class CopyCircuit(QLayer):
    def __init__(self):
        super(CopyCircuit, self).__init__(1, qnode_kwargs={"diff_method": "backprop"})
        self.rotation = Rotation(1)
        self.observable = "Z"

    def inner_gates(self, x):
        qml.RY(x[:, 0], wires=0)
        self.rotation()

    def measurement(self):
        return qml.expval(qml.Z(0) if self.observable == "Z" else qml.X(0))


class NoInputCircuit(CopyCircuit):
    def inner_gates(self):
        self.rotation()


class QLayerCopyTests(unittest.TestCase):
    def setUp(self):
        self.x = torch.tensor([[0.2], [0.7]], dtype=torch.float64)

    def assert_independent(self, original, cloned):
        self.assertIsNot(cloned.rotation.params, original.rotation.params)
        self.assertNotEqual(cloned.rotation.params.data_ptr(), original.rotation.params.data_ptr())
        self.assertIsNot(cloned.qnode, original.qnode)
        self.assertIsNot(cloned.q_device, original.q_device)
        self.assertIs(cloned.qnode.device, cloned.q_device)
        self.assertIs(cloned.qnode.func.__self__, cloned)

    def test_copy_before_and_after_execution(self):
        for phase in ("fresh", "forward", "backward"):
            with self.subTest(phase=phase):
                original = CopyCircuit().double()
                if phase != "fresh":
                    output = original(self.x.clone().requires_grad_())
                    if phase == "backward":
                        output.sum().backward()
                original.zero_grad(set_to_none=True)
                cloned = copy.deepcopy(original)
                self.assert_independent(original, cloned)
                expected = original(self.x).detach()
                torch.testing.assert_close(cloned(self.x), expected)
                cloned.rotation.set_parameters(torch.tensor([0.9]))
                torch.testing.assert_close(cloned(self.x), torch.cos(self.x + cloned.rotation.params))
                self.assertFalse(torch.allclose(cloned(self.x), expected))
                torch.testing.assert_close(original(self.x), expected)
                cloned(self.x).sum().backward()
                self.assertIsNone(original.rotation.params.grad)
                self.assertGreater(cloned.rotation.params.grad.abs().item(), 0)

    def test_copy_optimizer_and_input_gradients(self):
        original = CopyCircuit().double()
        original_params = original.rotation.params.detach().clone()
        cloned = copy.deepcopy(original)
        x = self.x.clone().requires_grad_()
        optimizer = torch.optim.SGD(cloned.parameters(), lr=0.1)
        cloned(x).sum().backward()
        expected_gradient = -torch.sin(self.x + cloned.rotation.params.detach())
        torch.testing.assert_close(x.grad, expected_gradient)
        torch.testing.assert_close(cloned.rotation.params.grad, expected_gradient.sum().reshape(1))
        optimizer.step()
        torch.testing.assert_close(original.rotation.params, original_params)
        self.assertIsNone(original.rotation.params.grad)
        self.assertFalse(torch.equal(cloned.rotation.params, original_params))

    def test_measurement_uses_copied_instance(self):
        original = CopyCircuit().double()
        cloned = copy.deepcopy(original)
        cloned.observable = "X"
        torch.testing.assert_close(cloned(self.x), torch.sin(self.x + cloned.rotation.params))
        torch.testing.assert_close(original(self.x), torch.cos(self.x + original.rotation.params))

    def test_copy_after_device_replacement(self):
        original = CopyCircuit().double()
        original.update_qdevice("default.qubit", qnode_kwargs={"diff_method": "parameter-shift"})
        cloned = copy.deepcopy(original)
        self.assert_independent(original, cloned)
        self.assertEqual(cloned.qnode.diff_method, "parameter-shift")
        self.assertEqual(cloned._qnode_kwargs, original._qnode_kwargs)
        self.assertIsNot(cloned._qnode_kwargs, original._qnode_kwargs)
        cloned.rotation.set_parameters(torch.tensor([0.9]))
        cloned(self.x).sum().backward()
        self.assertIsNone(original.rotation.params.grad)
        self.assertGreater(cloned.rotation.params.grad.abs().item(), 0)

    def test_device_replacement_on_copy_stays_independent(self):
        original = CopyCircuit().double()
        original_device, original_qnode = original.q_device, original.qnode
        cloned = copy.deepcopy(original)
        cloned.update_qdevice(qml.device("default.qubit", wires=1),
                              qnode_kwargs={"diff_method": "parameter-shift"})
        self.assert_independent(original, cloned)
        self.assertIs(original.q_device, original_device)
        self.assertIs(original.qnode, original_qnode)
        self.assertEqual(original.qnode.diff_method, "backprop")
        cloned(self.x).sum().backward()
        self.assertIsNone(original.rotation.params.grad)
        self.assertGreater(cloned.rotation.params.grad.abs().item(), 0)

    def test_copy_inside_parent_preserves_internal_sharing(self):
        block = CopyCircuit().double()
        original = nn.ModuleList([block, block])
        original[0](self.x)
        cloned = copy.deepcopy(original)
        self.assertIs(cloned[0], cloned[1])
        self.assert_independent(original[0], cloned[0])
        (cloned[0](self.x) + cloned[1](self.x)).sum().backward()
        self.assertIsNone(block.rotation.params.grad)
        self.assertGreater(cloned[0].rotation.params.grad.abs().item(), 0)

    def test_no_input_and_repeated_copies(self):
        original = NoInputCircuit().double()
        original()
        first = copy.deepcopy(original)
        first()
        second = copy.deepcopy(first)
        self.assert_independent(original, first)
        self.assert_independent(first, second)
        second.rotation.set_parameters(torch.tensor([0.9]))
        second().sum().backward()
        self.assertIsNone(original.rotation.params.grad)
        self.assertIsNone(first.rotation.params.grad)
        torch.testing.assert_close(second(), torch.cos(second.rotation.params))

    def test_finite_shot_copy_keeps_device_and_tape_consistent(self):
        original = NoInputCircuit().double()
        original.update_qdevice("default.qubit", q_device_kwargs={"shots": (10, 20), "seed": 7},
                                qnode_kwargs={"diff_method": "parameter-shift"})
        original()
        cloned = copy.deepcopy(original)
        self.assert_independent(original, cloned)
        self.assertEqual(cloned.q_device.shots, original.q_device.shots)
        cloned.rotation.set_parameters(torch.zeros_like(cloned.rotation.params))
        partitions = cloned()
        self.assertEqual(len(partitions), 2)
        for output in partitions:
            torch.testing.assert_close(output, torch.ones(1, dtype=torch.float64))
        self.assertIsNot(cloned.qnode._tape, original.qnode._tape)


if __name__ == "__main__":
    unittest.main()

"""Run from the project root: python -m unittest discover -s test -v."""
import copy
import unittest

import pennylane as qml
import torch

from torchpenny import QLayer, QSubLayer


class TemplateModel(QLayer):
    def __init__(self):
        super(TemplateModel, self).__init__(wires=2)
        self.feature = QSubLayer.from_pennylane(
            qml.AngleEmbedding, wires=2, param_received=True,
            parameter_shape=(2,), operation_kwargs={"rotation": "Y"})
        self.ansatz = QSubLayer.from_pennylane(
            qml.StronglyEntanglingLayers, wires=2, shape_kwargs={"n_layers": 2})

    def inner_gates(self, x):
        self.feature(x)
        self.ansatz()

    def measurement(self):
        return [qml.expval(qml.Z(i)) for i in range(self.wires)]


class PennyLaneSubLayerTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)

    def test_shape_and_owned_parameter_registration(self):
        for template, shape in ((qml.StronglyEntanglingLayers, (2, 3, 3)),
                                (qml.BasicEntanglerLayers, (2, 3))):
            layer = QSubLayer.from_pennylane(template, 3, shape_kwargs={"n_layers": 2})
            self.assertIsInstance(layer, QSubLayer)
            self.assertEqual(layer.template_shape, shape)
            self.assertEqual(layer.parameter_shape, shape)
            self.assertEqual(layer.num_params, layer.params.numel())
            self.assertIsNone(layer.input_dim)
            self.assertIs(dict(layer.named_parameters())["params"], layer.params)
            self.assertEqual(list(layer.state_dict()), ["params"])
            self.assertIn(template.__name__, repr(layer))

    def test_direct_template_state_and_gradient_parity(self):
        for template in (qml.StronglyEntanglingLayers, qml.BasicEntanglerLayers):
            layer = QSubLayer.from_pennylane(template, 3, shape_kwargs={"n_layers": 2}).double()
            reference_params = layer.params.detach().clone().requires_grad_()

            @qml.qnode(qml.device("default.qubit", wires=3), interface="torch")
            def wrapped():
                layer()
                return qml.state()

            @qml.qnode(qml.device("default.qubit", wires=3), interface="torch")
            def reference(params):
                template(params, wires=range(3))
                return qml.state()

            actual, expected = wrapped(), reference(reference_params)
            torch.testing.assert_close(actual, expected)
            actual.real.sum().backward()
            expected.real.sum().backward()
            torch.testing.assert_close(layer.params.grad, reference_params.grad)
            self.assertGreater(layer.params.grad.norm().item(), 0)

    def test_external_shape_batch_and_gradient_parity(self):
        for template in (qml.StronglyEntanglingLayers, qml.BasicEntanglerLayers):
            layer = QSubLayer.from_pennylane(
                template, 3, param_received=True, shape_kwargs={"n_layers": 2})
            self.assertEqual(layer.input_dim, (1, layer.num_params))
            self.assertEqual(list(layer.parameters()), [])
            self.assertEqual(dict(layer.state_dict()), {})
            self.assertIsNone(layer.params)
            layer.double()
            x = torch.rand(4, layer.num_params, dtype=torch.float64, requires_grad=True)
            reference_x = x.detach().clone().requires_grad_()

            @qml.qnode(qml.device("default.qubit", wires=3), interface="torch")
            def wrapped(values):
                layer(values)
                return qml.expval(qml.Z(0))

            @qml.qnode(qml.device("default.qubit", wires=3), interface="torch")
            def reference(values):
                template(values.reshape((4,) + layer.template_shape), wires=range(3))
                return qml.expval(qml.Z(0))

            actual, expected = wrapped(x), reference(reference_x)
            self.assertEqual(actual.shape, (4,))
            torch.testing.assert_close(actual, expected)
            actual.sum().backward()
            expected.sum().backward()
            torch.testing.assert_close(x.grad, reference_x.grad)
            self.assertTrue(torch.isfinite(x.grad).all())
            self.assertGreater(x.grad.norm().item(), 0)

    def test_qlayer_integration_and_optimizer(self):
        model = TemplateModel().double()
        self.assertEqual(model.input_features, {"feature": (2,)})
        self.assertEqual(set(model._qsublayers), {"feature", "ansatz"})
        x = torch.rand(2, 3, 2, dtype=torch.float64, requires_grad=True)
        original = model.ansatz.params.detach().clone()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        output = model(x)
        self.assertEqual(output.shape, (2, 3, 2))
        output.square().sum().backward()
        for gradient in (x.grad, model.ansatz.params.grad):
            self.assertTrue(torch.isfinite(gradient).all())
            self.assertGreater(gradient.norm().item(), 0)
        optimizer.step()
        self.assertFalse(torch.equal(model.ansatz.params, original))
        self.assertEqual(model(torch.rand(2, dtype=torch.float64)).shape, (2,))

    def test_state_dict_dtype_and_parameter_identity(self):
        model = TemplateModel().double()
        params = model.ansatz.params
        model.ansatz.set_parameters(torch.full_like(params, 0.2))
        self.assertIs(model.ansatz.params, params)
        self.assertEqual(params.dtype, torch.float64)
        restored = TemplateModel().double()
        restored.load_state_dict(copy.deepcopy(model.state_dict()))
        x = torch.rand(3, 2, dtype=torch.float64)
        torch.testing.assert_close(model(x), restored(x))
        self.assertIsNot(restored.ansatz.params, params)
        self.assertEqual(set(model.state_dict()), {"ansatz.params"})

    def test_fresh_operations_and_custom_wire_labels(self):
        with qml.queuing.AnnotatedQueue() as queue:
            layer = QSubLayer.from_pennylane(
                qml.BasicEntanglerLayers, 2, shape_kwargs={"n_layers": 1})
        self.assertEqual(len(queue), 0)
        with qml.queuing.AnnotatedQueue() as queue:
            first = layer(wires=["a", "b"])
            second = layer(wires=["b", "a"])
        self.assertEqual(len(queue), 2)
        self.assertIsNot(first, second)
        self.assertEqual(tuple(first.wires), ("a", "b"))
        self.assertEqual(tuple(second.wires), ("b", "a"))
        self.assertIs(first.data[0], layer.params)

    def test_options_are_separate_and_copied(self):
        shape_options = {"n_layers": 2}
        operation_options = {"rotation": qml.RY}
        layer = QSubLayer.from_pennylane(
            qml.BasicEntanglerLayers, 3, shape_kwargs=shape_options,
            operation_kwargs=operation_options)
        self.assertEqual(shape_options, {"n_layers": 2})
        self.assertEqual(operation_options, {"rotation": qml.RY})
        shape_options["n_layers"] = 4
        operation_options["rotation"] = qml.RX
        self.assertEqual(layer.template_shape, (2, 3))
        self.assertIs(layer().hyperparameters["rotation"], qml.RY)

    def test_explicit_shape_and_scalar_operation(self):
        layer = QSubLayer.from_pennylane(qml.StronglyEntanglingLayers, 2,
                                        parameter_shape=(1, 2, 3))
        self.assertEqual(layer().data[0].shape, (1, 2, 3))
        scalar = QSubLayer.from_pennylane(qml.RX, 1, parameter_shape=())
        self.assertEqual(scalar.params.shape, ())
        self.assertEqual(scalar.num_params, 1)
        self.assertEqual(scalar().data[0].shape, ())

    def test_invalid_shapes_and_shape_options(self):
        for shape in ((0, 2), (-1, 2), (True, 2), (2.0, 2), ((2, 3), (2, 3))):
            with self.subTest(shape=shape), self.assertRaises(ValueError):
                QSubLayer.from_pennylane(qml.BasicEntanglerLayers, 2, parameter_shape=shape)
        with self.assertRaises(TypeError):
            QSubLayer.from_pennylane(qml.BasicEntanglerLayers, 2, parameter_shape=2)
        for options in ({"shape_kwargs": {"n_layers": 2, "n_wires": 3}},
                        {"parameter_shape": (2, 2), "shape_kwargs": {"n_layers": 2}}):
            with self.assertRaises(ValueError):
                QSubLayer.from_pennylane(qml.BasicEntanglerLayers, 2, **options)
        with self.assertRaises(TypeError):
            QSubLayer.from_pennylane(qml.BasicEntanglerLayers, 2)
        with self.assertRaises(ValueError):
            QSubLayer.from_pennylane(qml.AngleEmbedding, 2)

    def test_invalid_templates_and_constructor_options(self):
        for template in (qml.RX(0.2, wires=0), lambda: None, qml.Rot, qml.QAOAEmbedding):
            with self.subTest(template=template), self.assertRaises(TypeError):
                QSubLayer.from_pennylane(template, 2, parameter_shape=(2,))
        for options in ({"wires": [0, 1]}, {"weights": torch.rand(1, 2)},
                        {"n_layers": 1}, {"unknown": True}):
            with self.assertRaises(TypeError):
                QSubLayer.from_pennylane(qml.BasicEntanglerLayers, 2,
                                        parameter_shape=(1, 2), operation_kwargs=options)

    def test_invalid_wires_and_external_inputs(self):
        for wires, error in ((True, TypeError), (2.0, TypeError), (0, ValueError), (-1, ValueError)):
            with self.assertRaises(error):
                QSubLayer.from_pennylane(qml.AngleEmbedding, wires, parameter_shape=(2,))
        layer = QSubLayer.from_pennylane(qml.AngleEmbedding, 2, True, parameter_shape=(2,))
        for x, error in ((None, TypeError), ([1, 2], TypeError), (torch.rand(2), ValueError),
                         (torch.rand(3, 3), ValueError), (torch.rand(2, 3, 2), ValueError)):
            with self.assertRaises(error):
                layer(x)
        with self.assertRaises(ValueError):
            layer(torch.rand(3, 2), wires=[0])
        with self.assertRaises(qml.wires.WireError):
            layer(torch.rand(3, 2), wires=[0, 0])


if __name__ == "__main__":
    unittest.main()

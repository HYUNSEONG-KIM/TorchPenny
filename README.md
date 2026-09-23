# TorchPenny


TorchPenny is a [Pennylane](https://docs.pennylane.ai) based quantum algorithm framework.
It is a light wrapper for user to design and to integrate quantum circuit into PyTorch module and computation workflow.
The object of the framework is to provide a simple way to manage the reusable quantum circuit design with Pennylane to be used in PyTorch.

## Workflow in Pennylanea and PyTorch integration.

- [PyTorch Interface of Pennylane](https://docs.pennylane.ai/en/stable/introduction/interfaces/torch.html)

Pennylane is well combined in PyTorch ecosystem, still it's design is far from the usual workflow of PyTorch framework.
However, when you want to reuse the quantum circuit module, you should manage the input and output parameters for each circuit function.
When you forgot the input dimension and output dimension, it will disrupt the training and workflow.
However, in PyTorch all the module has their on information of in/out tensor dimension and even though you forgot the information, you can directly check the module information from the object.

The object of this project is to get the benefit of PyTorch module design when we design a QML module, providing more familiar interface to PyTorch developers and keeping the design process of Pennylane.


## Basic usage

The next code is an example to generate a quantum circuit layer.

Each `QLayer` must return measurements of one type only. A single `qml.probs()` or multiple `qml.expval()` results can be used, but mixing measurement types raises `ValueError` before device execution. Sample-only circuits remain supported.

`QLayer` defaults to `interface="torch"`, including circuits without external inputs. An interface explicitly supplied in `qnode_kwargs` is not overwritten; the layer's output processing expects Torch tensors.

`update_qdevice()` preserves the configured QNode settings, including `interface` and `diff_method`. Pass `qnode_kwargs` to override individual settings. When switching from analytic backpropagation to finite shots, explicitly select a compatible method such as `"best"` or `"parameter-shift"`.

```.{py}
from torchpenny.module import QLayer
from torchpenny.lib.features import ZZfeatureMap
from torchpenny.lib.ansatz import TwoLocalAnsatz

class QuantumLayer(QLayer): # Quantum Layer definition.
    def __init__(self, *args, **kwargs):
        super(QuantumLayer, self).__init__(*args, **kwargs)

        # Internal layer 
        self.zzf_input = ZZfeatureMap(self.wires, param_received=True)
        self.two_local1 = TwoLocalAnsatz(self.wires, ['rx', 'rz'], 'cz', "linear")
        self.two_local2 = TwoLocalAnsatz(self.wires, ['rx', 'rz'], 'cz', "circular", except_initial=True)

    def input_encoding(self, x): # Define the input data flow.
        self.zzf_input(x, wires=range(self.wires))

    def inner_gates(self): # Internal quantum gates
        qml.Barrier(wires=range(self.wires))
        self.two_local1(wires=range(self.wires))
        qml.Barrier(wires=range(self.wires))
        self.two_local2(wires=range(self.wires))

    def measurement(self): # Measurement setting
        return [qml.expval(qml.PauliZ(i)) for i in range(self.wires)]

qlayer = QuantumLayer(wires= 8, q_device = "default.qubit", qnode_kwargs={"diff_method": "backprop"})

batch_dim = 4
input_dim = qlayer.zzf_input.num_params # 15 It is determined by parameterized gates.

data = torch.rand((batch_dim, input_dim))

qlayer(data)
'''
tensor([[ 0.3023,  0.2878,  0.0862,  0.1278,  0.0516,  0.0289,  0.2345,  0.1783],
        [ 0.0641,  0.4929,  0.1542,  0.2590,  0.3207, -0.0395,  0.0663,  0.0376],
        [ 0.0848,  0.2385,  0.1502,  0.2679,  0.0890, -0.0244,  0.1650,  0.1028],
        [ 0.3325,  0.4807,  0.1975,  0.3510,  0.5032,  0.2242,  0.1110,  0.1923]],
       dtype=torch.float64, grad_fn=<ViewBackward0>)
'''
qml.draw_mpl(qlayer.qnode)(data)
```

The defined `QuantumLayer` could be attached to common PyTorch module workflow.
Like, 

```
from torch import nn

module = nn.Sequential([
    nn.Linear(20, 15)
    qlayer # Its inoutput dimension is (15, 8)
    nn.Linear(8, 20)

])
```
There are two modules `QLayer` and `QSubLayer`.

### QLayer

`QLyaer` is a main torch module managing Pennylane `Device` and `Qnode` objects.
Original Pennylane library provides `TorchLayer`. 
With `TorchLayer` you have to manage all parameters and input data at once on single function.

1. `inner_gates`: Internal circuit structure definition function.
2. `measurement`: Measurement part of the quantum circuit.

These inner methods are equivalent to the next circuit.

```
@qnode(dev)
def circuit(x):
    input_encoding(x)
    inner_gates()
    return measurement()
```

### QSubLayer

`QSubLayer` module is an alternative api of `Operation` or `Template` class of Pennylane framework. In Pennylane, `Template` class also provides a block layer of quantum gates. However, their parameters are given by user and external source. Users have to manage the parameter dimension data for each `Template` in the circuit. 

Meanwhile, in Pytorch, all `Module` class manages its own parameter automatically. The input data and inherited weight and bias are seperated. To construct complicated circuit and program, this automatic management has many benefits not only for rapid prototyping but also for maintainance.

Thus, `QSubLayer` provides automatic parameter management routine based on `Moudle` class. 


## Examples

See `docs` notebooks.

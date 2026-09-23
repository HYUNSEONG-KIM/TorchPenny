from typing import Union, Tuple, LiteralString, Optional
from numbers import Number
from abc import ABC, abstractmethod
from inspect import signature
import warnings

import torch
from torch import Tensor
from torch.nn import Module, Parameter

from pennylane import (device as get_q_device, QNode)
from pennylane.devices import Device as QDevice
from pennylane.measurements import ExpectationMP

from pennylane.operation import Operation, AnyWires
from pennylane.wires import Wires

class QSubLayer(Module, ABC):
    """`QSubLayer` is an thin wrapper of `Operation` or `Template` class of Pennylane framework with `Module` class in PyTorch.
    This layer can either have its own mutable parameter or get the parameter as an argument.

    Args:
        Module (_type_): _description_

    Raises:
        NotImplementedError: _description_
        NotImplementedError: _description_
        NotImplementedError: _description_

    Returns:
        _type_: _description_
    """
    _input_module: dict[str, Optional[Tensor]]

    def __init__(self, wires:int, param_received=False):
        super(QSubLayer, self).__init__()
        assert type(wires) is int, "The given wire valye must be integer type."
        assert wires > 0, "The given wires must be positive integer larger than 0."
        
        self.wires = int(wires)
        self.param_received = bool(param_received)
        params = self._init_weights()
        if self.param_received:
            self.register_parameter("params", None)
        else:
            self.params:Tensor = params
    def __repr__(self):
        st = super().__repr__()
        if self.input_dim is not None:
            class_name = self._get_name()
            instance_name = class_name+f"[{self.input_dim[1]}]"
            return st.replace(class_name, instance_name)
        else:
            return st
    def __setattr__(self, name:str, value):
        if name == "params" and isinstance(value, Tensor):
            if not self.param_received:
                value = Parameter(value) 
        super().__setattr__(name, value)
    def set_parameters(self, params):
        if self.param_received:
            raise RuntimeError("External parameter module does not own parameters. Pass them as input.")
        if not isinstance(params, Tensor):
            raise TypeError("`params` must be a Tensor.")
        if params.shape != self.params.shape:
            raise ValueError("Dimensions are not matched.")
        with torch.no_grad():
            self.params.copy_(params)
    @property
    def parameter_shape(self):
        return self._parameter_shape
    @property
    def input_dim(self):
        return self.parameter_shape if self.param_received else None
    @property
    def num_params(self):
        size = 1
        for d in self.parameter_shape:
            size *= d
        return size
    def _init_weights(self):
        params = self.init_weights()
        if self.param_received and (params.dim() != 2 or params.shape[0] != 1):
            raise ValueError("External parameter shape must be (1, N). Reshape it in init_weights().")
        self._parameter_shape = params.shape
        return params
    def get_params(self, x:Union[Tensor, None]=None):
        if self.param_received:
            assert x is not None, "Received param module must get argument."
            assert x.shape[-1] == self.num_params, "The given data was not matched with the layer input."
            params = x
        else:
            params = self.params
        return params
    @classmethod
    def from_pennylane(cls, template, wires:int, param_received=False, *,
                       shape_kwargs=None, parameter_shape=None, operation_kwargs=None):
        """Wrap a PennyLane Operation class taking one tensor followed by wires.

        Returns a QSubLayer instance, not an instance of the calling subclass.
        If parameter_shape is omitted, template.shape(**shape_kwargs) is used;
        n_wires is filled from wires when the shape function declares it.
        Shape options and operation constructor options are kept separate.

        Owned parameters use the native template shape and torch.rand initialization.
        External inputs use (B, N), where N is the product of the template shape.
        Forward restores (B, *template_shape); broadcasting and differentiability
        must be supported by the template itself. Multiple parameter tensors and
        already constructed operations are not supported.

        Example:
            QSubLayer.from_pennylane(qml.StronglyEntanglingLayers, wires=4,
                                    shape_kwargs={"n_layers": 2})
        """
        return _PennyLaneSubLayer(template, wires, param_received,
                                 shape_kwargs=shape_kwargs, parameter_shape=parameter_shape,
                                 operation_kwargs=operation_kwargs)
    @abstractmethod
    def init_weights(self):
        # init_method
        # define the shape of the params
        # You don't have to set Parameter or Tensor it will be set automatically considering param_recevied argument. 
        raise NotImplementedError
    @abstractmethod
    def forward(self, x:Union[Tensor, None], wires:Tuple[int,...]):
        raise NotImplementedError
    def to_operation(self):
        """Generate the custom Pennylane gates of the defined model.
        """
        raise NotImplementedError

    #def to_operation(self):
    #    class QSubLayer2Op(Operation):
    #        num_wires = AnyWires
    #        def __init__(self, wires:int, id=None):
    #            all_wires = Wires()
    #        
    #        @staticmethod
    #        def compute_decomposition(*params, wires = None, **hyperparameters):
    #            return 


class _PennyLaneSubLayer(QSubLayer):
    def __init__(self, template, wires:int, param_received=False, *,
                 shape_kwargs=None, parameter_shape=None, operation_kwargs=None):
        if type(wires) is not int:
            raise TypeError("`wires` must be an integer.")
        if wires <= 0:
            raise ValueError("`wires` must be positive.")
        if not isinstance(template, type) or not issubclass(template, Operation):
            raise TypeError("`template` must be a PennyLane Operation class, not an instance.")

        shape_options = dict(shape_kwargs or {})
        operation_options = dict(operation_kwargs or {})
        constructor = signature(template)
        arguments = list(constructor.parameters)
        if len(arguments) < 2 or arguments[1] != "wires":
            raise TypeError("The template must take one parameter tensor followed by `wires`.")
        # Also reject duplicated parameters, wires, unknown options and missing options.
        constructor.bind(None, wires=range(wires), **operation_options)

        if parameter_shape is None:
            shape_function = getattr(template, "shape", None)
            if not callable(shape_function):
                raise ValueError("This template has no shape() method. Supply `parameter_shape`.")
            if "n_wires" in signature(shape_function).parameters:
                if "n_wires" in shape_options and shape_options["n_wires"] != wires:
                    raise ValueError("`shape_kwargs['n_wires']` must match `wires`.")
                shape_options["n_wires"] = wires
            parameter_shape = shape_function(**shape_options)
        elif shape_options:
            raise ValueError("Use either `parameter_shape` or `shape_kwargs`, not both.")
        if not isinstance(parameter_shape, (tuple, list, torch.Size)):
            raise TypeError("`parameter_shape` must be a tuple or list of dimensions.")
        if any(type(d) is not int or d <= 0 for d in parameter_shape):
            raise ValueError("Template shape dimensions must be positive integers.")

        self.template = template
        self.template_shape = tuple(parameter_shape)
        self.operation_kwargs = operation_options
        super(_PennyLaneSubLayer, self).__init__(wires, param_received)

    def init_weights(self):
        if self.param_received:
            size = 1
            for d in self.template_shape:
                size *= d
            return torch.empty(1, size)
        return torch.rand(self.template_shape)

    def forward(self, x=None, wires=None):
        wires = Wires(range(self.wires) if wires is None else wires)
        if len(wires) != self.wires:
            raise ValueError("Applied wires must match the defined qubit dimension.")
        if self.param_received:
            if not isinstance(x, Tensor):
                raise TypeError("External parameters must be a Tensor.")
            if x.dim() != 2 or x.shape[1] != self.num_params:
                raise ValueError("External parameters must have shape (B, N), with N = num_params.")
        params = self.get_params(x)
        if self.param_received:
            params = params.reshape((params.shape[0],) + self.template_shape)
        operation = self.template(params, wires=wires, **self.operation_kwargs)
        if operation.num_params != 1:
            raise ValueError("Only templates with one parameter tensor are supported.")
        return operation

    def extra_repr(self):
        return f"template={self.template.__name__}, wires={self.wires}, template_shape={self.template_shape}"


class QLayer(Module):
    """
        Default Pytorch Module interface class for quantum circuit. 
        Alternative version of `TorchLayer` in Pennylane. 
        This layer can contain multiple `QSubLayer` however, all the `Qnode` should be only one.

        Args:
            wires (int): Number of qubit wires
            q_device (Union[str, QDevice], optional): Pennylane quantum device. It could be a string or `Device` object of Pennylane. Defaults to 'default.qubit'.
            q_device_kwargs (dict, optional): Device arguments. Defaults to {}.
            qnode_kwargs (dict, optional): Qnode initial setting arguments. Defaults to {"interface": "torch"}.

        Returns:
            _type_: _description_
            
        In children class, by redefining `inner_gates` and `measurment` attributes, you can define torch interfaced Pennylane circuit.
        Example:
            ```
            class QuantumLayer(QLayer):
                def __init__(self, *args, **kwargs):
                    super(QuantumLayer, self).__init__(*args, **kwargs)
                def inner_gates(self, x=None):
                    # You can define Pennylane circuit inside here.
                    for w in range(self.wires):
                        qml.Hadamard(wires=w)
                        qml.PauliX(wires=w)
                    qml.CNOT([0, 1])
                def measurement(self): # measurement part.
                    return qml.probs()
            
            qlayer = QuantumLayer(wires=4, qnode_kwargs={"diff_method":"backprops"})
            ```
            The above `QuantumLayer` definition is identical to next code.
            
            ```
            import pennylane as qml
            
            wires = 4
            dev = qml.dev("default.qubit", wires = wires)
            @qml.qnode(dev, interface="torch", **qnode_kwargs)
            def circuit(x=None):
                for w in range(wires):
                    qml.Hadamard(wires=w)
                    qml.PauliX(wires=w)
                wml.CNOT([0, 1])
                return qml.probs()
            ```
    """
    def __init__(self, 
                 wires:int, 
                 q_device:Union[str, QDevice] = 'default.qubit', 
                 q_device_kwargs = None,
                 qnode_kwargs = None
                 ):
        
        super(QLayer, self).__init__()

        if type(wires) is not int:
            raise TypeError("`wires` must be an integer.")
        if wires <=0:
            raise ValueError("`wires` value must be a positive integer.")

        q_device_options = dict(q_device_kwargs or {}) # To prevent the modification of outside dictionary.
        qnode_options = dict(qnode_kwargs or {})
        # Quantum device verification
        if isinstance(q_device, str):
            if "wires" in q_device_options.keys():
                if q_device_options["wires"] != wires and isinstance(q_device, str):
                    warnings.warn("Number of wires in QLayer arg and q_device are different.", Warning)
            q_device_options["wires"] = wires
            self.q_device = get_q_device(q_device, **q_device_options)
        elif isinstance(q_device, QDevice):
            if wires != len(q_device.wires):
                warnings.warn("Number of wires in QLayer arg and q_device are different.", Warning)
                wires = len(q_device.wires)
            self.q_device = q_device
        else:
            raise TypeError("`q_device` should be `QDevice` instance or `str` of Pennylane device name.")
            
        self.wires = wires 

        if "interface" not in qnode_options.keys():
            qnode_options["interface"] = "torch"
        self.qnode = QNode(self._circuit, device=self.q_device, **qnode_options)
        self._qnode_kwargs = qnode_options.copy()

    def _circuit(self, x:Optional[Union[Tensor, Tuple[Tensor]]]=None):
        # A bound method follows the copied module; a local closure keeps the original self.
        self._inner_gates(x)
        return self._measurement()

    def __repr__(self):
        st = super().__repr__()
        class_name = self._get_name()
        st = st.replace(class_name, class_name+f"[Qnode, wires={self.wires}]")
        return st
        
    def _register_qsublayer(self, name, qsublayer:QSubLayer):
        """Add QSubLayer to QLayer.

        Args:
            name (_type_): _description_
            qsublayer (QSubLayer): _description_
        """
        self.add_module(name, qsublayer)
    @property
    def _qsublayers(self):
        return {k:v for k, v in self._modules.items() if isinstance(v, QSubLayer)}
    @property
    def input_features(self):
        input_f_dict = {}
        for k, v in self._qsublayers.items():
            if v.input_dim is not None:
                input_f_dict[k] = v.input_dim[1:]
        return input_f_dict
    
    def _inner_gates(self, x):
        arg_len = len(signature(self.inner_gates).parameters)
        if arg_len == 0:
            return self.inner_gates()
        else:
            return self.inner_gates(x)

    def _measurement(self):
        measurements = self.measurement()
        pending = [measurements]
        measurement_types = set()
        while pending:
            value = pending.pop()
            if isinstance(value, (tuple, list)):
                pending.extend(value)
            else:
                measurement_types.add(type(value))
        if len(measurement_types) > 1:
            raise ValueError("QLayer must return only one measurement type. Do not mix probs, expval, or sample.")
        return measurements
        
    def update_qdevice(self, q_device:Union[str, QDevice], q_device_kwargs:dict=None, qnode_kwargs:dict=None):
        q_device_options = dict(q_device_kwargs or {})
        qnode_options = self._qnode_kwargs.copy()
        qnode_options.update(qnode_kwargs or {})
        if isinstance(q_device, QDevice):
            new_device = q_device
        elif isinstance(q_device, str):
            q_device_options.setdefault("wires", self.wires)
            new_device = get_q_device(q_device, **q_device_options)
        else:
            raise TypeError("`q_device` should be `QDevice` instance or `str` of Pennylane device name.")
        if new_device.wires is not None and len(new_device.wires) != self.wires:
            raise ValueError("The new device must have the same number of wires as the layer.")

        new_qnode = QNode(self._circuit, device=new_device, **qnode_options)
        # Replace the working objects only after both constructions succeed.
        self.q_device = new_device
        self.qnode = new_qnode
        self._qnode_kwargs = qnode_options.copy()
        
    @abstractmethod
    def inner_gates(self, x:Optional[Union[Tensor, Tuple[Tensor]]]=None):
        pass
    @abstractmethod
    def measurement(self):
        """Return measurements of one type only. Multiple expectation values are allowed.
        """
        raise NotImplementedError

    def _restore_shot_batch(self, vals, batch_dims=None):
        # Use the executed tape, so measurement() is not called a second time.
        tape = self.qnode._tape
        if tape.batch_size is None:
            batch_dims = None

        def restore_partition(value, shots):
            measurements = iter(tape.measurements)
            def restore(result):
                if isinstance(result, tuple):
                    return tuple(restore(v) for v in result)
                if isinstance(result, list):
                    return [restore(v) for v in result]
                measurement = next(measurements)
                if batch_dims is None:
                    if isinstance(result, Tensor) and isinstance(measurement, ExpectationMP):
                        return result.unsqueeze(-1)
                    return result
                if not isinstance(result, Tensor):
                    raise TypeError("Batched finite-shot results must contain tensors.")
                shape = measurement.shape(shots=shots, num_device_wires=self.wires)
                if isinstance(measurement, ExpectationMP):
                    shape = (1,)
                return result.reshape(tuple(batch_dims) + tuple(shape))
            return restore(value)

        if tape.shots.has_partitioned_shots:
            shots = [s.shots for s in tape.shots.shot_vector for _ in range(s.copies)]
            return tuple(restore_partition(v, s) for v, s in zip(vals, shots))
        return restore_partition(vals, tape.shots.total_shots)
    
    def forward(self, x:Optional[Tensor]=None):
        if isinstance(x, Tensor):
            if x.dim() == 0:
                raise TypeError("Input tensor must have a feature dimension.")
            # Assume that feature dimension is the last dimension.
            *batch_dims, feat = x.shape
            # Flatten leading dims
            x_flat = x.reshape(-1, feat)
        elif x is None:
            x_flat = None
            batch_dims = ()
        elif isinstance(x, (tuple, list)):
            if not x or not all(isinstance(v, Tensor) for v in x):
                raise TypeError("Multiple inputs must be a nonempty tuple or list of tensors.")
            x_flat = x # In this case, the user have to verify and manage the batch cases.
            batch_dims = ()
        else:
            raise TypeError("Input must be a Tensor, a tuple/list of tensors, or None.")
            
        vals = self.qnode(x_flat)
        if self.q_device.shots.total_shots is not None: # Sampling
            if isinstance(x, Tensor):
                return self._restore_shot_batch(vals, batch_dims)
            return self._restore_shot_batch(vals, () if x is None else None)
        
        # vals: list of (batch_flat,) tensors
        if isinstance(vals, Tensor):
            out_flat = vals
            if isinstance(self.qnode._tape.measurements[0], ExpectationMP):
                out_flat = out_flat.unsqueeze(-1)
            # Scalar Tensor
            if out_flat.dim()==0:
                return out_flat
        else:# Iterable
            #Batched data
            # Pennylane return column bached tensors.
            axis = 0 if vals[0].dim() ==0 else 1
            out_flat = torch.stack(vals, axis=axis)
        # Reshape back to batch dims
        return out_flat.reshape(*batch_dims, out_flat.shape[-1])
    
    #def draw(self, backend="")
    
        

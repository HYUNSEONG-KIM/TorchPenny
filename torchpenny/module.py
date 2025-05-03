from typing import Union, Tuple, LiteralString, Optional, Iterable
from numbers import Number
from abc import abstractmethod
from inspect import signature

import torch
from torch import Tensor
from torch.nn import Module, Parameter

from pennylane import (device as get_q_device, QNode)
from pennylane.devices import Device as QDevice

from pennylane.operation import Operation, AnyWires
from pennylane.wires import Wires

class QSubLayer(Module):
    #  구현한 Sublayer는 자체 Parameter를 가질 수도 있고, 외부에서 Parameter를 줄 수도 있어야 한다.
    _input_module: dict[str, Optional[Tensor]]

    def __init__(self, wires:int, param_received=False):
        super(QSubLayer, self).__init__()
        assert wires > 0 and isinstance(wires, Number), "The given wires must be positive integer larger than 0."
        self.wires = int(wires)
        self.param_received = bool(param_received)
        self.params:Tensor = self._init_weights()
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
        assert params.shape == self.params.shape, "Dimensions are not matched."
        self.params.data=(params)
    @property
    def input_dim(self):
        return self.params.shape if self.param_received else None
    @property
    def num_params(self):
        size = 1
        for d in self._num_params:
            size *= d
        return size
    def _init_weights(self):
        params = self.init_weights()
        self._num_params = params.shape
        return params
    @abstractmethod
    def init_weights(self):
        # init_method
        # define the shape of the params
        # You don't have to set Parameter or Tensor it will be set automatically considering param_recevied argument. 
        raise NotImplementedError
    @abstractmethod
    def forward(self, x:Union[Tensor, None], wires:Tuple[int,...]):
        raise NotImplementedError
    

    @abstractmethod
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


class QLayer(Module):
    """
        Default Pytorch Module interface class for quantum circuit. 
        Alternative version of `TorchLayer` in Pennylane. 

        Args:
            wires (int): Number of qubit wires
            q_device (Union[str, QDevice], optional): Pennylane quantum device. It could be a string or `Device` object of Pennylane. Defaults to 'default.qubit'.
            q_device_kwargs (dict, optional): Device arguments. Defaults to {}.
            qnode_kwargs (dict, optional): Qnode initial setting arguments. Defaults to {"diff_method": "parameter-shift"}.

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
    _qsublayers:{Optional[QSubLayer]}

    def __init__(self, 
                 wires:int, 
                 q_device:Union[str, QDevice] = 'default.qubit', 
                 q_device_kwargs = {},
                 qnode_kwargs = {"diff_method": "parameter-shift"}
                 ):
        
        super(QLayer, self).__init__()
        super().__setattr__("_qsublayers", {})

        self.wires = wires
        q_device_kwargs["wires"] = self.wires
        if isinstance(q_device, QDevice):
            self.q_device = q_device
        elif isinstance(q_device, str):
            self.q_device = get_q_device(q_device, **q_device_kwargs)
    
        def _circuit(x:Optional[Union[Tensor, Tuple[Tensor]]] =None):
            self._inner_gates(x)
            return self.measurement()
        
        if "interface" not in qnode_kwargs.keys():
            qnode_kwargs["interface"] = "torch"
        self.qnode = QNode(_circuit, device=self.q_device, **qnode_kwargs)
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
        if hasattr(self, name) and name not in self._qsublayers:
            raise KeyError(f"attribute '{name}' already exists.")
        self._qsublayers[name] = qsublayer

    def __setattr__(self, name, value):
        if isinstance(value, QSubLayer):
            self._register_qsublayer(name, value)
        super().__setattr__(name, value)
    def __getattr__(self, name):
        if name in self._qsublayers.keys():
            return self._qsublayers[name]
        return super().__getattr__(name)
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
    def update_qdevice(self, q_device:Union[str, QDevice], q_device_kwargs:dict={}, qnode_kwargs:dict={}):
        if isinstance(q_device, QDevice):
            self.q_device = q_device
        elif isinstance(q_device, str):
            self.q_device = get_q_device(q_device, **q_device_kwargs)
        def _circuit(x:Optional[Union[Tensor, Tuple[Tensor]]] =None):
            self._inner_gates(x)
            return self.measurement()
        self.qnode = QNode(_circuit, device=self.q_device, **qnode_kwargs)
    @abstractmethod
    def inner_gates(self, x:Optional[Union[Tensor, Tuple[Tensor]]]=None):
        pass
    @abstractmethod
    def measurement(self):
        raise NotImplementedError
    
    def forward(self, x:Optional[Tensor]=None):
        if isinstance(x, Tensor):
            # Assume that feature dimension is the last dimension.
            *batch_dims, feat = x.shape
            # Flatten leading dims
            x_flat = x.reshape(-1, feat)
        elif x is None:
            x_flat = None
            batch_dims = ()
        elif isinstance(x, Iterable):
            x_flat = x # In this case, the user have to verify and manage the batch cases.
            batch_dims = ()
            
        vals = self.qnode(x_flat)
        if self.q_device.shots.total_shots is not None: # Sampling
            return vals
        
        # vals: list of (batch_flat,) tensors
        if isinstance(vals, Tensor):
            out_flat = vals
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
    
        

from typing import Union, Tuple, LiteralString
from numbers import Number
from abc import abstractmethod

import torch
from torch import Tensor
from torch.nn import Module

from pennylane import (device as get_q_device, QNode)
from pennylane.devices import Device as QDevice

from pennylane.operation import Operation, AnyWires
from pennylane.wires import Wires

class QSubLayer(Module):
    def __init__(self, wires:int, param_received=False):
        super(QSubLayer, self).__init__()
        assert wires > 0 and isinstance(wires, Number), "The given wires must be positive integer larger than 0."
        self.wires = int(wires)
        self.param_received = bool(param_received)
        self.params:Tensor = self._init_weights()
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
    def __init__(self, 
                 wires:int, 
                 q_device:Union[str, QDevice], 
                 q_device_kwargs = {},
                 qnode_kwargs = {"diff_method": "parameter-shift"}
                 ):
        super(QLayer, self).__init__()

        self.wires = wires
        if isinstance(q_device, QDevice):
            self.q_device = q_device
        elif isinstance(q_device, str):
            self.q_device = get_q_device(q_device, **q_device_kwargs)
    
        def _circuit(x):
            self.input_encoding(x)
            self.inner_gates()
            return self.measurement()
        
        if "interface" not in qnode_kwargs.keys():
            qnode_kwargs["interface"] = "torch"
        self.qnode = QNode(_circuit, device=self.q_device, **qnode_kwargs)

    @abstractmethod
    def input_encoding(self, x:Tensor):
        raise NotImplementedError
    def inner_gates(self):
        pass
    @abstractmethod
    def measurement(self):
        raise NotImplementedError
    
    def forward(self, x:Tensor):
        # Assume that feature dimension is the last dimension.
        *batch_dims, feat = x.shape
        # Flatten leading dims
        x_flat = x.reshape(-1, feat)
        vals = self.qnode(x_flat)
        # vals: list of (batch_flat,) tensors
        if isinstance(vals, Tensor):
            out_flat = vals
        else:
            out_flat = torch.stack(vals, axis=1)
        # Reshape back to batch dims
        return out_flat.reshape(*batch_dims, out_flat.shape[1])
    
        

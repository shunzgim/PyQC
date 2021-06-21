from types import CodeType
from pyqc.gates import *
import numpy as np


class Var:
    def __init__(self, value=0, grad=0, coefficient=1, id=-1):
        self.value = value
        self.grad = grad
        self.coef = coefficient
        self.id = id

    @property
    def angle(self):
        return self.value*self.coef

    @property
    def name(self):
        return '%s:var(%s)'%(self.id,round(self.value,3))


def var(value, coefficient):
    res = None
    if not isinstance(value, np.ndarray):
        raise ValueError('type of var must be numpy.ndarray are not ',type(value))
    else:
        res = []
        for i in range(len(value)):
            res.append(Var(value=value[i],coefficient=coefficient[i],id=i))
    return res


class Optimizer:
    def __init__(self, v, lr):
        self.var = v
        self.lr = lr

    def update(self):
        for v in self.var:
            v.value += self.lr*v.grad



class  VariationalQuantumFramework:
    """
    """
    def __init__(self, backend, vqc, cost_Hamiltonian):
        """
        """
        self.backend = backend
        self.vqc = vqc
        self.cost_h = cost_Hamiltonian
        
    def loss(self):
        """
        """
        loss = 0
        for _,value in self.cost_h.items():
            target = list(value.keys())
            loss += self.backend.getExpectation(target)
        return loss

    def _update_helper(self, gate, target, control):
        if isinstance(gate, OneGate):
            to_array = gate.matrix.H.A.reshape(4)
            self.backend.fullAlib.applyOneGate(to_array,target[0],0)
            self.backend.fullAlib.applyOneGate(to_array,target[0],1)
        elif gate.name=="CNOT" or gate.name=="CNOT.dag" or gate.name=="CZ" or gate.name=="CZ.dag":
            to_array = gate.cmatrix.A.reshape(4)
            self.backend.fullAlib.applyControlOneGate(to_array, target[0], control[0], 0)
            self.backend.fullAlib.applyControlOneGate(to_array, target[0], control[0], 1)
        elif gate.name == "Toffili":
            h_array = H.matrix.A.reshape(4)
            s_array = S.matrix.A.reshape(4)
            s_d_array = SDag.matrix.A.reshape(4)
            x_array = X.matrix.A.reshape(4)
            self.backend.fullAlib.applyOneGate(h_array,target[0], 0)
            self.backend.fullAlib.applyControlOneGate(s_d_array, target[0], control[0], 0)
            self.backend.fullAlib.applyControlOneGate(x_array, control[1], control[0], 0)
            self.backend.fullAlib.applyControlOneGate(s_array, target[0], control[1], 0)
            self.backend.fullAlib.applyControlOneGate(x_array, control[1], control[0], 0)
            self.backend.fullAlib.applyControlOneGate(s_d_array, target[0], control[1], 0)
            self.backend.fullAlib.applyOneGate(h_array,target[0], 0)
            self.backend.fullAlib.applyOneGate(h_array,target[0], 1)
            self.backend.fullAlib.applyControlOneGate(s_d_array, target[0], control[0], 1)
            self.backend.fullAlib.applyControlOneGate(x_array, control[1], control[0], 1)
            self.backend.fullAlib.applyControlOneGate(s_array, target[0], control[1], 1)
            self.backend.fullAlib.applyControlOneGate(x_array, control[1], control[0], 1)
            self.backend.fullAlib.applyControlOneGate(s_d_array, target[0], control[1], 1)
            self.backend.fullAlib.applyOneGate(h_array,target[0], 1)
        elif gate.name == "Toffili.dag":
            h_array = H.matrix.A.reshape(4)
            s_array = S.matrix.A.reshape(4)
            s_d_array = SDag.matrix.A.reshape(4)
            x_array = X.matrix.A.reshape(4)
            self.backend.fullAlib.applyOneGate(h_array,target[0], 0)
            self.backend.fullAlib.applyControlOneGate(s_array, target[0], control[1], 0)
            self.backend.fullAlib.applyControlOneGate(x_array, control[1], control[0], 0)
            self.backend.fullAlib.applyControlOneGate(s_d_array, target[0], control[1], 0)
            self.backend.fullAlib.applyControlOneGate(x_array, control[1], control[0], 0)
            self.backend.fullAlib.applyControlOneGate(s_array, target[0], control[0], 0)
            self.backend.fullAlib.applyOneGate(h_array,target[0], 0)
            self.backend.fullAlib.applyOneGate(h_array,target[0], 1)
            self.backend.fullAlib.applyControlOneGate(s_array, target[0], control[1], 1)
            self.backend.fullAlib.applyControlOneGate(x_array, control[1], control[0], 1)
            self.backend.fullAlib.applyControlOneGate(s_d_array, target[0], control[1], 1)
            self.backend.fullAlib.applyControlOneGate(x_array, control[1], control[0], 1)
            self.backend.fullAlib.applyControlOneGate(s_array, target[0], control[0], 1)
            self.backend.fullAlib.applyOneGate(h_array,target[0], 1)
        elif gate.name=="Swap" or gate.name=="Swap.dag":
            to_array = X.matrix.A.reshape(4)
            self.backend.fullAlib.applyControlOneGate(to_array, target[1], target[0], 0)
            self.backend.fullAlib.applyControlOneGate(to_array, target[0], target[1], 0)
            self.backend.fullAlib.applyControlOneGate(to_array, target[1], target[0], 0)
            self.backend.fullAlib.applyControlOneGate(to_array, target[1], target[0], 1)
            self.backend.fullAlib.applyControlOneGate(to_array, target[0], target[1], 1)
            self.backend.fullAlib.applyControlOneGate(to_array, target[1], target[0], 1)
        else:
            print(gate.name)
            raise RuntimeError('error')

    def backward(self):
        """
        """
        size = []
        target = []
        to_array = np.array([])
        for _,value in self.cost_h.items():
            for id,gate in value.items():
                target.append(id)
                to_array = np.append(gate.matrix.A.reshape(4), to_array)
            size.append(len(value))
        self.backend.fullAlib.grad_helper_init(to_array,target,size,len(size))            #初始化|psi_b>
        layer_id = list(self.vqc.layer_info.keys())
        layer_id.reverse()
        for layer in layer_id:
            item = self.vqc.layer_info[layer]
            theta, op_dict, gate_ops = item['theta'], item['op_dict'], item['gate_ops']
            size = []
            target = []
            to_array = np.array([])
            for _,value in op_dict.items():
                for id,gate in value.items():
                    target.append(id)
                    to_array = np.append(gate.matrix.A.reshape(4), to_array)
                size.append(len(value))
            grad = self.backend.fullAlib.grad_helper(to_array,target,size,len(size))     #得到梯度cost_Hamiltonian关于theta_j的梯度
            gate_ops.reverse()
            for gid in gate_ops:
                gate, target, control = self.vqc.qgate_list[gid]
                self._update_helper(gate, target, control)                               #更新|psi_a>、|psi_b>
            theta.grad = grad
        

    



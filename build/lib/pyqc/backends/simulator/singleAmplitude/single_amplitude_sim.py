from .import copy
from .import numpy as np

from .tensor_network import TensorNetwork, Node
from pyqc.gates import *


class SingleAmplitudeSimulator:
    """
    """
    def __init__(self):
        self.tn = TensorNetwork()

    def _getSuperoperator(self, gate, target, control):
        tensor, qubits = [], []
        tensor = gate.superop
        if isinstance(gate, OneGate):
            qubits = target
        elif gate.name=="CNOT" or gate.name=="CNOT.dag" or gate.name=="CZ" or gate.name=="CZ.dag":
            qubits = control + target
        elif gate.name == "Toffili":
            pass
        elif gate.name == "Toffili.dag":
            pass
        elif gate.name=="Swap" or gate.name=="Swap.dag":
            pass
        else:
            print(gate.name)
            raise RuntimeError('error')
        return tensor,qubits
    
    def exec_circ(self, circ):
        """
        """
        self.circ = circ
        self.all_nodes = []
        self.indice = 0
        self.taxis_indice = []
        #默认初始化|0><0|，转化为[1,0,0,0]
        for i in range(self.circ.qubit_nums):
            node = Node(np.array([1, 0, 0, 0],dtype=np.complex64))
            node.edges[0] = self.indice
            self.taxis_indice.append(self.indice)
            self.indice += 1
            self.all_nodes.append(node)
        #处理量子线路
        for tup in self.circ.qgate_list:
            gate, target, control = tup
            tensor, qubits = self._getSuperoperator(gate, target, control)
            node = Node(tensor)
            ndim = int(tensor.ndim / 2)
            if ndim!=1 and ndim!=2: 
                raise RuntimeError('tensor.ndim',2*ndim)
            for i in range(ndim):
                node.edges[i] = self.taxis_indice[qubits[i]]
                node.edges[i+len(qubits)] = self.indice
                self.taxis_indice[qubits[i]] = self.indice
                self.indice += 1
            self.all_nodes.append(node)
        #填充
        all_nodes = copy.deepcopy(self.all_nodes)
        for i in range(self.circ.qubit_nums):
            node = Node(np.array([1, 0, 0, 1],dtype=np.complex64))
            node.edges[0] = self.taxis_indice[i]
            all_nodes.append(node)
        self.tn.find_path(all_nodes)

    def getOneProbabilityFromBinstring(self, binstring):
        all_nodes = copy.deepcopy(self.all_nodes)
        taxis_indice = copy.deepcopy(self.taxis_indice)
        #计算binstring振幅
        assert(len(binstring)==self.circ.qubit_nums)
        for qid, s in enumerate(binstring):
            if s=='0':
                node = Node(np.array([1, 0, 0, 0],dtype=np.complex64))
            else:
                node = Node(np.array([0, 0, 0, 1],dtype=np.complex64))
            node.edges[0] = taxis_indice[qid]
            all_nodes.append(node)
        return self.tn.contract(all_nodes)

    def getExpectation(self, target_list):
        all_nodes = copy.deepcopy(self.all_nodes)
        taxis_indice = copy.deepcopy(self.taxis_indice)
        #处理期望超算子
        for i in range(self.circ.qubit_nums):
            if i in target_list:
                j = target_list.index(i)
                node = Node(np.array([1, 0, 0, -1],dtype=np.complex64))
                node.edges[0] = taxis_indice[i]
            else:
                node = Node(np.array([1, 0, 0, 1],dtype=np.complex64))
                node.edges[0] = taxis_indice[i]
            all_nodes.append(node)
        return self.tn.contract(all_nodes)
        
        

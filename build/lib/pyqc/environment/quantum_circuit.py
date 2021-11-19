from pyqc.gates import *


class QuantumCircuit:
    """
    """
    def __init__(self, qubit_nums):
        self.qubit_nums = qubit_nums
        self.pos = 0
        self.qgate_list = []
        self.measure_list = []
        
    def insert(self, gate, target, control=None, in_layer=None):
        """
        """
        if isinstance(target, list): # [Qubit,Qubit,...]
            target = [qt.id for qt in target]
        else: # Qubit
            target = [target.id] 
        if not control is None:
            if isinstance(control, list): # [Qubit,Qubit,...]
                control =  [qc.id for qc in control]
            else:
                control = [control.id]
        if isinstance(gate, MeasureGate):
            self.measure_list.append(target[0])
        else:
            if not in_layer is None:
                self.layer_info[in_layer]['gate_ops'].append(self.pos)
            self.qgate_list.append((gate,target,control))
            self.pos += 1

    def show(self, f=None):
        """
        """
        for i,tup in enumerate(self.qgate_list):
            if f is None:
                print(i,'gate:',tup[0],'  target:',tup[1],'  control:',tup[2])
            else:
                print(i,'gate:',tup[0],'  target:',tup[1],'  control:',tup[2], file=f)


class VariationalQuantumCircuit(QuantumCircuit):
    """
    """
    def __init__(self, qubit_nums):
        self.layer_info = {}
        super(VariationalQuantumCircuit,self).__init__(qubit_nums)
        
    def create_layer(self, layer, theta, op):
        """
        U_j(theta_j) = exp(-i*theta_j*op)
        op:Hamiltonian
        """
        self.layer_info[layer] = {'theta':theta, 'op_dict':op, 'gate_ops':[]}
    
    def show_layer_info(self):
        for layer,value in self.layer_info.items():
            print('layer_id:',layer)
            print('------theta:',value['theta'].name)
            print('------op_dict:')
            for idx, item in enumerate(value['op_dict'].items()):
                print('----------Hamiltonian_sub:',idx,[(id,gate.name) for id,gate in item[1].items()])
            print('------gate_pos:',value['gate_ops'])

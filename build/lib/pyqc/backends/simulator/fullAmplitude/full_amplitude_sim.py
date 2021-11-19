from pyqc.gates import *

from pyqc.backends.simulator.libs import fullAlib


class FullAmplitudeSimulator:
    """
    """
    def __init__(self):
        self.fullASim = fullAlib.fullASim()

    def exec_circ(self, circ):
        self.fullASim.flush(circ.qubit_nums)
        for tup in circ.qgate_list:
            gate, target, control = tup
            if isinstance(gate, OneGate):
                to_array = gate.matrix.A.reshape(4)
                self.fullASim.applyOneGate(to_array,target[0], 0)
            elif gate.name=="CNOT" or gate.name=="CNOT.dag" or gate.name=="CZ" or gate.name=="CZ.dag":
                to_array = gate.cmatrix.A.reshape(4)
                self.fullASim.applyControlOneGate(to_array, target[0], control[0], 0)
            elif isinstance(gate ,CR) or isinstance(gate, CRDag):
                to_array = gate.cmatrix.A.reshape(4)
                self.fullASim.applyControlOneGate(to_array, target[0], control[0], 0)
            elif isinstance(gate, CUOne):
                to_array = gate.cmatrix.A.reshape(4)
                self.fullASim.applyControlOneGate(to_array, target[0], control[0], 0)
            elif gate.name == "Toffili":
                h_array = H.matrix.A.reshape(4)
                s_array = S.matrix.A.reshape(4)
                s_d_array = SDag.matrix.A.reshape(4)
                x_array = X.matrix.A.reshape(4)
                self.fullASim.applyOneGate(h_array,target[0], 0)
                self.fullASim.applyControlOneGate(s_array, target[0], control[1], 0)
                self.fullASim.applyControlOneGate(x_array, control[1], control[0], 0)
                self.fullASim.applyControlOneGate(s_d_array, target[0], control[1], 0)
                self.fullASim.applyControlOneGate(x_array, control[1], control[0], 0)
                self.fullASim.applyControlOneGate(s_array, target[0], control[0], 0)
                self.fullASim.applyOneGate(h_array,target[0], 0)
            elif gate.name == "Toffili.dag":
                h_array = H.matrix.A.reshape(4)
                s_array = S.matrix.A.reshape(4)
                s_d_array = SDag.matrix.A.reshape(4)
                x_array = X.matrix.A.reshape(4)
                self.fullASim.applyOneGate(h_array,target[0], 0)
                self.fullASim.applyControlOneGate(s_d_array, target[0], control[0], 0)
                self.fullASim.applyControlOneGate(x_array, control[1], control[0], 0)
                self.fullASim.applyControlOneGate(s_array, target[0], control[1], 0)
                self.fullASim.applyControlOneGate(x_array, control[1], control[0], 0)
                self.fullASim.applyControlOneGate(s_d_array, target[0], control[1], 0)
                self.fullASim.applyOneGate(h_array,target[0], 0)
            elif gate.name=="Swap" or gate.name=="Swap.dag":
                to_array = X.matrix.A.reshape(4)
                self.fullASim.applyControlOneGate(to_array, target[1], target[0], 0)
                self.fullASim.applyControlOneGate(to_array, target[0], target[1], 0)
                self.fullASim.applyControlOneGate(to_array, target[1], target[0], 0)
            elif gate.name == "CMExp":
                self.fullASim.applyConstantModExp(gate.a, gate.N, len(control))
            else:
                print(gate.name)
                raise RuntimeError('error')

    def getOneAmplitudeFromBinstring(self,binstring):
        return self.fullASim.getOneAmplitudeFromBinstring(binstring)
        
    def getExpectation(self, target):
        return self.fullASim.getExpectation(target, len(target))

    def getMeasureResultHandle(self, target):
        return self.fullASim.getMeasureResultHandle(len(target))



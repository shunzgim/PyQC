import numpy as np
from pyqc.gates import *
from pyqc.types import *
from pyqc.backends.simulator import *
from .quantum_circuit import QuantumCircuit


class simulatorType:
    def __init__(self):
        self._SINGLE_AMPLITUDE = True
        self._FULL_AMPLITUDE = False
simType = simulatorType()


class Environment:
    """
    """
    def __init__(self, backend_type, show=True):
        """
        """
        import psutil
        def get_log2n(available):
            n = 1 
            while 1<<n < available: 
                n += 1
            m = n-1
            while 3*(1<<m) > available: # 留出备份的空间
                m -= 1
            return m
        if show:
            print('##################################################################################')
            print('可用资源信息')
            print("物理核数:%s"%psutil.cpu_count(logical=False))
            print("逻辑核数:%s"%psutil.cpu_count())     
            print("虚拟内存(total):%sG"%round((float(psutil.virtual_memory().total)/1024/1024/1024), 2))
            print("虚拟内存(available):%sG"%round((float(psutil.virtual_memory().available)/1024/1024/1024), 2))
            print("交换内存(total):%sG"%round((float(psutil.swap_memory().total)/1024/1024/1024), 2))
        amplitude_byte = 8 #单精度振幅占用8Bytes
        available = float(psutil.virtual_memory().available) / amplitude_byte
        self.MAX_QUBIT_NUM = get_log2n(available)
        if backend_type is None or backend_type is False:
            self.backend = FullAmplitudeSimulator()
        else:
            self.backend = SingleAmplitudeSimulator()
        self.IS_SINGLE_AMPLITUDE = backend_type
        if show:
            print('##################################################################################')
            if self.IS_SINGLE_AMPLITUDE:
                print('注意：单振幅模拟初始化成功')
            else:
                print('注意：全振幅模拟初始化成功, 最大可申请%s个量子比特!              '%self.MAX_QUBIT_NUM)
            print('##################################################################################')
       
    def allocateQubits(self, n):
        """
        """
        if n>self.MAX_QUBIT_NUM:
            raise RuntimeError("要求量子比特数小于%s"%self.MAX_QUBIT_NUM)
        else:
            self.qubit_nums = n
            self.quantum_circuit = QuantumCircuit(n)
        return [Qubit(i) for i in range(n)]

    def exec(self, qcirc=None):
        """
        """
        if not qcirc is None:
            self.quantum_circuit = qcirc
        self.backend.exec_circ(self.quantum_circuit)

    def getAmplitude(self, binstring):
        """
        """
        if self.IS_SINGLE_AMPLITUDE:
            raise RuntimeError("只对全振幅提供")
        if len(binstring)!= self.qubit_nums:
            raise RuntimeError('01串长度%s与比特数%s不同!'%(len(binstring),self.qubit_nums))
        return self.backend.getOneAmplitudeFromBinstring(binstring)
    
    def getProbabilityBinstring(self, binstring):
        """
        """
        if self.IS_SINGLE_AMPLITUDE:
            res = self.backend.getOneProbabilityFromBinstring(binstring)
            return res.real
        else:
            res = self.backend.getOneAmplitudeFromBinstring(binstring)
            res = np.abs(res)**2
            return res

    def getState(self):
        """
        """
        if self.IS_SINGLE_AMPLITUDE:
            raise RuntimeError("只对全振幅提供")
        res_amplitude_dict = {}
        if self.qubit_nums > 16:
            raise RuntimeError('getState 只有量子比特数不大于16时被使用')
        else:
            for i in range(1<<self.qubit_nums):
                binstring = bin(i).split('b')[1].zfill(self.qubit_nums)
                res_amplitude_dict[binstring] = self.backend.getOneAmplitudeFromBinstring(binstring)
        return res_amplitude_dict
        
    def getProbability(self):
        """
        """
        res_proba_dict = {}
        if self.qubit_nums > 16:             
            raise RuntimeError('getProbability 只有量子比特数不大于16时被使用')
        else:
            for i in range(1<<self.qubit_nums):
                binstring = bin(i).split('b')[1].zfill(self.qubit_nums)
                proba = self.getProbabilityBinstring(binstring)
                key = ''
                for j in self.quantum_circuit.measure_list:
                    key += binstring[j]
                if key in res_proba_dict:
                    res_proba_dict[key] += proba
                else:
                    res_proba_dict[key] = proba
        return res_proba_dict
    
    def getMeasureResult(self, sample=100, show=False, name='result'):
        """
        """
        if self.qubit_nums > 16:                          
            raise RuntimeError('getMeasureResult 只有量子比特数不大于16时被使用')
        res_proba_dict = self.getProbability()
        keys = list(res_proba_dict.keys())
        values = list(res_proba_dict.values())
        for i in range(1,len(values)):
            values[i] += values[i-1]
        sample_res_dict = dict.fromkeys(keys,0.0)
        res_string = ''
        n = 0
        while n<sample:
            r = np.random.rand()
            for s,p in zip(keys, values):
                if r <= p:
                    res_string = s
                    if s in sample_res_dict:
                        sample_res_dict[s] += 1 / sample
                    break
            n += 1
        if show:
            if len(self.quantum_circuit.measure_list)>7:
                print('量子比特数大于6时不提供可视化')
            else:
                import matplotlib.pyplot as plt
                fig = plt.figure(figsize=(15,9))
                ax = fig.subplots()
                _proba = list(sample_res_dict.values())
                proba = list(res_proba_dict.values())
                xlocation = np.linspace(1, len(keys)*0.75, len(keys))
                b1 = ax.bar(xlocation, _proba, width = 0.2, label='Simulated probability')
                b2 = ax.bar(xlocation+0.2, proba, width = 0.2, label='Theoretical probability')
                if self.qubit_nums>=5:
                    rotation = 90
                else:
                    rotation = 0
                plt.xticks(xlocation+0.15,keys,fontsize=12 ,rotation = rotation)
                plt.title('sample frequency %d'%n)
                plt.legend()
                plt.grid()
                plt.xlabel('The measured results')
                plt.ylabel('probability')
                plt.savefig('%s.png'%name)
                plt.show()
        return res_string

    def getMeasureResultHandle(self, target):
        """
        """
        if self.IS_SINGLE_AMPLITUDE:
            raise RuntimeError("只对全振幅提供")
        if isinstance(target, Qubit):
            target = [target.id]
        else:
            target = [q.id for q in target]
        return self.backend.getMeasureResultHandle(target)

    def getExpectation(self, target):
        """
        """
        if isinstance(target, Qubit):
            target = [target.id]
        else:
            target = [q.id for q in target]
        return self.backend.getExpectation(target)
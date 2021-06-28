import sys
sys.path.append('../')

from pyqc import *
import numpy as np


if __name__ == '__main__':
    A = np.array([[1.5, 0.5],
                  [0.5, 1.5]])
    t = 2*np.pi
    r = 2**4
    ########## step1 ########### 申请后端模拟器
    qubit_nums = 4
    env = Environment(simType._FULL_AMPLITUDE) 
    q = env.allocateQubits(qubit_nums)
    ########## step2 ########### 设置量子线路
    qcirc = env.quantum_circuit
    """
    #相位估计
    qcirc.insert(H,target=q[3])
    qcirc.insert(CNOT, target=q[2], control=q[3])
    qcirc.insert(CNOT, target=q[1], control=q[2])
    qcirc.insert(X,target=q[2])
    qcirc.insert(Swap, target=[q[1],q[2]])
    #提取占比
    qcirc.insert(CUOne(Ry(Var(np.pi))), target=q[0], control=q[2])
    qcirc.insert(CUOne(Ry(Var(np.pi/3))), target=q[0], control=q[1])
    #逆相位估计
    qcirc.insert(Swap, target=[q[1],q[2]])
    qcirc.insert(X,target=q[2])
    qcirc.insert(CNOT, target=q[1], control=q[2])
    qcirc.insert(CNOT, target=q[2], control=q[3])
    qcirc.insert(H,target=q[3])
    """
    #相位估计
    qcirc.insert(H,target=q[1])
    qcirc.insert(H,target=q[2])
    qcirc.insert(CUOne(HSimOneGate(A,t,1/4)), target=q[3], control=q[2])
    qcirc.insert(CUOne(HSimOneGate(A,t,1/2)), target=q[3], control=q[1])
    qcirc.insert(Swap, target=[q[1],q[2]])
    qcirc.insert(H, target=q[2])
    qcirc.insert(CRDag(Var(0.5*np.pi)), target=q[1], control=q[2])
    qcirc.insert(H, target=q[1])
    #提取占比
    #qcirc.insert(Swap, target=[q[1],q[2]])
    #qcirc.insert(CUOne(Ry(Var(2*np.pi/r))), target=q[0], control=q[1])
    #qcirc.insert(CUOne(Ry(Var(np.pi/r))), target=q[0], control=q[2])
    qcirc.insert(CUOne(Ry(Var(np.pi))), target=q[0], control=q[2])
    qcirc.insert(CUOne(Ry(Var(np.pi/3))), target=q[0], control=q[1])
    #逆相位估计
    qcirc.insert(H, target=q[1])
    qcirc.insert(CR(Var(0.5*np.pi)), target=q[1], control=q[2])
    qcirc.insert(H, target=q[2])
    qcirc.insert(Swap, target=[q[1],q[2]])
    qcirc.insert(CUOne(HSimDagOneGate(A,t,1/2)), target=q[3], control=q[1])
    qcirc.insert(CUOne(HSimDagOneGate(A,t,1/4)), target=q[3], control=q[2])
    qcirc.insert(H,target=q[2])
    qcirc.insert(H,target=q[1])
    #"""
    for i in range(qubit_nums):
        qcirc.insert(Measure, target=q[i])
    ########## step3 ########### 可视化量子线路(可选)
    qcirc.show()
    ########## step4 ########### 执行线路
    env.exec()
    ########## step5 ########### 返回结果
    res_str = env.getMeasureResult(show=True,name='HHL_result')
    print('sample result:',res_str)
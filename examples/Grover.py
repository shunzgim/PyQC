from pyqc import *


if __name__ == "__main__":
    ########## step1 ########### 申请后端模拟器
    qubit_nums = 3
    env = Environment(backend_type=simType._FULL_AMPLITUDE)
    #env = Environment(simType._SINGLE_AMPLITUDE)
    q = env.allocateQubits(qubit_nums)
    ########## step2 ########### 设置量子线路
    qcirc = env.quantum_circuit
    qcirc.insert(X, target=q[2])
    for i in range(qubit_nums):
        qcirc.insert(H, target=q[i])
    ######################### oracle
    #qcirc.insert(X, target=q[0])
    qcirc.insert(Toffili, target=q[2], control=q[:2])
    #qcirc.insert(X, target=q[0])
    #########################
    for i in range(qubit_nums-1):
        qcirc.insert(H, target=q[i])
    for i in range(qubit_nums-1):
        qcirc.insert(X, target=q[i])
    qcirc.insert(H,target=q[1])
    qcirc.insert(CNOT,target=q[1], control=q[0])
    qcirc.insert(H,target=q[1])
    for i in range(qubit_nums-1):
        qcirc.insert(X, target=q[i])
    for i in range(qubit_nums-1):
        qcirc.insert(H, target=q[i])
    for i in range(qubit_nums-1):
        qcirc.insert(Measure, target=q[i])
    ########## step3 ########### 可视化量子线路(可选)
    qcirc.show()
    ########## step4 ########### 执行线路
    env.exec()
    ########## step5 ########### 返回结果
    res_str = env.getMeasureResult(show=True, name='Grover_result')
    print('sample result:',res_str)



from os import name
import sys
sys.path.append('../')

import time
import numpy as np
from pyqc import *


def create_qcirc(vqc, betas, gammas, p, qubits, hp, hb):
    for qid in qubits:
        vqc.insert(H, target=qid)
    for i in range(p):
        vqc.create_layer(layer=i*2, theta=gammas[i], op=hp)
        for j in hp:
            a, b = [int(t) for t in j]
            vqc.insert(gate=CNOT, target=qubits[int(b)], control=qubits[int(a)], in_layer=i*2)
            vqc.insert(gate=Rz(gammas[i]), target=qubits[int(b)], in_layer=i*2) 
            vqc.insert(gate=CNOT, target=qubits[int(b)], control=qubits[int(a)], in_layer=i*2)
        vqc.create_layer(layer=i*2+1, theta=betas[i], op=hb)
        for qid in qubits:
            vqc.insert(gate=Rx(betas[i]), target=qid, in_layer=i*2+1)
    for qid in qubits:
        vqc.insert(Measure, target=qid)

def create_hp_hb(n):
    hp = {}
    hb = {}
    for i in range(n):
        hb[i] = {i:X}
    deg_2 = list(range(n))
    for i in deg_2:
        hp[(i,(i+1)%n)] = {i:Z,(i+1)%n:Z}
    """
    while len(deg_2) > 0:
        if len(deg_2) > 4:
            a, b = np.random.choice(deg_2,2,replace=False)
            if (a,b) not in hp and (b,a) not in hp:    
                hp[(a,b)] = {a:Z,b:Z}
                deg_2.remove(a)
                deg_2.remove(b)
        else:
            a,b,c,d = deg_2
            if (a,b) not in hp and (b,a) not in hp and (c,d) not in hp and (d,c) not in hp:
                hp[(a,b)] = {a:Z,b:Z}
                hp[(c,d)] = {c:Z,d:Z}
                deg_2.remove(a)
                deg_2.remove(b)
                deg_2.remove(c)
                deg_2.remove(d)
            elif (a,c) not in hp and (c,a) not in hp and (b,d) not in hp and (d,b) not in hp:
                hp[(a,c)] = {a:Z,c:Z}
                hp[(b,d)] = {b:Z,d:Z}
                deg_2.remove(a)
                deg_2.remove(b)
                deg_2.remove(c)
                deg_2.remove(d)
            else:
                hp[(a,d)] = {a:Z,d:Z}
                hp[(b,c)] = {b:Z,c:Z}
                deg_2.remove(a)
                deg_2.remove(b)
                deg_2.remove(c)
                deg_2.remove(d)
    """
    ######################################可视化
    import networkx as nx
    import matplotlib.pyplot as plt
    g = nx.Graph()
    g.add_edges_from(hp.keys())
    fig, ax = plt.subplots()
    nx.draw(g, ax=ax, with_labels=True)
    fig.savefig('./network.png')
    #plt.show()
    return hp,hb


if __name__ == '__main__':
    #################### 申请模拟器
    qubit_nums = 4
    env = Environment(simType._FULL_AMPLITUDE)
    qubits = env.allocateQubits(qubit_nums)

    #################### 定义哈密顿量
    hp, hb = create_hp_hb(qubit_nums)
    
    #################### 定义变分参数
    p = 3
    lr = 0.003
    coefficient = [2]*2*p
    thetas = var(np.random.random(2*p)*2*np.pi, coefficient)
    
    #################### 创建变分量子线路并初始化变分量子框架
    vqc = VariationalQuantumCircuit(qubit_nums)
    create_qcirc(vqc, thetas[:p], thetas[p:], p, qubits, hp, hb)
    #vqc.show()
    #vqc.show_layer_info()
    vqf = VariationalQuantumFramework(env.backend, vqc, cost_Hamiltonian=hp)

    #################### 定义优化器
    opt = Optimizer(thetas, lr)
    
    #################### 训练变分量子线路
    iteration = 50
    loss_list = []
    beta_list = [[] for i in range(p)]
    gamma_list = [[] for i in range(p)]
    for i in range(iteration):
        env.exec(vqc)
        ###################### 计算损失
        start = time.time()
        loss = vqf.loss()
        loss_list.append(loss)
        end = time.time()
        print('iter:%s ------------------------------------ loss: %.5f'%(i,loss))
        print('loss runtime:%s'%time.strftime('%M:%S',time.localtime(end-start)))
        ###################### 计算梯度 
        start = time.time()
        vqf.backward()  
        end = time.time()
        print('backward runtime:%s'%time.strftime('%M:%S',time.localtime(end-start)))
        ##################################################################  训练过程打印
        b_, g_ = 0, 0
        for i, var in enumerate(thetas):
            if i < p:
                beta_list[i].append(var.grad)
                #print('       ----beta%s  grad: %.5f   value: %.5f'%(b_, var.grad, var.value))
                b_ += 1
            else:
                gamma_list[i-p].append(var.grad)
                #print('       ----gamma%s  grad: %.5f   value: %.5f'%(g_, var.grad, var.value))
                g_ += 1
        ###################### 更新参数
        opt.update()
        
    #################### 参数分析
    x = np.arange(len(loss_list))
    import matplotlib.pyplot as plt
    plt.figure(figsize=(15,9))
    plt.title('parameter analysis')
    ax1 = plt.subplot(2,1,1)
    ax1.plot(x,loss_list,label='loss')
    ax1.legend(loc=4)
    ax1.grid()
    ax2 = plt.subplot(2,1,2)
    b_, g_ = 0, 0
    for i, var in enumerate(thetas):
        if i < p:
            ax2.plot(x[1:],beta_list[i][1:],label='beta%s_grad'%b_)
            b_ += 1
        else:
            ax2.plot(x[1:],gamma_list[i-p][1:],label='gamma%s_grad'%g_)
            g_ += 1
    ax2.legend(loc=4)
    ax2.grid()
    ax2.set_xlabel('iterations')
    plt.savefig('parameter_analysis.png')
    plt.show()
    
    #################### 执行量子线路并打印测量结果
    env.exec(vqc)
    res_str = env.getMeasureResult(show=True,name='QAOA_result')
    print('sample result:',res_str)

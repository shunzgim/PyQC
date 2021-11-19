from pyqc import *
import math
import numpy as np


def log2(N):
    i = 0
    while(1<<i < N):
        i += 1
    return i

def continued_fraction_expansion(a,b):
    from fractions import Fraction
    c = Fraction(a,b)
    return c.denominator


def run_shor(a,N,cnt):
    ########## step1 ########### 申请后端模拟器
    n = log2(N)
    m = 2*n+3
    qubit_nums = m+n
    env = Environment(simType._FULL_AMPLITUDE,show=False) 
    q = env.allocateQubits(qubit_nums)
    print("调用子程序次数:%s"%cnt)
    print("a:%s, N:%s, n:%s ,qubit_nums:%s"%(a,N,n,qubit_nums))
    print("Shor start ......")
    ########## step2 ########### 设置量子线路
    qcirc = env.quantum_circuit
    #傅利叶变换(初始化)
    for i in range(m):
        qcirc.insert(H, target=q[i])
    qcirc.insert(X,target=q[-1])
    #模指运算|a^j mod N >
    qcirc.insert(ConstantModExp(a,N), target=q[m:], control=q[:m])
    #傅利叶逆变换
    lens = m//2
    for i in range(lens):
        qcirc.insert(Swap, target=[q[i], q[m-i-1]])
    for i in range(m-1,-1,-1):
            for j in range(m-1,i,-1):
                theta = round(2*np.pi/2**(j-i+1), 3)
                qcirc.insert(CRDag(Var(theta)), target=q[i], control=q[j])
            qcirc.insert(H, target=q[i])
    for i in range(m):
        qcirc.insert(Measure,target=q[i])
    #qcirc.show()
    ########## step3 ########### 执行线路
    env.exec()
    ########## step4 ########### 返回结果
    #得到psi/2^m
    psi = env.getMeasureResultHandle(target=q[:m])
    print("Shor end")
    r = continued_fraction_expansion(psi, 1<<m)
    print("psi:%s, 2^m:%s, r:%s"%(psi, 1<<m, r))
    return r


if __name__ == "__main__":
    N = int(input('输入数字N:'))
    cnt = 1
    while True:
        a = int(np.random.randint(2,N))
        f = math.gcd(a, N)
        if f!= 1 and f!=N:
            print("幸运：随机选到了因子，结束算法！")
            print("%s = %s X %s"%(N,f,N//f))
            break
        else:
            r = run_shor(a,N, cnt)
            cnt += 1
            if r % 2 == 0 :
                t = int(math.pow(a, r//2))
                p = math.gcd(t + 1, N)
                q = math.gcd(t - 1, N)
                if p!=1 and q!=1 and N==p*q:
                    print('分解成功!')
                    print("%s = %s X %s"%(N,p,q))
                    break
    
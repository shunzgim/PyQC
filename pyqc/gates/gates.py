import math
import cmath
import numpy as np
from scipy.linalg import expm



##############################################################################################
# 可用的量子门:
# H、I、X、Y、Z、S、T、Rx、Ry、Rz;
# CNOT、CZ、Toffili、Swap;
# Measure;
##############################################################################################


##############################################################################################
#定义单比特门H、I、X、Y、Z、S、T、Ph、Rx、Ry、Rz
##############################################################################################
class OneGate:
    """
    """
    def __init__(self, na=None, mat=None, sop=None):
        """
        """
        self.na = na
        self.mat = mat
        self.sop = sop

    @property
    def matrix(self):
        return self.mat

    @property
    def superop(self):
        return self.sop

    @property
    def name(self):
        return self.na

    def __str__(self):
        return self.name


class HGate(OneGate):
    """ 
    """
    def __init__(self):
        name = 'H'
        superop = np.array([[0.5, 0.5, 0.5, 0.5],
                            [0.5, -0.5, 0.5, -0.5],
                            [0.5, 0.5, -0.5, -0.5],
                            [0.5, -0.5, -0.5, 0.5]], dtype=np.complex64)
        matrix = 1. / cmath.sqrt(2.) * np.matrix([[1, 1], [1, -1]], dtype=np.complex64)
        super(HGate, self).__init__(name, matrix, superop)


class HDagGate(OneGate):
    """ 
    """
    def __init__(self):
        name = 'H.dag'
        superop = np.array([[0.5, 0.5, 0.5, 0.5],
                            [0.5, -0.5, 0.5, -0.5],
                            [0.5, 0.5, -0.5, -0.5],
                            [0.5, -0.5, -0.5, 0.5]], dtype=np.complex64)
        matrix = 1. / cmath.sqrt(2.) * np.matrix([[1, 1], [1, -1]], dtype=np.complex64)
        super(HDagGate, self).__init__(name, matrix, superop)


class IGate(OneGate):
    """ 
    """
    def __init__(self):
        name = 'I'
        matrix = np.matrix([[1, 0], [0, 1]], dtype=np.complex64)
        superop = np.array([[1, 0, 0, 0],
                            [0, 1, 0, 0],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1]], dtype=np.complex64)
        super(IGate, self).__init__(name, matrix, superop)


class IDagGate(OneGate):
    """ 
    """
    def __init__(self):
        name = 'I.dag'
        matrix = np.matrix([[1, 0], [0, 1]], dtype=np.complex64)
        superop = np.array([[1, 0, 0, 0],
                            [0, 1, 0, 0],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1]], dtype=np.complex64)
        super(IDagGate, self).__init__(name, matrix, superop)


class XGate(OneGate):
    """
    """
    def __init__(self):
        name = 'X'
        matrix = np.matrix([[0, 1], [1, 0]], dtype=np.complex64)
        superop = np.array([[0, 0, 0, 1],
                            [0, 0, 1, 0],
                            [0, 1, 0, 0],
                            [1, 0, 0, 0]], dtype=np.complex64)
        super(XGate, self).__init__(name, matrix, superop)


class XDagGate(OneGate):
    """
    """
    def __init__(self):
        name = 'X.dag'
        matrix = np.matrix([[0, 1], [1, 0]], dtype=np.complex64)
        superop = np.array([[0, 0, 0, 1],
                            [0, 0, 1, 0],
                            [0, 1, 0, 0],
                            [1, 0, 0, 0]], dtype=np.complex64)
        super(XDagGate, self).__init__(name, matrix, superop)


class YGate(OneGate):
    """ 
    """
    def __init__(self):
        name = 'Y'
        matrix = np.matrix([[0, -1j], [1j, 0]], dtype=np.complex64)
        superop = np.array([[0, 0, 0, 1],
                            [0, 0, -1, 0],
                            [0, -1, 0, 0],
                            [1, 0, 0, 0]], dtype=np.complex64)
        super(YGate, self).__init__(name, matrix, superop)


class YDagGate(OneGate):
    """ 
    """
    def __init__(self):
        name = 'Y.dag'
        matrix = np.matrix([[0, -1j], [1j, 0]], dtype=np.complex64)
        superop = np.array([[0, 0, 0, 1],
                            [0, 0, -1, 0],
                            [0, -1, 0, 0],
                            [1, 0, 0, 0]], dtype=np.complex64)
        super(YDagGate, self).__init__(name, matrix, superop)


class ZGate(OneGate):
    """ 
    """
    def __init__(self):
        name = 'Z'
        matrix = np.matrix([[1, 0], [0, -1]], dtype=np.complex64)
        superop = np.array([[1, 0, 0, 0],
                            [0, -1, 0, 0],
                            [0, 0, -1, 0],
                            [0, 0, 0, 1]], dtype=np.complex64)
        super(ZGate, self).__init__(name, matrix, superop)


class ZDagGate(OneGate):
    """ 
    """
    def __init__(self):
        name = 'Z.dag'
        matrix = np.matrix([[1, 0], [0, -1]], dtype=np.complex64)
        superop = np.array([[1, 0, 0, 0],
                            [0, -1, 0, 0],
                            [0, 0, -1, 0],
                            [0, 0, 0, 1]], dtype=np.complex64)
        super(ZDagGate, self).__init__(name, matrix, superop)


class SGate(OneGate):
    """
    """
    def __init__(self):
        name = 'S'
        matrix = np.matrix([[1, 0], [0, 1j]], dtype=np.complex64)
        superop = np.array([[1, 0, 0, 0],
                            [0, -1j, 0, 0],
                            [0, 0, 1j, 0],
                            [0, 0, 0, 1]], dtype=np.complex64)
        super(SGate, self).__init__(name, matrix, superop)


class SDagGate(OneGate):
    """
    """
    def __init__(self):
        name = 'S.dag'
        matrix = np.matrix([[1, 0], [0, -1j]], dtype=np.complex64)
        superop = np.array([[1, 0, 0, 0],
                            [0, 1j, 0, 0],
                            [0, 0, -1j, 0],
                            [0, 0, 0, 1]], dtype=np.complex64)
        super(SDagGate, self).__init__(name, matrix, superop)


class TGate(OneGate):
    """
    """
    def __init__(self):
        name = 'T'
        matrix = np.matrix([[1, 0], [0, cmath.exp(1j * cmath.pi / 4)]], dtype=np.complex64)
        superop = np.array([[1, 0, 0, 0],
                            [0, cmath.exp(-1j * cmath.pi / 4), 0, 0],
                            [0, 0, cmath.exp(1j * cmath.pi / 4), 0],
                            [0, 0, 0, 1]], dtype=np.complex64)
        super(TGate, self).__init__(name, matrix, superop)


class TDagGate(OneGate):
    """
    """
    def __init__(self):
        name = 'T.dag'
        matrix = np.matrix([[1, 0], [0, cmath.exp(-1j * cmath.pi / 4)]], dtype=np.complex64)
        superop = np.array([[1, 0, 0, 0],
                            [0, cmath.exp(1j * cmath.pi / 4), 0, 0],
                            [0, 0, cmath.exp(-1j * cmath.pi / 4), 0],
                            [0, 0, 0, 1]], dtype=np.complex64)
        super(TDagGate, self).__init__(name, matrix, superop)


class Rx(OneGate):
    """
    """
    def __init__(self, v):
        self.var = v   # 默认Rx(var(theta))
        name = 'Rx(%s)'%round(self.var.angle,3)
        matrix = np.matrix([[math.cos(0.5 * self.var.angle), -1j * math.sin(0.5 * self.var.angle)],
                            [-1j * math.sin(0.5 * self.var.angle),math.cos(0.5 * self.var.angle)]], dtype=np.complex64)
        superop = np.array([[cmath.cos(0.5*self.var.angle)**2, 0.5j*cmath.sin(self.var.angle), -0.5j*cmath.sin(self.var.angle),
                                                                                                 cmath.sin(0.5*self.var.angle)**2],
                            [0.5j*cmath.sin(self.var.angle), cmath.cos(0.5*self.var.angle)**2, cmath.sin(0.5*self.var.angle)**2,
                                                                                                 -0.5j*cmath.sin(self.var.angle)],
                            [-0.5j*cmath.sin(self.var.angle), cmath.sin(0.5*self.var.angle)**2, cmath.cos(0.5*self.var.angle)**2,
                                                                                                 0.5j*cmath.sin(self.var.angle)],
                            [cmath.sin(0.5*self.var.angle)**2, -0.5j*cmath.sin(self.var.angle), 0.5j*cmath.sin(self.var.angle),
                                                                                                 cmath.cos(0.5*self.var.angle)**2]],dtype=np.complex64)
        super(Rx, self).__init__(name, matrix, superop)

    @property
    def matrix(self):
        return np.matrix([[math.cos(0.5 * self.var.angle), -1j * math.sin(0.5 * self.var.angle)],
                          [-1j * math.sin(0.5 * self.var.angle), math.cos(0.5 * self.var.angle)]], dtype=np.complex64)

    @property 
    def superop(self):
        superop = np.array([[cmath.cos(0.5*self.var.angle)**2, 0.5j*cmath.sin(self.var.angle), -0.5j*cmath.sin(self.var.angle),
                                                                                                 cmath.sin(0.5*self.var.angle)**2],
                            [0.5j*cmath.sin(self.var.angle), cmath.cos(0.5*self.var.angle)**2, cmath.sin(0.5*self.var.angle)**2,
                                                                                                 -0.5j*cmath.sin(self.var.angle)],
                            [-0.5j*cmath.sin(self.var.angle), cmath.sin(0.5*self.var.angle)**2, cmath.cos(0.5*self.var.angle)**2,
                                                                                                 0.5j*cmath.sin(self.var.angle)],
                            [cmath.sin(0.5*self.var.angle)**2, -0.5j*cmath.sin(self.var.angle), 0.5j*cmath.sin(self.var.angle),
                                                                                                 cmath.cos(0.5*self.var.angle)**2]],dtype=np.complex64)
        return superop
        
    @property
    def name(self):
        return 'Rx(%s)'%round(self.var.angle,3)


class RxDag(OneGate):
    """
    """
    def __init__(self, v):
        self.var = v   # 默认Rx(var(theta))
        name = 'Rx(%s).dag'%round(self.var.angle,3)
        matrix = np.matrix([[math.cos(0.5 * self.var.angle), 1j * math.sin(0.5 * self.var.angle)],
                            [1j * math.sin(0.5 * self.var.angle), math.cos(0.5 * self.var.angle)]], dtype=np.complex64)
        superop = np.array([[cmath.cos(0.5*self.var.angle)**2, -0.5j*cmath.sin(self.var.angle), 0.5j*cmath.sin(self.var.angle),
                                                                                                 cmath.sin(0.5*self.var.angle)**2],
                            [-0.5j*cmath.sin(self.var.angle), cmath.cos(0.5*self.var.angle)**2, cmath.sin(0.5*self.var.angle)**2,
                                                                                                 0.5j*cmath.sin(self.var.angle)],
                            [0.5j*cmath.sin(self.var.angle), cmath.sin(0.5*self.var.angle)**2, cmath.cos(0.5*self.var.angle)**2,
                                                                                                 -0.5j*cmath.sin(self.var.angle)],
                            [cmath.sin(0.5*self.var.angle)**2, 0.5j*cmath.sin(self.var.angle), -0.5j*cmath.sin(self.var.angle),
                                                                                                 cmath.cos(0.5*self.var.angle)**2]],dtype=np.complex64)
        super(RxDag, self).__init__(name, matrix,superop)

    @property
    def matrix(self):
        return np.matrix([[math.cos(0.5 * self.var.angle), 1j * math.sin(0.5 * self.var.angle)],
                          [1j * math.sin(0.5 * self.var.angle), math.cos(0.5 * self.var.angle)]], dtype=np.complex64)

    @property
    def superop(self):
        superop = np.array([[cmath.cos(0.5*self.var.angle)**2, -0.5j*cmath.sin(self.var.angle), 0.5j*cmath.sin(self.var.angle),
                                                                                                 cmath.sin(0.5*self.var.angle)**2],
                            [-0.5j*cmath.sin(self.var.angle), cmath.cos(0.5*self.var.angle)**2, cmath.sin(0.5*self.var.angle)**2,
                                                                                                 0.5j*cmath.sin(self.var.angle)],
                            [0.5j*cmath.sin(self.var.angle), cmath.sin(0.5*self.var.angle)**2, cmath.cos(0.5*self.var.angle)**2,
                                                                                                 -0.5j*cmath.sin(self.var.angle)],
                            [cmath.sin(0.5*self.var.angle)**2, 0.5j*cmath.sin(self.var.angle), -0.5j*cmath.sin(self.var.angle),
                                                                                                 cmath.cos(0.5*self.var.angle)**2]],dtype=np.complex64)
        return superop

    @property
    def name(self):
        return 'Rx(%s).dag'%round(self.var.angle,3)


class Ry(OneGate):
    """
    """
    def __init__(self, v):
        self.var = v
        name = 'Ry(%s)'%round(self.var.angle,3)
        matrix = np.matrix([[math.cos(0.5 * self.var.angle), -math.sin(0.5 * self.var.angle)],
                            [math.sin(0.5 * self.var.angle), math.cos(0.5 * self.var.angle)]], dtype=np.complex64)
        superop = np.array([[cmath.cos(0.5*self.var.angle)**2, -0.5*cmath.sin(self.var.angle), -0.5*cmath.sin(self.var.angle),
                                                                                                cmath.sin(0.5*self.var.angle)**2],
                            [0.5*cmath.sin(self.var.angle), cmath.cos(0.5*self.var.angle)**2, -cmath.sin(0.5*self.var.angle)**2,
                                                                                                -0.5*cmath.sin(self.var.angle)],
                            [0.5*cmath.sin(self.var.angle), -cmath.sin(0.5*self.var.angle)**2, cmath.cos(0.5*self.var.angle)**2,
                                                                                                -0.5*cmath.sin(self.var.angle)],
                            [cmath.sin(0.5*self.var.angle)**2, 0.5*cmath.sin(self.var.angle), 0.5*cmath.sin(self.var.angle),
                                                                                                cmath.cos(0.5*self.var.angle)**2]],dtype=np.complex64)
        super(Ry, self).__init__(name, matrix, superop)

    @property
    def matrix(self):
        return np.matrix([[math.cos(0.5 * self.var.angle), -math.sin(0.5 * self.var.angle)],
                          [math.sin(0.5 * self.var.angle), math.cos(0.5 * self.var.angle)]], dtype=np.complex64)
    @property
    def superop(self):
        superop = np.array([[cmath.cos(0.5*self.var.angle)**2, -0.5*cmath.sin(self.var.angle), -0.5*cmath.sin(self.var.angle),
                                                                                                cmath.sin(0.5*self.var.angle)**2],
                            [0.5*cmath.sin(self.var.angle), cmath.cos(0.5*self.var.angle)**2, -cmath.sin(0.5*self.var.angle)**2,
                                                                                                -0.5*cmath.sin(self.var.angle)],
                            [0.5*cmath.sin(self.var.angle), -cmath.sin(0.5*self.var.angle)**2, cmath.cos(0.5*self.var.angle)**2,
                                                                                                -0.5*cmath.sin(self.var.angle)],
                            [cmath.sin(0.5*self.var.angle)**2, 0.5*cmath.sin(self.var.angle), 0.5*cmath.sin(self.var.angle),
                                                                                                cmath.cos(0.5*self.var.angle)**2]],dtype=np.complex64)
        return superop

    @property
    def name(self):
        return 'Ry(%s)'%round(self.var.angle,3)
    

class RyDag(OneGate):
    """
    """
    def __init__(self, v):
        self.var = v
        name = 'Ry(%s).dag'%round(self.var.angle,3)
        matrix = np.matrix([[math.cos(0.5 * self.var.angle.value), -math.sin(0.5 * self.var.angle)],
                            [math.sin(0.5 * self.var.angle), math.cos(0.5 * self.var.angle)]], dtype=np.complex64)
        superop = np.array([[cmath.cos(0.5*self.var.angle)**2, 0.5*cmath.sin(self.var.angle), 0.5*cmath.sin(self.var.angle),
                                                                                                cmath.sin(0.5*self.var.angle)**2],
                            [-0.5*cmath.sin(self.var.angle), cmath.cos(0.5*self.var.angle)**2, -cmath.sin(0.5*self.var.angle)**2,
                                                                                                0.5*cmath.sin(self.var.angle)],
                            [-0.5*cmath.sin(self.var.angle), -cmath.sin(0.5*self.var.angle)**2, cmath.cos(0.5*self.var.angle)**2,
                                                                                                0.5*cmath.sin(self.var.angle)],
                            [cmath.sin(0.5*self.var.angle)**2, -0.5*cmath.sin(self.var.angle), -0.5*cmath.sin(self.var.angle),
                                                                                                cmath.cos(0.5*self.var.angle)**2]],dtype=np.complex64)
        super(RyDag, self).__init__(name, matrix, superop)

    @property
    def matrix(self):
        return np.matrix([[math.cos(0.5 * self.var.angle), math.sin(0.5 * self.var.angle)],
                          [-math.sin(0.5 * self.var.angle), math.cos(0.5 * self.var.angle)]], dtype=np.complex64)

    @property
    def superop(self):
        superop = np.array([[cmath.cos(0.5*self.var.angle)**2, 0.5*cmath.sin(self.var.angle), 0.5*cmath.sin(self.var.angle),
                                                                                                cmath.sin(0.5*self.var.angle)**2],
                            [-0.5*cmath.sin(self.var.angle), cmath.cos(0.5*self.var.angle)**2, -cmath.sin(0.5*self.var.angle)**2,
                                                                                                0.5*cmath.sin(self.var.angle)],
                            [-0.5*cmath.sin(self.var.angle), -cmath.sin(0.5*self.var.angle)**2, cmath.cos(0.5*self.var.angle)**2,
                                                                                                0.5*cmath.sin(self.var.angle)],
                            [cmath.sin(0.5*self.var.angle)**2, -0.5*cmath.sin(self.var.angle), -0.5*cmath.sin(self.var.angle),
                                                                                                cmath.cos(0.5*self.var.angle)**2]],dtype=np.complex64)
        return superop

    @property
    def name(self):
        return 'Ry(%s).dag'%round(self.var.angle,3)
    

class Rz(OneGate):
    """
    """
    def __init__(self, v):
        self.var = v
        name = 'Rz(%s)'%round(self.var.angle,3)
        matrix = np.matrix([[cmath.exp(-0.5 * 1j * self.var.angle), 0],
                            [0, cmath.exp(0.5 * 1j * self.var.angle)]], dtype=np.complex64)
        superop = np.array([[1, 0, 0, 0],
                            [0, cmath.exp(-1j*self.var.angle), 0, 0],
                            [0, 0, cmath.exp(1j*self.var.angle), 0],
                            [0, 0, 0, 1]], dtype=np.complex64)
        super(Rz, self).__init__(name, matrix,superop)

    @property
    def matrix(self):
        return np.matrix([[cmath.exp(-0.5 * 1j * self.var.angle), 0],
                        [0, cmath.exp(0.5 * 1j * self.var.angle)]], dtype=np.complex64)

    @property
    def superop(self):
        superop = np.array([[1, 0, 0, 0],
                            [0, cmath.exp(-1j*self.var.angle), 0, 0],
                            [0, 0, cmath.exp(1j*self.var.angle), 0],
                            [0, 0, 0, 1]], dtype=np.complex64)
        return superop

    @property
    def name(self):
        return 'Rz(%s)'%round(self.var.angle,3)


class RzDag(OneGate):
    """
    """
    def __init__(self, v):
        self.var = v
        name = 'Rz(%s).dag'%round(self.var.angle,3)
        matrix = np.matrix([[cmath.exp(0.5 * 1j * self.var.angle), 0],
                            [0, cmath.exp(-0.5 * 1j * self.var.angle)]], dtype=np.complex64)
        superop = np.array([[1, 0, 0, 0],
                            [0, cmath.exp(1j*self.var.angle), 0, 0],
                            [0, 0, cmath.exp(-1j*self.var.angle), 0],
                            [0, 0, 0, 1]], dtype=np.complex64)
        super(RzDag, self).__init__(name, matrix, superop)

    @property
    def matrix(self):
        return np.matrix([[cmath.exp(0.5 * 1j * self.var.angle), 0],
                          [0, cmath.exp(-0.5 * 1j * self.var.angle)]], dtype=np.complex64)

    @property
    def superop(self):
        superop = np.array([[1, 0, 0, 0],
                            [0, cmath.exp(1j*self.var.angle), 0, 0],
                            [0, 0, cmath.exp(-1j*self.var.angle), 0],
                            [0, 0, 0, 1]], dtype=np.complex64)
        return superop

    @property
    def name(self):
        return 'Rz(%s).dag'%round(self.var.angle,3)


class HSimOneGate(OneGate):
    """
    exp(i*A*t)
    """
    def __init__(self, A, t, theta):
        name = 'HSim_%s'%round(theta,3)
        matrix = np.matrix(expm(1j*t*A*theta))
        superop = None
        super(HSimOneGate, self).__init__(name, matrix, superop)


class HSimDagOneGate(OneGate):
    def __init__(self, A, t, theta):
        name = 'HSim_Dag%s'%round(theta,3)
        matrix = np.matrix(expm(1j*t*A*theta)).H
        superop = None
        super(HSimDagOneGate, self).__init__(name, matrix, superop)


##############################################################################################
#定义受控门CNOTGate、CZGate、ToffiliGate以及交换门SwapGate,注意：量子门的矩阵需要在运行时计算
##############################################################################################
class ControledGate:
    def __init__(self, na=None, cmat=None, sop=None):
        self.na = na
        self.cmat = cmat
        self.sop = sop

    @property
    def name(self):
        return self.na

    @property
    def cmatrix(self):
        return self.cmat

    @property
    def superop(self):
        return self.sop

    def __str__(self):
        return self.name


class CNOTGate(ControledGate):
    def __init__(self):
        name = 'CNOT'
        cmatrix = np.matrix([[0, 1],
                            [1, 0]], dtype=np.complex64)
        superop = np.array([[[[1,0,0,0],
                              [0,0,0,0],
                              [0,0,0,0],
                              [0,0,0,0]],
                             [[0,1,0,0],
                              [0,0,0,0],
                              [0,0,0,0],
                              [0,0,0,0]],
                             [[0,0,1,0],
                              [0,0,0,0],
                              [0,0,0,0],
                              [0,0,0,0]],
                             [[0,0,0,1],
                              [0,0,0,0],
                              [0,0,0,0],
                              [0,0,0,0]]],
                            [[[0,0,0,0],
                              [0,1,0,0],
                              [0,0,0,0],
                              [0,0,0,0]],
                             [[0,0,0,0],
                              [1,0,0,0],
                              [0,0,0,0],
                              [0,0,0,0]],
                             [[0,0,0,0],
                              [0,0,0,1],
                              [0,0,0,0],
                              [0,0,0,0]],
                             [[0,0,0,0],
                              [0,0,1,0],
                              [0,0,0,0],
                              [0,0,0,0]]],
                            [[[0,0,0,0],
                              [0,0,0,0],
                              [0,0,1,0],
                              [0,0,0,0]],
                             [[0,0,0,0],
                              [0,0,0,0],
                              [0,0,0,1],
                              [0,0,0,0]],
                             [[0,0,0,0],
                              [0,0,0,0],
                              [1,0,0,0],
                              [0,0,0,0]],
                             [[0,0,0,0],
                              [0,0,0,0],
                              [0,1,0,0],
                              [0,0,0,0]]],
                            [[[0,0,0,0],
                              [0,0,0,0],
                              [0,0,0,0],
                              [0,0,0,1]],
                             [[0,0,0,0],
                              [0,0,0,0],
                              [0,0,0,0],
                              [0,0,1,0]],
                             [[0,0,0,0],
                              [0,0,0,0],
                              [0,0,0,0],
                              [0,1,0,0]],
                             [[0,0,0,0],
                              [0,0,0,0],
                              [0,0,0,0],
                              [1,0,0,0]]]],dtype=np.complex64)
        super(CNOTGate,self).__init__(name, cmatrix, superop)


class CNOTDagGate(ControledGate):
    def __init__(self):
        name = 'CNOT.dag'
        cmatrix = np.matrix([[0, 1],
                             [1, 0]], dtype=np.complex64)
        superop = np.array([[[[1,0,0,0],
                              [0,0,0,0],
                              [0,0,0,0],
                              [0,0,0,0]],
                             [[0,1,0,0],
                              [0,0,0,0],
                              [0,0,0,0],
                              [0,0,0,0]],
                             [[0,0,1,0],
                              [0,0,0,0],
                              [0,0,0,0],
                              [0,0,0,0]],
                             [[0,0,0,1],
                              [0,0,0,0],
                              [0,0,0,0],
                              [0,0,0,0]]],
                            [[[0,0,0,0],
                              [0,1,0,0],
                              [0,0,0,0],
                              [0,0,0,0]],
                             [[0,0,0,0],
                              [1,0,0,0],
                              [0,0,0,0],
                              [0,0,0,0]],
                             [[0,0,0,0],
                              [0,0,0,1],
                              [0,0,0,0],
                              [0,0,0,0]],
                             [[0,0,0,0],
                              [0,0,1,0],
                              [0,0,0,0],
                              [0,0,0,0]]],
                            [[[0,0,0,0],
                              [0,0,0,0],
                              [0,0,1,0],
                              [0,0,0,0]],
                             [[0,0,0,0],
                              [0,0,0,0],
                              [0,0,0,1],
                              [0,0,0,0]],
                             [[0,0,0,0],
                              [0,0,0,0],
                              [1,0,0,0],
                              [0,0,0,0]],
                             [[0,0,0,0],
                              [0,0,0,0],
                              [0,1,0,0],
                              [0,0,0,0]]],
                            [[[0,0,0,0],
                              [0,0,0,0],
                              [0,0,0,0],
                              [0,0,0,1]],
                             [[0,0,0,0],
                              [0,0,0,0],
                              [0,0,0,0],
                              [0,0,1,0]],
                             [[0,0,0,0],
                              [0,0,0,0],
                              [0,0,0,0],
                              [0,1,0,0]],
                             [[0,0,0,0],
                              [0,0,0,0],
                              [0,0,0,0],
                              [1,0,0,0]]]],dtype=np.complex64)
        super(CNOTDagGate,self).__init__(name, cmatrix,superop)


class CZGate(ControledGate):
    def __init__(self):
        name = 'CZ'
        cmatrix = np.matrix([[1, 0],
                             [0, -1]], dtype=np.complex64)
        superop = np.array([[[[1,0,0,0],
                          [0,0,0,0],
                          [0,0,0,0],
                          [0,0,0,0]],
                         [[0,1,0,0],
                          [0,0,0,0],
                          [0,0,0,0],
                          [0,0,0,0]],
                         [[0,0,1,0],
                          [0,0,0,0],
                          [0,0,0,0],
                          [0,0,0,0]],
                         [[0,0,0,1],
                          [0,0,0,0],
                          [0,0,0,0],
                          [0,0,0,0]]],
                        [[[0,0,0,0],
                          [1,0,0,0],
                          [0,0,0,0],
                          [0,0,0,0]],
                         [[0,0,0,0],
                          [0,-1,0,0],
                          [0,0,0,0],
                          [0,0,0,0]],
                         [[0,0,0,0],
                          [0,0,1,0],
                          [0,0,0,0],
                          [0,0,0,0]],
                         [[0,0,0,0],
                          [0,0,0,-1],
                          [0,0,0,0],
                          [0,0,0,0]]],
                        [[[0,0,0,0],
                          [0,0,0,0],
                          [1,0,0,0],
                          [0,0,0,0]],
                         [[0,0,0,0],
                          [0,0,0,0],
                          [0,1,0,0],
                          [0,0,0,0]],
                         [[0,0,0,0],
                          [0,0,0,0],
                          [0,0,-1,0],
                          [0,0,0,0]],
                         [[0,0,0,0],
                          [0,0,0,0],
                          [0,0,0,-1],
                          [0,0,0,0]]],
                        [[[0,0,0,0],
                          [0,0,0,0],
                          [0,0,0,0],
                          [1,0,0,0]],
                         [[0,0,0,0],
                          [0,0,0,0],
                          [0,0,0,0],
                          [0,-1,0,0]],
                         [[0,0,0,0],
                          [0,0,0,0],
                          [0,0,0,0],
                          [0,0,-1,0]],
                         [[0,0,0,0],
                          [0,0,0,0],
                          [0,0,0,0],
                          [0,0,0,1]]]],dtype=np.complex64)
        super(CZGate,self).__init__(name, cmatrix, superop)


class CZDagGate(ControledGate):
    def __init__(self):
        name = 'CZ.dag'
        cmatrix = np.matrix([[1, 0],
                             [0, -1]], dtype=np.complex64)
        superop = np.array([[[[1,0,0,0],
                          [0,0,0,0],
                          [0,0,0,0],
                          [0,0,0,0]],
                         [[0,1,0,0],
                          [0,0,0,0],
                          [0,0,0,0],
                          [0,0,0,0]],
                         [[0,0,1,0],
                          [0,0,0,0],
                          [0,0,0,0],
                          [0,0,0,0]],
                         [[0,0,0,1],
                          [0,0,0,0],
                          [0,0,0,0],
                          [0,0,0,0]]],
                        [[[0,0,0,0],
                          [1,0,0,0],
                          [0,0,0,0],
                          [0,0,0,0]],
                         [[0,0,0,0],
                          [0,-1,0,0],
                          [0,0,0,0],
                          [0,0,0,0]],
                         [[0,0,0,0],
                          [0,0,1,0],
                          [0,0,0,0],
                          [0,0,0,0]],
                         [[0,0,0,0],
                          [0,0,0,-1],
                          [0,0,0,0],
                          [0,0,0,0]]],
                        [[[0,0,0,0],
                          [0,0,0,0],
                          [1,0,0,0],
                          [0,0,0,0]],
                         [[0,0,0,0],
                          [0,0,0,0],
                          [0,1,0,0],
                          [0,0,0,0]],
                         [[0,0,0,0],
                          [0,0,0,0],
                          [0,0,-1,0],
                          [0,0,0,0]],
                         [[0,0,0,0],
                          [0,0,0,0],
                          [0,0,0,-1],
                          [0,0,0,0]]],
                        [[[0,0,0,0],
                          [0,0,0,0],
                          [0,0,0,0],
                          [1,0,0,0]],
                         [[0,0,0,0],
                          [0,0,0,0],
                          [0,0,0,0],
                          [0,-1,0,0]],
                         [[0,0,0,0],
                          [0,0,0,0],
                          [0,0,0,0],
                          [0,0,-1,0]],
                         [[0,0,0,0],
                          [0,0,0,0],
                          [0,0,0,0],
                          [0,0,0,1]]]],dtype=np.complex64)
        super(CZDagGate,self).__init__(name, cmatrix, superop)


class CR(ControledGate):
    def __init__(self, v):
        self.var = v
        name = 'CR(%s)'%round(self.var.angle,3)
        cmatrix = np.matrix([[1, 0],
                             [0, cmath.exp(1j*self.var.angle)]], dtype=np.complex64)
        superop = np.array([[[[1,0,0,0],
                          [0,0,0,0],
                          [0,0,0,0],
                          [0,0,0,0]],
                         [[0,1,0,0],
                          [0,0,0,0],
                          [0,0,0,0],
                          [0,0,0,0]],
                         [[0,0,1,0],
                          [0,0,0,0],
                          [0,0,0,0],
                          [0,0,0,0]],
                         [[0,0,0,1],
                          [0,0,0,0],
                          [0,0,0,0],
                          [0,0,0,0]]],
                        [[[0,0,0,0],
                          [1,0,0,0],
                          [0,0,0,0],
                          [0,0,0,0]],
                         [[0,0,0,0],
                          [0,cmath.exp(1j*self.var.angle),0,0],
                          [0,0,0,0],
                          [0,0,0,0]],
                         [[0,0,0,0],
                          [0,0,1,0],
                          [0,0,0,0],
                          [0,0,0,0]],
                         [[0,0,0,0],
                          [0,0,0,cmath.exp(1j*self.var.angle)],
                          [0,0,0,0],
                          [0,0,0,0]]],
                        [[[0,0,0,0],
                          [0,0,0,0],
                          [1,0,0,0],
                          [0,0,0,0]],
                         [[0,0,0,0],
                          [0,0,0,0],
                          [0,1,0,0],
                          [0,0,0,0]],
                         [[0,0,0,0],
                          [0,0,0,0],
                          [0,0,cmath.exp(1j*self.var.angle),0],
                          [0,0,0,0]],
                         [[0,0,0,0],
                          [0,0,0,0],
                          [0,0,0,cmath.exp(1j*self.var.angle)],
                          [0,0,0,0]]],
                        [[[0,0,0,0],
                          [0,0,0,0],
                          [0,0,0,0],
                          [1,0,0,0]],
                         [[0,0,0,0],
                          [0,0,0,0],
                          [0,0,0,0],
                          [0,cmath.exp(1j*self.var.angle),0,0]],
                         [[0,0,0,0],
                          [0,0,0,0],
                          [0,0,0,0],
                          [0,0,cmath.exp(1j*self.var.angle),0]],
                         [[0,0,0,0],
                          [0,0,0,0],
                          [0,0,0,0],
                          [0,0,0,1]]]],dtype=np.complex64)
        super(CR,self).__init__(name, cmatrix, superop)

    @property
    def cmatrix(self):
        return np.matrix([[1, 0],
                          [0, cmath.exp(1j*self.var.angle)]], dtype=np.complex64)

    @property
    def superop(self):
        superop = np.array([[[[1,0,0,0],
                          [0,0,0,0],
                          [0,0,0,0],
                          [0,0,0,0]],
                         [[0,1,0,0],
                          [0,0,0,0],
                          [0,0,0,0],
                          [0,0,0,0]],
                         [[0,0,1,0],
                          [0,0,0,0],
                          [0,0,0,0],
                          [0,0,0,0]],
                         [[0,0,0,1],
                          [0,0,0,0],
                          [0,0,0,0],
                          [0,0,0,0]]],
                        [[[0,0,0,0],
                          [1,0,0,0],
                          [0,0,0,0],
                          [0,0,0,0]],
                         [[0,0,0,0],
                          [0,cmath.exp(1j*self.var.angle),0,0],
                          [0,0,0,0],
                          [0,0,0,0]],
                         [[0,0,0,0],
                          [0,0,1,0],
                          [0,0,0,0],
                          [0,0,0,0]],
                         [[0,0,0,0],
                          [0,0,0,cmath.exp(1j*self.var.angle)],
                          [0,0,0,0],
                          [0,0,0,0]]],
                        [[[0,0,0,0],
                          [0,0,0,0],
                          [1,0,0,0],
                          [0,0,0,0]],
                         [[0,0,0,0],
                          [0,0,0,0],
                          [0,1,0,0],
                          [0,0,0,0]],
                         [[0,0,0,0],
                          [0,0,0,0],
                          [0,0,cmath.exp(1j*self.var.angle),0],
                          [0,0,0,0]],
                         [[0,0,0,0],
                          [0,0,0,0],
                          [0,0,0,cmath.exp(1j*self.var.angle)],
                          [0,0,0,0]]],
                        [[[0,0,0,0],
                          [0,0,0,0],
                          [0,0,0,0],
                          [1,0,0,0]],
                         [[0,0,0,0],
                          [0,0,0,0],
                          [0,0,0,0],
                          [0,cmath.exp(1j*self.var.angle),0,0]],
                         [[0,0,0,0],
                          [0,0,0,0],
                          [0,0,0,0],
                          [0,0,cmath.exp(1j*self.var.angle),0]],
                         [[0,0,0,0],
                          [0,0,0,0],
                          [0,0,0,0],
                          [0,0,0,1]]]],dtype=np.complex64)
        return superop

    @property
    def name(self):
        return 'CR(%s)'%round(self.var.angle,3)


class CRDag(ControledGate):
    def __init__(self, v):
        self.var = v
        name = 'CR(%s).dag'%round(self.var.angle,3)
        cmatrix = np.matrix([[1, 0],
                             [0, cmath.exp(-1j*self.var.angle)]], dtype=np.complex64)
        superop = np.array([[[[1,0,0,0],
                          [0,0,0,0],
                          [0,0,0,0],
                          [0,0,0,0]],
                         [[0,1,0,0],
                          [0,0,0,0],
                          [0,0,0,0],
                          [0,0,0,0]],
                         [[0,0,1,0],
                          [0,0,0,0],
                          [0,0,0,0],
                          [0,0,0,0]],
                         [[0,0,0,1],
                          [0,0,0,0],
                          [0,0,0,0],
                          [0,0,0,0]]],
                        [[[0,0,0,0],
                          [1,0,0,0],
                          [0,0,0,0],
                          [0,0,0,0]],
                         [[0,0,0,0],
                          [0,cmath.exp(-1j*self.var.angle),0,0],
                          [0,0,0,0],
                          [0,0,0,0]],
                         [[0,0,0,0],
                          [0,0,1,0],
                          [0,0,0,0],
                          [0,0,0,0]],
                         [[0,0,0,0],
                          [0,0,0,cmath.exp(-1j*self.var.angle)],
                          [0,0,0,0],
                          [0,0,0,0]]],
                        [[[0,0,0,0],
                          [0,0,0,0],
                          [1,0,0,0],
                          [0,0,0,0]],
                         [[0,0,0,0],
                          [0,0,0,0],
                          [0,1,0,0],
                          [0,0,0,0]],
                         [[0,0,0,0],
                          [0,0,0,0],
                          [0,0,cmath.exp(-1j*self.var.angle),0],
                          [0,0,0,0]],
                         [[0,0,0,0],
                          [0,0,0,0],
                          [0,0,0,cmath.exp(-1j*self.var.angle)],
                          [0,0,0,0]]],
                        [[[0,0,0,0],
                          [0,0,0,0],
                          [0,0,0,0],
                          [1,0,0,0]],
                         [[0,0,0,0],
                          [0,0,0,0],
                          [0,0,0,0],
                          [0,cmath.exp(-1j*self.var.angle),0,0]],
                         [[0,0,0,0],
                          [0,0,0,0],
                          [0,0,0,0],
                          [0,0,cmath.exp(-1j*self.var.angle),0]],
                         [[0,0,0,0],
                          [0,0,0,0],
                          [0,0,0,0],
                          [0,0,0,1]]]],dtype=np.complex64)
        super(CRDag,self).__init__(name, cmatrix, superop)

    @property
    def cmatrix(self):
        return np.matrix([[1, 0],
                          [0, cmath.exp(-1j*self.var.angle)]], dtype=np.complex64)

    @property
    def superop(self):
        superop = np.array([[[[1,0,0,0],
                          [0,0,0,0],
                          [0,0,0,0],
                          [0,0,0,0]],
                         [[0,1,0,0],
                          [0,0,0,0],
                          [0,0,0,0],
                          [0,0,0,0]],
                         [[0,0,1,0],
                          [0,0,0,0],
                          [0,0,0,0],
                          [0,0,0,0]],
                         [[0,0,0,1],
                          [0,0,0,0],
                          [0,0,0,0],
                          [0,0,0,0]]],
                        [[[0,0,0,0],
                          [1,0,0,0],
                          [0,0,0,0],
                          [0,0,0,0]],
                         [[0,0,0,0],
                          [0,cmath.exp(-1j*self.var.angle),0,0],
                          [0,0,0,0],
                          [0,0,0,0]],
                         [[0,0,0,0],
                          [0,0,1,0],
                          [0,0,0,0],
                          [0,0,0,0]],
                         [[0,0,0,0],
                          [0,0,0,cmath.exp(-1j*self.var.angle)],
                          [0,0,0,0],
                          [0,0,0,0]]],
                        [[[0,0,0,0],
                          [0,0,0,0],
                          [1,0,0,0],
                          [0,0,0,0]],
                         [[0,0,0,0],
                          [0,0,0,0],
                          [0,1,0,0],
                          [0,0,0,0]],
                         [[0,0,0,0],
                          [0,0,0,0],
                          [0,0,cmath.exp(-1j*self.var.angle),0],
                          [0,0,0,0]],
                         [[0,0,0,0],
                          [0,0,0,0],
                          [0,0,0,cmath.exp(-1j*self.var.angle)],
                          [0,0,0,0]]],
                        [[[0,0,0,0],
                          [0,0,0,0],
                          [0,0,0,0],
                          [1,0,0,0]],
                         [[0,0,0,0],
                          [0,0,0,0],
                          [0,0,0,0],
                          [0,cmath.exp(-1j*self.var.angle),0,0]],
                         [[0,0,0,0],
                          [0,0,0,0],
                          [0,0,0,0],
                          [0,0,cmath.exp(-1j*self.var.angle),0]],
                         [[0,0,0,0],
                          [0,0,0,0],
                          [0,0,0,0],
                          [0,0,0,1]]]],dtype=np.complex64)
        return superop

    @property
    def name(self):
        return 'CR(%s).dag'%round(self.var.angle,3)


class CUOne(ControledGate):
    def __init__(self, cgate):
        name = "CU(%s)"%cgate.name
        cmatrix = cgate.matrix
        superop = None
        super(CUOne,self).__init__(name, cmatrix, superop)


class ToffiliGate:
    def __init__(self):
        pass

    @property
    def name(self):
        return "Toffili"

    def __str__(self):
        return "Toffili"


class ToffiliDagGate:
    def __init__(self):
        pass

    @property
    def name(self):
        return "Toffili.dag"

    def __str__(self):
        return "Toffili.dag"


class SwapGate:
    def __init__(self):
        pass

    @property
    def name(self):
        return "Swap"

    def __str__(self):
        return "Swap"


class SwapDagGate:
    def __init__(self):
        pass

    @property
    def name(self):
        return "Swap.dag"

    def __str__(self):
        return "Swap.dag"


##############################################################################################
#定义高水平门ConstantModExp,注意：量子门的矩阵需要在运行时计算
##############################################################################################
class ConstantModExp:
    def __init__(self,a,N):
        self.a = a
        self.N = N

    @property
    def name(self):
        return "CMExp"

    def __str__(self):
        return "CMExp"


##############################################################################################
#定义测量门MeasureGate，注意：量子门的矩阵需要在运行时计算
##############################################################################################
class MeasureGate:
    def __init__(self):
        pass

    @property
    def name(self):
        return "Measure"

    def __str__(self):
        return "Measure"


I = IGate()
IDag = IDagGate()
X = XGate()
XDag = XDagGate()
Y = YGate()
YDag = YDagGate()
Z = ZGate()
ZDag = ZDagGate()
S = SGate()
SDag = SDagGate()
T = TGate()
TDag = TDagGate()
H = HGate()
HDag = HDagGate()
CNOT = CNOTGate()
CNOTDag = CNOTDagGate
CZ = CZGate()
CZDag = CZDagGate()
Toffili = ToffiliGate()
ToffiliDag = ToffiliDagGate()
Swap = SwapGate()
SwapDag = SwapDagGate()
Measure = MeasureGate()




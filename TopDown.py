import numpy as np
import normalization

#自顶向下图推理
def fHGI(alpha, A_RR, A_DD, A_RD):


    normWRR = normalization.normFun(A_RR) #这是药物相似性矩阵
    normWDD = normalization.normFun(A_DD) #这是疾病相似性矩阵



    Wdr0 = A_RD

    Wdr_i = Wdr0



    Wdr_I = alpha * normWRR @ Wdr_i @ normWDD + (1 - alpha) * Wdr0


    while np.max(np.abs(Wdr_I - Wdr_i)) > 1e-10:
        Wdr_i = Wdr_I
        Wdr_I = alpha * normWRR @ Wdr_i @ normWDD + (1 - alpha) * Wdr0

    T_recovery = Wdr_I
    return T_recovery
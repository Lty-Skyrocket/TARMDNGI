import numpy as np


from sklearn.model_selection import KFold
from sklearn.metrics import roc_curve, auc
import copy
import TopDown
import BottomUp
import Neighborhood_constraint


drug = np.loadtxt(r"drug_integration.txt", dtype=float)
disease = np.loadtxt(r"disease_integration.txt", dtype=float)



Y = np.loadtxt(r"association.txt",dtype=float)

drug_disease_k = np.loadtxt(r"known.txt",dtype=int)
drug_disease_uk = np.loadtxt(r"unknown.txt",dtype=int)



#截断反正切秩最小化和双策略邻域约束图推理技术

def DC(D,mu,T0):
    U,S,V = np.linalg.svd(D)
    T1 = np.zeros(np.size(T0))
    for i in range(1,100):
        T1 = DCInner(S,mu,T0)
        err = np.sum(np.square(T1-T0))
        if err < 1e-6:
            break
        T0 = T1

    l_1 = np.dot(U, np.diag(T1))
    l = np.dot(l_1, V)
    return l,T1

def DCInner(S,mu,T_k):
    lamb = 1/mu
    grad = 1/(1+np.square(T_k))
    T_k1 = S-lamb*grad
    T_k1[T_k1<0]=0
    return T_k1


def GAMA(H,A,B):
    muzero = 60
    mu = muzero
    rho = 5
    tol = 1e-3
    alpha = 20

    m, n = np.shape(H)
    L = copy.deepcopy(H)
    Y = np.zeros((m,n))

    omega = np.zeros(H.shape)
    omega[H.nonzero()] = 1

    for i in range(0, 500):


        #这些代码是求W的
        tran = (1/mu) * (Y+alpha*(H*omega)+np.dot(A,B))+L
        W = tran - (alpha/(alpha+mu))*omega*tran
        W[W < 0] = 0
        W[W > 1] = 1

        #这三项整体算是求奇异值的,也就是X,在这里L就相当于X了
        D = W-Y/mu  #更新C
        sig = np.zeros(min(m, n)) #存奇异值的
        L, sig = DC(copy.deepcopy(D),mu,copy.deepcopy(sig)) #求奇异值的

        #求Y
        Y= Y+mu*(L-W)     #更新Y
        mu = mu*rho         #更新u
        sigma = np.linalg.norm(L-W,'fro')
        RRE = sigma/np.linalg.norm(H,'fro')
        if RRE < tol:
            break

    return W

def truncated(H0):
    for i in range(0,1):
        U, S, V = np.linalg.svd(H0)
        r = 20#Therapeutic dataset:5#lncRNA-disease dataset:5#microbe-disease dataset:5
        A = U[:, :r]
        B = V[:r, :]
        H0 = GAMA(H0,A,B)

    Smmi = H0
    return Smmi

def main():
        a3 = np.hstack((drug, Y))  # 将参数元组的元素数组按水平方向进行叠加
        a4 = np.hstack((np.transpose(Y), disease))  # 对矩阵b进行转置操作
        H = np.vstack((a3, a4))  # 将参数元组的元素数组按垂直方向进行叠加

        L_1 = truncated(H)

        L_1 = L_1[0:drug.shape[0], drug.shape[0]:H.shape[1]]  # 把补充的关联矩阵原来A位置给取出来
        # # 计算药物的局部相似性
        L_drug = Neighborhood_constraint.row_normalization(drug, 10)#Therapeutic dataset:100;#lncRNA-disease dataset:50#microbe-disease dataset:50
        # # 计算疾病的局部相似性
        L_disease = Neighborhood_constraint.row_normalization(disease, 20)#Therapeutic dataset:100;#lncRNA-disease dataset:50#microbe-disease dataset:50

        # 自顶向下图推理
        Yh1 = TopDown.fHGI(0.01, L_drug, L_disease, L_1)
        # 自底向上图推理
        Yh2 = BottomUp.fHGI(0.01, L_drug, L_disease, L_1)
        M_1 = (0.5 * Yh1 + 0.5 * Yh2)
        return M_1


if __name__ == "__main__":

        l=main()

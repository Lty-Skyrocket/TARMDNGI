import numpy as np

def normFun(M):
    num = M.shape[0]
    nM = np.zeros((num, num))
    result = np.zeros((num, num))

    for i in range(num):
        nM[i, i] = np.sum(M[i, :])

    for i in range(num):
        rsum = nM[i, i]
        for j in range(num):
            csum = nM[j, j]
            if (rsum == 0) or (csum == 0):
                result[i, j] = 0
            else:
                result[i, j] = M[i, j] / np.sqrt(rsum * csum)

    return result


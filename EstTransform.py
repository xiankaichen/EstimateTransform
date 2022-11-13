import numpy as np;
from scipy.spatial.transform import Rotation

import logging
class EstTransform():
    def __init__(self):
        pass

    def rpy2rotate(self, rpy, euler_type='xyz'):
        '''
        rpy: 3x1 rotation matrix
        :param euler_type:
        :return:
        '''
        rpy = Rotation.from_euler(euler_type, rpy, True)
        rotate = rpy.as_matrix()
        return rotate

    def rotate2rpy(self, matrix, euler_type='xyz'):
        '''
        matrix: 3x3 rotation matrix
        :param euler_type:
        :return:
        '''
        x = Rotation.from_matrix(matrix)
        rpy = x.as_euler(euler_type, True)
        return rpy

    def getTransform(self, src, dst):
        '''
        src: (M, N) array, Source coordinates.N表示维度，M表示点数
        :param dst: (M, N) array, Destination coordinates.N表示维度，M表示点数
        :return: T : (N + 1, N + 1),变换矩阵
        '''

        dim = src.shape[1]

        # Compute mean of src and dst.
        src_mean = src.mean(axis=0)
        dst_mean = dst.mean(axis=0)

        # Subtract mean from src and dst.
        src_demean = src - src_mean
        dst_demean = dst - dst_mean

        A = np.dot(dst_demean.T, src_demean)

        d = np.ones((dim,), dtype=np.double)
        if np.linalg.det(A) < 0:
            d[dim - 1] = -1

        T = np.eye(dim + 1, dtype=np.double)

        U, S, Vh = np.linalg.svd(A)

        rank = np.linalg.matrix_rank(A)

        if rank == 0:
            return np.nan * T,1000000
        elif rank == dim - 1:
            if np.linalg.det(U) * np.linalg.det(Vh) > 0:
                T[:dim, :dim] = np.dot(U, Vh)
            else:
                s = d[dim - 1]
                d[dim - 1] = -1
                T[:dim, :dim] = np.dot(U, np.dot(np.diag(d), Vh))
                d[dim - 1] = s
        else:
            T[:dim, :dim] = np.dot(U, np.dot(np.diag(d), Vh))

        if(S[1]<S[0]*0.02*0.02):
            print("transform between 3D with estTransform() is almost singular!!!")
            return np.nan,1000000

        T[:dim, dim] = dst_mean - np.dot(T[:dim, :dim], src_mean.T)

        errMx = np.concatenate((src,np.ones([src.shape[0], 1])),axis=1).dot(T[:dim, :].transpose()) - dst
        err = np.sqrt(np.mean(np.sum(errMx*errMx,axis=1)))

        return  T,err

    def test(self, srcPoints0,dstPoints0):
        '''
        刚体变换估计方法，使用scipy库中的Rotation类
        :param srcPoints0: 原始点
        :param dstPoints0: 目标点
        :return:
        '''
        errThr = 1
        T, err = self.getTransform(srcPoints0, dstPoints0)
        logging.warning(" 变换误差 = {}".format(err))
        logging.warning("变换矩阵T = {}".format(T))
        if err > errThr:
            logging.warning("变换误差过大 = {}".format(err))
            print("###################################################################################")
            # return np.nan

        for i in range(0, 4):
            posM0 = T.dot(np.array([[srcPoints0[i][0]], [srcPoints0[i][1]], [srcPoints0[i][2]], [1]]))
            print("原始点" + str(i) + "：" + str(srcPoints0[i]) +
                  "，\n变换点：" + str(posM0.transpose()) +
                  "，\n期望点：" + str(dstPoints0[i]))
            print(posM0.T)
            # print("\n")
        MxM0 = T[:3, : 3]
        angle = self.rotate2rpy(MxM0, euler_type='xyz')
        print("平移向量 = {}".format(T[:3, 3]))
        print("旋转角度 = {}".format(angle))
        return

if __name__ == '__main__':
    srcPoints0 = np.array([[2692, 2208, 2167], [3219, 770, 3229],
                           [4876, -188, 4661], [5227, 1274, 5204]])
    dstPoints0 = np.array(
        [[1928.891, 3509.387, 872.249], [2501.499, 2951.888, 2555.915],
         [4023.698, 3245.273, 4375.496], [4019.806, 4820.297, 4102.038]])

    estTransform = EstTransform()
    # 调用测试方法
    estTransform.test(srcPoints0, dstPoints0)

    # 仅调用估计变换的方法
    T, err = estTransform.getTransform(srcPoints0, dstPoints0)
    print("变换矩阵T = {}".format(T))
    print("变换误差 = {}".format(err))

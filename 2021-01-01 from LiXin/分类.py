import cv2
import numpy as np
import math
from copy import deepcopy
 
 
# 设置样本数据，设置样本标签
def Dataset():
    # 共35个样本数据  杂草10 背景10 作物15
    samples_data = [[199,114,34], [199,113,36], [188,93,11],
                    [196,101,17], [193,97,13], [85,37,23],
                    [192,99,19], [178,83,3], [87,38,24],
                    [87,37,26], [126,137,143], [125,136,142],
                    [131,142,148], [130,141,147], [129,140,146],
                    [181,197,210], [183,201,213], [182,200,212],
                    [180,198,210], [181,199,209], [166,108,94],
                    [167,108,94], [165,107,93], [159,99,89],
                    [156,96,86], [154,94,84], [149,81,78],
                    [155,87,84], [157,82,86], [156,81,85],
                    [156,84,85], [130,58,62], [166,110,93],
                    [150,83,77], [130,58,62]]
    class_lable = [1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2]
    # 杂草--1 背景--0 作物--2
    return samples_data,class_lable
 
# 获取RGB均值向量
def Get_Junzhi(samples_data,class_lable):
    L = len(samples_data)
    m = 0  # 植物
    n = 0  # 背景
    k = 0  # 作物
    r1 = 0
    g1 = 0
    b1 = 0
    r0 = 0
    g0 = 0
    b0 = 0
    r2 = 0
    g2 = 0
    b2 = 0
    mv = [0]
    mean_vector1 = [mv]*3
    mean_vector0 = [mv]*3
    mean_vector2 = [mv]*3
    for i in range(L):
        if class_lable[i] == 1:
            m += 1
        elif class_lable[i] == 0:
            n += 1
        else:
            k += 1
 
    for i in range(L):
        if i < m:
            r1 += samples_data[i][0]
            g1 += samples_data[i][1]
            b1 += samples_data[i][2]
        elif i >= m+n:
            r2 += samples_data[i][0]
            g2 += samples_data[i][1]
            b2 += samples_data[i][2]
        else:
            r0 += samples_data[i][0]
            g0 += samples_data[i][1]
            b0 += samples_data[i][2]
 
    mean_vector1 = [[int(r1/m)],[int(g1/m)],[int(b1/m)]]  # 三维均值向量
    mean_vector0 = [[int(r0/n)],[int(g0/n)],[int(b0/n)]]
    mean_vector2 = [[int(r2/k)],[int(g2/k)],[int(b2/k)]]
    #print(mean_vector1)
    #print(mean_vector0)
    #print(mean_vector2)
    return mean_vector1, mean_vector0, mean_vector2
 
# 计算协方差矩阵 3x3
def Get_Cov(samples_data,mean_vector1,mean_vector0,mean_vector2):
    L= len(samples_data)
    m = 10
    n = 10
    k = 15
    cov = [0]*3
    Cov_1 = [cov]*3
    Cov_0 = [cov]*3
    Cov_2 = [cov]*3
    cov_bb1 = 0
    cov_gb1 = 0
    cov_gg1 = 0
    cov_rb1 = 0
    cov_rg1 = 0
    cov_rr1 = 0
    cov_bb0 = 0
    cov_gb0 = 0
    cov_gg0 = 0
    cov_rb0 = 0
    cov_rg0 = 0
    cov_rr0 = 0
    cov_bb2 = 0
    cov_gb2 = 0
    cov_gg2 = 0
    cov_rb2 = 0
    cov_rg2 = 0
    cov_rr2 = 0
    for i in range(L):
        if i < m:
            cov_rr1 += (samples_data[i][0]-mean_vector1[0][0])*(samples_data[i][0]-mean_vector1[0][0])
            cov_rg1 += (samples_data[i][0]-mean_vector1[0][0])*(samples_data[i][1]-mean_vector1[1][0])
            cov_rb1 += (samples_data[i][0]-mean_vector1[0][0])*(samples_data[i][2]-mean_vector1[2][0])
            cov_gg1 += (samples_data[i][1]-mean_vector1[1][0])*(samples_data[i][1]-mean_vector1[1][0])
            cov_gb1 += (samples_data[i][1]-mean_vector1[1][0])*(samples_data[i][2]-mean_vector1[2][0])
            cov_bb1 += (samples_data[i][2]-mean_vector1[2][0])*(samples_data[i][2]-mean_vector1[2][0])
        elif i >= m+n:
            cov_rr2 += (samples_data[i][0] - mean_vector2[0][0]) * (samples_data[i][0] - mean_vector2[0][0])
            cov_rg2 += (samples_data[i][0] - mean_vector2[0][0]) * (samples_data[i][1] - mean_vector2[1][0])
            cov_rb2 += (samples_data[i][0] - mean_vector2[0][0]) * (samples_data[i][2] - mean_vector2[2][0])
            cov_gg2 += (samples_data[i][1] - mean_vector2[1][0]) * (samples_data[i][1] - mean_vector2[1][0])
            cov_gb2 += (samples_data[i][1] - mean_vector2[1][0]) * (samples_data[i][2] - mean_vector2[2][0])
            cov_bb2 += (samples_data[i][2] - mean_vector2[2][0]) * (samples_data[i][2] - mean_vector2[2][0])
        else:
            cov_rr0 += (samples_data[i][0] - mean_vector0[0][0]) * (samples_data[i][0] - mean_vector0[0][0])
            cov_rg0 += (samples_data[i][0] - mean_vector0[0][0]) * (samples_data[i][1] - mean_vector0[1][0])
            cov_rb0 += (samples_data[i][0] - mean_vector0[0][0]) * (samples_data[i][2] - mean_vector0[2][0])
            cov_gg0 += (samples_data[i][1] - mean_vector0[1][0]) * (samples_data[i][1] - mean_vector0[1][0])
            cov_gb0 += (samples_data[i][1] - mean_vector0[1][0]) * (samples_data[i][2] - mean_vector0[2][0])
            cov_bb0 += (samples_data[i][2] - mean_vector0[2][0]) * (samples_data[i][2] - mean_vector0[2][0])
 
    a = m-1
    b = n-1
    c = k-1
    Cov_1 = [[cov_rr1/a,cov_rg1/a,cov_rb1/a],[cov_rg1/a,cov_gg1/a,cov_gb1/a],[cov_rb1/a,cov_gb1/a,cov_bb1/a]]
    Cov_0 = [[cov_rr0/b,cov_rg0/b,cov_rb0/b],[cov_rg0/b,cov_gg0/b,cov_gb0/b],[cov_rb0/b,cov_gb0/b,cov_bb0/b]]
    Cov_2 = [[cov_rr2/c,cov_rg2/c,cov_rb2/c],[cov_rg2/c,cov_gg2/c,cov_gb2/c],[cov_rb2/c,cov_gb2/c,cov_bb2/c]]
    return Cov_1, Cov_0, Cov_2
 
# 计算矩阵的逆 行列式
def Get_Inverse(mean_vector1, mean_vector0, mean_vector2, Cov_1, Cov_0, Cov_2, test_data):
 
    Inv_1 = np.linalg.inv(Cov_1)  # 逆矩阵
    Inv_0 = np.linalg.inv(Cov_0)
    Inv_2 = np.linalg.inv(Cov_2)
    rows_1 = np.linalg.det(Cov_1)  #行列式
    rows_0 = np.linalg.det(Cov_0)
    rows_2 = np.linalg.det(Cov_2)
 
    L = len(test_data)
    a1 = [[0], [0], [0]]  # 存储 x - 均值向量的值
    a0 = [[0], [0], [0]]
    a2 = [[0], [0], [0]]
    b1 = [0, 0, 0]  # 转置
    b0 = [0, 0, 0]
    b2 = [0, 0, 0]
 
    for i in range(L):
        a1[i][0] = test_data[i][0] - mean_vector1[i][0]
        a0[i][0] = test_data[i][0] - mean_vector0[i][0]
        a2[i][0] = test_data[i][0] - mean_vector2[i][0]
        b1[i] = a1[i][0]   # 求 a 的转置
        b0[i] = a0[i][0]
        b2[i] = a2[i][0]
 
    #计算转置*逆矩阵*差
    m1 = np.dot(b1,Inv_1)
    M1 = np.dot(m1,a1)
    m0 = np.dot(b0,Inv_0)
    M0 = np.dot(m0,a0)
    m2 = np.dot(b2,Inv_2)
    M2 = np.dot(m2,a2)
 
    P1 = 0.5* (math.log(rows_1) + M1 + 3*math.log(2*math.pi))
    P0 = 0.5* (math.log(rows_0) + M0 + 3*math.log(2*math.pi))
    P2 = 0.5* (math.log(rows_2) + M2 + 3*math.log(2*math.pi))
    # print(P1)
    # print(P0)
    return P1, P0, P2
 
# 判断类别
def Get_Classify(P1, P0, P2):
    k = min(P1, P0, P2)
    if k == P1:
        print('该数据是杂草')
    elif k == P0:
        print('该数据是背景')
    elif k == P2:
        print('该数据是作物')
 
# 读取测试图片的RGB,把BGR->RGB
# 并把每个像元转为3行1列的向量
def Get_RGB(image):
    w = image.shape[0]
    h = image.shape[1]
    data = []
    ve = [[0] for i in range(3)]
    new_data = [ve for i in range(w*h)]
    for i in range(w):
        for j in range(h):
            for k in range(1):  # B G 调换
                a = image[i,j,k+0]
                image[i,j,k+0] = image[i,j,k+2]
                image[i,j,k+2] = a
    # print(image)
    for i in range(w):
        for j in range(h):
            new_data[i*h+j][0][0] = image[i][j][0]
            new_data[i*h+j][1][0] = image[i][j][1]
            new_data[i*h+j][2][0] = image[i][j][2]
            V = deepcopy(ve)
            data.append(V)
    return data
 
# 把图片的RGB传进来对每一个像素做分类 杂草赋值 [255,128,0] ,背景赋值 255， 作物赋值 [255,0,0]
def Get_Cla_Image(test_image,image):
    samples_data, class_lable = Dataset()
    mv1, mv0, mv2 = Get_Junzhi(samples_data, class_lable)
    cov1, cov0, cov2 = Get_Cov(samples_data, mv1, mv0, mv2)
 
    w = image.shape[0]
    h = image.shape[1]
    for i in range(w):
        for j in range(h):
            P1, P0 ,P2 = Get_Inverse(mv1, mv0, mv2, cov1, cov0, cov2, test_image[i*h+j])
            k = min(P1, P0, P2)
            if k == P1:
                image[i][j][0] = 0
                image[i][j][1] = 128
                image[i][j][2] = 255
            elif k == P0:
                image[i][j] = 255
            elif k == P2:
                image[i][j][0] = 0
                image[i][j][1] = 0
                image[i][j][2] = 255
    return image
 
image = cv2.imread('C:/Users/86130/Pictures/批注 2020-08-05 165113.png')
test_image = Get_RGB(image)
Cla_image = Get_Cla_Image(test_image,image)
cv2.imshow('Bayes Three', Cla_image)
cv2.waitKey(0)
cv2.destroyAllWindows()  

"""
test_data = [[129],[140],[146]]
samples_data, class_lable = Dataset()
mv1, mv0, mv2 = Get_Junzhi(samples_data,class_lable)
cov1, cov0, cov2 = Get_Cov(samples_data,mv1,mv0,mv2)
P1, P0, P2 = Get_Inverse(mv1, mv0, mv2, cov1, cov0, cov2, test_data)
Get_Classify(P1,P0,P2)

"""


# coding: utf-8

# # 1 矩阵运算
# 
# ## 1.1 创建一个 4*4 的单位矩阵

# In[34]:

# 这个项目设计来帮你熟悉 python list 和线性代数
# 你不能调用任何python库，包括NumPy，来完成作业
import math
import pprint

pp = pprint.PrettyPrinter(indent = 1,width=40)
test = [[1],[2],[3]]
A = [[1,2,3], 
     [2,3,3], 
     [1,2,5]]

B = [[1,2,3,5], 
     [2,3,3,5], 
     [1,2,5,1]]

#TODO 创建一个 4*4 单位矩阵
I = [[1.15362,2,3,4],
     [2,2.6346,3,4],
     [3,2,3,4],
     [4,2.1243,3.5342,4]]


# ## 1.2 返回矩阵的行数和列数

# In[35]:

# TODO 返回矩阵的行数和列数
def shape(M):
    row = len(M)
    col = len(M[0])
    return row,col

shape(test)


# ## 1.3 每个元素四舍五入到特定小数数位

# In[3]:

# TODO 每个元素四舍五入到特定小数数位
# 直接修改参数矩阵，无返回值

def matxRound(M, decPts=4):
    for l in range(len(M)):
        for i in range(len(M[l])):
            M[l][i] = round(M[l][i],decPts)
    pass

matxRound(I)


# ## 1.4 计算矩阵的转置

# In[4]:

# TODO 计算矩阵的转置
def transpose(M):
    return [[row[i] for row in M] for i in range(len(M[0]))]


# ## 1.5 计算矩阵乘法 AB

# In[5]:

# TODO 计算矩阵乘法 AB，如果无法相乘则返回None
def matxMultiply(A, B):
    if len(A[0]) == len(B):
        res = [[0] * len(B[0]) for i in range(len(A))]
        for i in range(len(A)):
            for j in range(len(B[0])):
                for k in range(len(B)):
                    res[i][j] += A[i][k] * B[k][j]
        return res
    else:
        return None


# ## 1.6 测试你的函数是否实现正确

# **提示：** 你可以用`from pprint import pprint`来更漂亮的打印数据，详见[用法示例](http://cn-static.udacity.com/mlnd/images/pprint.png)和[文档说明](https://docs.python.org/2/library/pprint.html#pprint.pprint)。

# In[6]:

#TODO 测试1.2 返回矩阵的行和列
print 'Row and Column in array are : ', shape(I)

#TODO 测试1.3 每个元素四舍五入到特定小数数位
matxRound(I)
pp.pprint(I)

#TODO 测试1.4 计算矩阵的转置
temp = transpose(I)
print 'Result for test 1.4 : '
pp.pprint(temp)

#TODO 测试1.5 计算矩阵乘法AB，AB无法相乘
print 'Test matrix multiply when not avalible : '
A = [[1,2,3,4], 
     [2,3,3,6], 
     [1,2,5,2]]

B = [[1,2,3,5], 
     [2,3,3,5], 
     [1,2,5,1]]
temp = matxMultiply(A,B)
print temp

#TODO 测试1.5 计算矩阵乘法AB，AB可以相乘
print 'Test matrix multiply when avalible : '
A = [[1,2,3], 
     [2,3,3], 
     [1,2,5]]

B = [[1,2,3,5], 
     [2,3,3,5], 
     [1,2,5,1]]

temp = matxMultiply(A,B)
print temp


# # 2 Gaussign Jordan 消元法
# 
# ## 2.1 构造增广矩阵
# 
# $ A = \begin{bmatrix}
#     a_{11}    & a_{12} & ... & a_{1n}\\
#     a_{21}    & a_{22} & ... & a_{2n}\\
#     a_{31}    & a_{22} & ... & a_{3n}\\
#     ...    & ... & ... & ...\\
#     a_{n1}    & a_{n2} & ... & a_{nn}\\
# \end{bmatrix} , b = \begin{bmatrix}
#     b_{1}  \\
#     b_{2}  \\
#     b_{3}  \\
#     ...    \\
#     b_{n}  \\
# \end{bmatrix}$
# 
# 返回 $ Ab = \begin{bmatrix}
#     a_{11}    & a_{12} & ... & a_{1n} & b_{1}\\
#     a_{21}    & a_{22} & ... & a_{2n} & b_{2}\\
#     a_{31}    & a_{22} & ... & a_{3n} & b_{3}\\
#     ...    & ... & ... & ...& ...\\
#     a_{n1}    & a_{n2} & ... & a_{nn} & b_{n} \end{bmatrix}$

# In[7]:

# TODO 构造增广矩阵，假设A，b行数相同
def augmentMatrix(A, b):
    if len(A) == len(b):
        temp = []
        for i in range(len(A)):
            A[i].extend(b[i])
        return A
    else:
        return 'Can not augment this two matrix'


# ## 2.2 初等行变换
# - 交换两行
# - 把某行乘以一个非零常数
# - 把某行加上另一行的若干倍：

# In[8]:

# TODO r1 <---> r2
# 直接修改参数矩阵，无返回值
def swapRows(M, r1, r2):
    trow = M[r1]
    M[r1] = M[r2]
    M[r2] = trow
    pass

# TODO r1 <--- r1 * scale， scale!=0
# 直接修改参数矩阵，无返回值
def scaleRow(M, r, scale):
    if scale != 0:
        M[r] = [M[r][i]*scale for i in range(len(M[r]))]
    else:
        raise ValueError
    pass

# TODO r1 <--- r1 + r2*scale
# 直接修改参数矩阵，无返回值
def addScaledRow(M, r1, r2, scale):
    M[r1] = [M[r1][i] + M[r2][i]*scale for i in range(len(M[r1]))]
    pass


# ## 2.3  Gaussian Jordan 消元法求解 Ax = b

# ### 提示：
# 
# 步骤1 检查A，b是否行数相同
# 
# 步骤2 构造增广矩阵Ab
# 
# 步骤3 逐列转换Ab为化简行阶梯形矩阵 [中文维基链接](https://zh.wikipedia.org/wiki/%E9%98%B6%E6%A2%AF%E5%BD%A2%E7%9F%A9%E9%98%B5#.E5.8C.96.E7.AE.80.E5.90.8E.E7.9A.84-.7Bzh-hans:.E8.A1.8C.3B_zh-hant:.E5.88.97.3B.7D-.E9.98.B6.E6.A2.AF.E5.BD.A2.E7.9F.A9.E9.98.B5)
#     
#     对于Ab的每一列（最后一列除外）
#         当前列为列c
#         寻找列c中 对角线以及对角线以下所有元素（行 c~N）的绝对值的最大值
#         如果绝对值最大值为0
#             那么A为奇异矩阵，返回None （请在问题2.4中证明该命题）
#         否则
#             使用第一个行变换，将绝对值最大值所在行交换到对角线元素所在行（行c） 
#             使用第二个行变换，将列c的对角线元素缩放为1
#             多次使用第三个行变换，将列c的其他元素消为0
#             
# 步骤4 返回Ab的最后一列
# 
# ### 注：
# 我们并没有按照常规方法先把矩阵转化为行阶梯形矩阵，再转换为化简行阶梯形矩阵，而是一步到位。如果你熟悉常规方法的话，可以思考一下两者的等价性。

# In[10]:

# TODO 实现 Gaussain Jordan 方法求解 Ax = b

""" Gaussian Jordan 方法求解 Ax = b.
    参数
        A: 方阵 
        b: 列向量
        decPts: 四舍五入位数，默认为4
        epsilon: 判读是否为0的阈值，默认 1.0e-16
        
    返回列向量 x 使得 Ax = b 
    返回None，如果 A，b 高度不同
    返回None，如果 A 为奇异矩阵
"""

def gj_Solve(A, b, decPts=4, eps = 1.0e-16):
    if len(A) == len(b):
        '''if A,b have same row, do result. otherwise return None'''
        Ab = augmentMatrix(A,b)
        (h, w) = (len(Ab), len(Ab[0]))
        #matxRound(Ab,decPts)
        i = 0
        while i < h:
            AbT = transpose(Ab)
            c = AbT[i]
            cy = [abs(ii) for ii in c[i:]]
            #print cy
            cy_max = max(cy)
            #print cy_max
            max_i = cy.index(cy_max) + i
            #print max_i
            if cy_max <= eps:
                return None
            else:
                swapRows(Ab,i,max_i) #交換所在行及最大值所在行
                scale = 1./Ab[i][i]  #計算scale
                scaleRow(Ab,i,scale) #將所在行的值變成1
                for j in range(len(Ab)): #將其他行的值歸零
                    if j != i:
                        addScaledRow(Ab,j,i, -1.*Ab[j][i]/Ab[i][i])
            i += 1
        N = transpose(Ab)[-1]
        return [[N[j]] for j in range(len(N))]
        #return Ab
    else:
        return None


# ## 2.4 证明下面的命题：
# 
# **如果方阵 A 可以被分为4个部分: ** 
# 
# $ A = \begin{bmatrix}
#     I    & X \\
#     Z    & Y \\
# \end{bmatrix} , \text{其中 I 为单位矩阵，Z 为全0矩阵，Y 的第一列全0}$，
# 
# **那么A为奇异矩阵。**
# 
# 提示：从多种角度都可以完成证明
# - 考虑矩阵 Y 和 矩阵 A 的秩
# - 考虑矩阵 Y 和 矩阵 A 的行列式
# - 考虑矩阵 A 的某一列是其他列的线性组合

# TODO 请使用 latex （请参照题目的 latex 写法学习）
# 
# TODO 证明：
# 
# 在$ A = \begin{bmatrix}
#     I    & X \\
#     Z    & Y \\
# \end{bmatrix} , \text{其中 I 为单位矩阵，Z 为全0矩阵，Y 的第一列全0}$，的情況下，
# 
# 此方陣Ａ可被表示為：
# 
# $A = \begin{bmatrix}
#     1 & 0 & 0 & ... & x_{11} & ... & x_{1n}\\
#     0 & 1 & 0 & ... & x_{21} & ... & x_{2n}\\
#     0 & 0 & 1 & ... & x_{31} & ... & x_{3n}\\
#     ... & ... & ... & ... & ... & ... & ...\\
#     0 & 0 & 0 & 1 & x_{1n} & ... & x_{nn}  \\
#     0 & 0 & 0 & 0 &  0     & ... &  0      \\
#     0 & 0 & 0 & 0 & y_{21} & ... & x_{2n}  \\
#     0 & 0 & 0 & 0 & ...    & ... & ...     \\
#     0 & 0 & 0 & 0 & y_{n1} & ... & y_{nn} \end{bmatrix}$
# 
# 由奇異矩陣的定義可知一方陣是否為奇異矩陣可由此方陣之行列式的絕對值是否為0來判斷，
# 
# 方陣Ａ由於Ｚ為全零矩陣 且 Ｙ矩陣的第一列全為零：因此在方陣Ａ的行列式必為0。
# 
# 可知方陣Ａ必為奇異矩陣。
# 

# ## 2.5 测试 gj_Solve() 实现是否正确

# In[11]:

# TODO 构造 矩阵A，列向量b，其中 A 为奇异矩阵
matrix1 = [  
    [0, 6, -1],  
    [0, 8, 3],  
    [0, 4, 1],  
]  
matrix2 = [  
    [1],  
    [0],
    [0]
]  

test1 = augmentMatrix(matrix1,matrix2)
print 'Matrix Ab of A is 奇异矩阵: ',test1

# TODO 构造 矩阵A，列向量b，其中 A 为非奇异矩阵
matrix3 = [  
    [4, 6, -1],  
    [5, -8, 3],  
    [1, 4, 1],  
]  
matrixA = [  
    [4, 6, -1],  
    [5, -8, 3],  
    [1, 4, 1],  
]  
matrix2 = [  
    [1],  
    [0],
    [0]
]  

test2 = augmentMatrix(matrix3,matrix2)
print 'Matrix Ab of A is 非奇异矩阵: ',test2
# TODO 求解 x 使得 Ax = b
x = gj_Solve(matrix3,matrix2)
print 'For Ax = b, x = ',x
# TODO 计算 Ax
a = matxMultiply(matrixA, x)
matxRound(a)
print 'Ax = ',a
# TODO 比较 Ax 与 b
print 'b = ',matrix2
print 'Ax = b'


# # 3 线性回归: 
# 
# ## 3.1 计算损失函数相对于参数的导数 (两个3.1 选做其一)
# 
# 我们定义损失函数 E ：
# $$
# E = \sum_{i=1}^{n}{(y_i - mx_i - b)^2}
# $$
# 
# 证明：
# $$
# \frac{\partial E}{\partial m} = \sum_{i=1}^{n}{-2x_i(y_i - mx_i - b)}
# $$
# 
# $$
# \frac{\partial E}{\partial b} = \sum_{i=1}^{n}{-2(y_i - mx_i - b)}
# $$
# 
# $$
# \begin{bmatrix}
#     \frac{\partial E}{\partial m} \\
#     \frac{\partial E}{\partial b} 
# \end{bmatrix} = 2X^TXh - 2X^TY
# $$
# 
# $$ 
# \text{其中 }
# Y =  \begin{bmatrix}
#     y_1 \\
#     y_2 \\
#     ... \\
#     y_n
# \end{bmatrix}
# ,
# X =  \begin{bmatrix}
#     x_1 & 1 \\
#     x_2 & 1\\
#     ... & ...\\
#     x_n & 1 \\
# \end{bmatrix},
# h =  \begin{bmatrix}
#     m \\
#     b \\
# \end{bmatrix}
# $$

# TODO 请使用 latex （参照题目的 latex写法学习）
# 
# TODO 证明：
# $$
# 定義損失函數 E = \sum_{i=1}^{n}{(y_i - mx_i - b)^2}
# $$
# 
# 
# 將此方程式展開後可得到
# $$
# E = \sum_{i=1}^{n}{(y_i^2 + (mx_i)^2 + b^2 - 2y_imx_i - 2y_ib + 2mx_ib)} \\
#   = \sum_{i=1}^{n}{(y_i^2 - 2y_imx_i - 2y_ib + (mx_i)^2 + 2mx_ib + b^2)} \\
#   = \sum_{i=1}^{n}{(y_i^2 - 2y_i(mx_i + b) + (mx_i + b)^2)} \\
#   = \sum_{i=1}^{n}{(y_i - (mx_i + b))^2}
# $$
# 
# 再將此方程式求導後可得到
# 
# $$
# \frac{\partial E}{\partial m} = \sum_{i=1}^{n}{-2x_i(y_i - mx_i - b)}
# $$
# 以及
# 
# $$
# \frac{\partial E}{\partial b} = \sum_{i=1}^{n}{-2(y_i - mx_i - b)}
# $$
# 則
# 
# $$
# \begin{bmatrix}
#     \frac{\partial E}{\partial m} \\
#     \frac{\partial E}{\partial b} 
# \end{bmatrix} = 
# \begin{bmatrix}
#     \sum_{i=1}^{n}{-2x_i(y_i - mx_i - b)} \\
#     \sum_{i=1}^{n}{-2(y_i - mx_i - b)}
# \end{bmatrix} \\
# =
# 2\begin{bmatrix}
#     \sum_{i=1}^{n}{(mx_i^2+x_ib-x_iy_i)} \\
#     \sum_{i=1}^{n}{(mx_i+b-y_i)}
# \end{bmatrix} \\
# =
# 2
# \begin{bmatrix}
#     \sum_{i=1}^{n}{(mx_i^2+x_ib)} \\
#     \sum_{i=1}^{n}{(mx_i+b)}
# \end{bmatrix}
# -2
# \begin{bmatrix}
#     \sum_{i=1}^{n}{(x_iy_i)} \\
#     \sum_{i=1}^{n}{(y_i)}
# \end{bmatrix} \\
# =
# 2\sum_{i=1}^{n}
# \begin{bmatrix}
#     {x_i^2} & {x_i} \\
#     {x_i}   & {1}
# \end{bmatrix}
# \begin{bmatrix}
#     {m} \\
#     {b}
# \end{bmatrix}
# -2\sum_{i=1}^{n}
# \begin{bmatrix}
#     {x_i} \\
#     {1}
# \end{bmatrix}
# \begin{bmatrix}
#     {y_i} \\
#     0
# \end{bmatrix}\\
# 將此矩陣展開 \\
# =2
# \begin{bmatrix}
#     x_1 & 1 \\
#     x_2 & 1\\
#     ... & ...\\
#     x_n & 1 \\
# \end{bmatrix}
# \begin{bmatrix}
#     x_1 & x_2 & ... & x_n \\
#     1   &  1  & ... &  1  \\
# \end{bmatrix}
# \begin{bmatrix}
#     {m} \\
#     {b}
# \end{bmatrix}
# -2
# \begin{bmatrix}
#     x_1 & x_2 & ... & x_n \\
#     1   &  1  & ... &  1  \\
# \end{bmatrix}
# \begin{bmatrix}
#     y_1 \\
#     y_2 \\
#     ... \\
#     y_n \\
# \end{bmatrix}
# $$
# 
# 故可得到下列方程式
# $$
# \begin{bmatrix}
#     \frac{\partial E}{\partial m} \\
#     \frac{\partial E}{\partial b} 
# \end{bmatrix} = 2X^TXh - 2X^TY
# $$
# 
# $$ 
# \text{其中 }
# Y =  \begin{bmatrix}
#     y_1 \\
#     y_2 \\
#     ... \\
#     y_n
# \end{bmatrix}
# ,
# X =  \begin{bmatrix}
#     x_1 & 1 \\
#     x_2 & 1\\
#     ... & ...\\
#     x_n & 1 \\
# \end{bmatrix},
# h =  \begin{bmatrix}
#     m \\
#     b \\
# \end{bmatrix}
# $$

# ## 3.1 计算损失函数相对于参数的导数（两个3.1 选做其一）
# 
# 证明：
# 
# $$
# E = Y^TY -2(Xh)^TY + (Xh)^TXh
# $$ 
# 
# $$
# \begin{bmatrix}
#     \frac{\partial E}{\partial m} \\
#     \frac{\partial E}{\partial b} 
# \end{bmatrix}  = \frac{\partial E}{\partial h} = 2X^TXh - 2X^TY
# $$
# 
# $$ 
# \text{其中 }
# Y =  \begin{bmatrix}
#     y_1 \\
#     y_2 \\
#     ... \\
#     y_n
# \end{bmatrix}
# ,
# X =  \begin{bmatrix}
#     x_1 & 1 \\
#     x_2 & 1\\
#     ... & ...\\
#     x_n & 1 \\
# \end{bmatrix},
# h =  \begin{bmatrix}
#     m \\
#     b \\
# \end{bmatrix}
# $$

# TODO 请使用 latex （请参照题目的 latex 写法学习）
# 
# TODO 证明：
# 

# ## 3.2  线性回归
# 
# ### 求解方程 $X^TXh = X^TY $, 计算线性回归的最佳参数 h

# In[48]:

def inverse_matrix(matrix):
    dimension=len(matrix)
    diagnoal=creat_diagnal_matrix(dimension)
    "This loop control every diagnal number must be dived as 1"
    for d in range(dimension):
        "temp store the information which every row of numbers should be devided"
        temp=matrix[d][d]
        for c in range(dimension):
            matrix[d][c]=matrix[d][c]/temp
            diagnoal[d][c]=diagnoal[d][c]/temp
        #print 'process matrix:\n',matrix
        #print 'process diagnoal:\n',diagnoal        
        "This loop control every row should mutiply a number(mu) and add to another row"
        for r in range(dimension):
            mu=-matrix[r][d]
            "But one row which we dived as 1 at the first time should be escaped"
            if(r!=d):
                for c in range(dimension):
                    matrix[r][c]=matrix[d][c]*mu+matrix[r][c]
                    diagnoal[r][c]=diagnoal[d][c]*mu+diagnoal[r][c]
                #print mu
                #print 'procee\n',matrix
    return diagnoal


# In[71]:

# TODO 实现线性回归
'''
参数：(x,y) 二元组列表
返回：m，b
'''
import numpy as np

def linearRegression(points):
    xtx = matxMultiply(transpose(points[0]),points[0])
    xty = matxMultiply(transpose(points[0]),points[1])
    h = matxMultiply(np.linalg.inv(xtx),xty)
    return h[0],h[1]


# In[72]:

a = [[1,1],[5,1],[2,1],[3,1],[5,1]]
b = [[7],[5],[4],[3],[5]]
t1 = [a,b]
print t1
temp = linearRegression(t1)
print temp


# ## 3.3 测试你的线性回归实现

# In[139]:

# TODO 构造线性函数
#y = 4x - 100
# TODO 构造 100 个线性函数上的点，加上适当的高斯噪音
import random
import numpy as np
mx = []
my = []
i = 0
for i in range(100):
    x = i
    y = 4*i - 100
    mx.append(x)
    my.append(y)
#print mx,my
mx2,my2=mx,my
for i in range(100):
    mx2[i] = mx[i] + random.gauss(np.mean(mx),np.std(mx))
    my2[i] = my[i] + random.gauss(np.mean(my),np.std(my))
#print 'mx2=',mx2,'my2=',my2
#TODO 对这100个点进行线性回归，将线性回归得到的函数和原线性函数比较
mx3 = []
my3 = []
for i in range(100):
    mx3.append([mx2[i],1])
    my3.append([my2[i]])
result = linearRegression([mx3,my3])
print '原設定線性函數為([4],[-100]), 100個亂數點進行回歸後得到的函數為：',result


# ## 4.1 单元测试
# 
# 请确保你的实现通过了以下所有单元测试。

# In[12]:

import unittest
import numpy as np

from decimal import *

class LinearRegressionTestCase(unittest.TestCase):
    """Test for linear regression project"""

    def test_shape(self):

        for _ in range(10):
            r,c = np.random.randint(low=1,high=25,size=2)
            matrix = np.random.randint(low=-10,high=10,size=(r,c))
            self.assertEqual(shape(matrix.tolist()),(r,c))


    def test_matxRound(self):

        for decpts in range(10):
            r,c = np.random.randint(low=1,high=25,size=2)
            matrix = np.random.random((r,c))

            mat = matrix.tolist()
            dec_true = [[Decimal(str(round(num,decpts))) for num in row] for row in mat]

            matxRound(mat,decpts)
            dec_test = [[Decimal(str(num)) for num in row] for row in mat]

            res = Decimal('0')
            for i in range(len(mat)):
                for j in range(len(mat[0])):
                    res += dec_test[i][j].compare_total(dec_true[i][j])

            self.assertEqual(res,Decimal('0'))


    def test_transpose(self):
        for _ in range(10):
            r,c = np.random.randint(low=1,high=25,size=2)
            matrix = np.random.random((r,c))

            mat = matrix.tolist()
            t = np.array(transpose(mat))

            self.assertEqual(t.shape,(c,r))
            self.assertTrue((matrix.T == t).all())


    def test_matxMultiply(self):

        for _ in range(10):
            r,d,c = np.random.randint(low=1,high=25,size=3)
            mat1 = np.random.randint(low=-10,high=10,size=(r,d)) 
            mat2 = np.random.randint(low=-5,high=5,size=(d,c)) 
            dotProduct = np.dot(mat1,mat2)

            dp = np.array(matxMultiply(mat1,mat2))

            self.assertTrue((dotProduct == dp).all())


    def test_augmentMatrix(self):

        for _ in range(10):
            r,c = np.random.randint(low=1,high=25,size=2)
            A = np.random.randint(low=-10,high=10,size=(r,c))
            b = np.random.randint(low=-10,high=10,size=(r,1))

            Ab = np.array(augmentMatrix(A.tolist(),b.tolist()))
            ab = np.hstack((A,b))

            self.assertTrue((Ab == ab).all())

    def test_swapRows(self):
        for _ in range(10):
            r,c = np.random.randint(low=1,high=25,size=2)
            matrix = np.random.random((r,c))

            mat = matrix.tolist()

            r1, r2 = np.random.randint(0,r, size = 2)
            swapRows(mat,r1,r2)

            matrix[[r1,r2]] = matrix[[r2,r1]]

            self.assertTrue((matrix == np.array(mat)).all())

    def test_scaleRow(self):

        for _ in range(10):
            r,c = np.random.randint(low=1,high=25,size=2)
            matrix = np.random.random((r,c))

            mat = matrix.tolist()

            rr = np.random.randint(0,r)
            with self.assertRaises(ValueError):
                scaleRow(mat,rr,0)

            scale = np.random.randint(low=1,high=10)
            scaleRow(mat,rr,scale)
            matrix[rr] *= scale

            self.assertTrue((matrix == np.array(mat)).all())
    
    def test_addScaleRow(self):

        for _ in range(10):
            r,c = np.random.randint(low=1,high=25,size=2)
            matrix = np.random.random((r,c))

            mat = matrix.tolist()

            r1,r2 = np.random.randint(0,r,size=2)

            scale = np.random.randint(low=1,high=10)
            addScaledRow(mat,r1,r2,scale)
            matrix[r1] += scale * matrix[r2]

            self.assertTrue((matrix == np.array(mat)).all())


    def test_gj_Solve(self):

        for _ in range(10):
            r = np.random.randint(low=3,high=10)
            A = np.random.randint(low=-10,high=10,size=(r,r))
            b = np.arange(r).reshape((r,1))
            x = gj_Solve(A.tolist(),b.tolist())
            if np.linalg.matrix_rank(A) < r:
                self.assertEqual(x,None)
            else:
                # Ax = matxMultiply(A.tolist(),x)
                Ax = np.dot(A,np.array(x))
                loss = np.mean((Ax - b)**2)
                # print Ax
                # print loss
                self.assertTrue(loss<0.1)


suite = unittest.TestLoader().loadTestsFromTestCase(LinearRegressionTestCase)
unittest.TextTestRunner(verbosity=3).run(suite)


# coding=utf-8

# 权值线性回归

import random
from matrix import Matrix
import time
import math
# 线性回归
def weight_linear_regression(data, flavor_list):
    raw = Matrix(data)
    alpha = 0.5
    ntheta = 5
    if raw.rows() < ntheta:
        ntheta = raw.rows()
    print('theta is %s' % ntheta)
    thetas = []
    remains = []
    for index in range(len(flavor_list)):
        n = int(flavor_list[index].name.split('flavor')[1])
        # print 'linear regression for flavor%s' % n
        theta = []
        weights = []
        for i in range(ntheta+1):
            theta.append(random.uniform(-1,1))
            numerator = (-1)*(i-ntheta)*(i-ntheta)
            denominator = 2*ntheta*ntheta
            s = float(numerator)/float(denominator)
            weights.append(math.exp(s))
            # theta.append(0)
        theta = Matrix(theta)
        theta.transposition()
        X = []
        for i in range(ntheta, len(data)):
            xline = raw.get_col(n)[i - ntheta:i]
            xline.append(1)
            X.append(xline)
        X = Matrix(X)
        Y = Matrix(raw.get_col(n)[ntheta:])
        Y.transposition()
        # 把最后一组数据存入
        remains.append(xline)
        thetas.append(gradient(X, Y, theta, alpha, weights))
    return thetas, remains


# 梯度下降
def gradient(X, Y, theta, alpha,weight):
    count = X.rows()
    n = theta.rows()
    lastJ = 0
    weight = Matrix(weight)
    weight.transposition()
    for i in range(500):
        temptheta = theta.dotmul(weight)
        for j in range(n):
            H = X * temptheta
            SUB =  H - Y
            SX = Matrix(X.get_col(j))
            SX.transposition()
            Z = SUB.dotmul(SX)
            SUM = 0
            for i in range(Z.rows()):
                SUM = SUM + Z.get_row(i)[0]
            theta.matrix[j][0] = theta.matrix[j][0] - alpha/count*SUM
        J = cost(X,Y,theta)
        if lastJ!=0 and lastJ<J:
            alpha = alpha/5
        if abs(lastJ-J) < 0.00001:
            break
        else:
            lastJ = J
    return theta


# 误差计算
def cost(X, Y, theta):
    count = X.rows()
    prediction = X * theta
    error = (prediction - Y).dotmul(prediction - Y)
    SUM = 0
    for i in range(error.rows()):
        SUM = SUM + error.get_row(i)[0]
    J = SUM / (2 * count)
    print('cost is %s' % J)
    return J


# 预测对象虚拟机的数量
def predic(thetas, remains, last_time, start_time, end_time):
    st = time.strptime(last_time + ' 0:00:00', '%Y-%m-%d %H:%M:%S')
    last_time = time.mktime(st)
    st = time.strptime(start_time + ' 0:00:00', '%Y-%m-%d %H:%M:%S')
    start_time = time.mktime(st)
    st = time.strptime(end_time + ' 0:00:00', '%Y-%m-%d %H:%M:%S')
    end_time = time.mktime(st)
    big_period = int((end_time - last_time) / 86400)
    days = int((end_time - start_time) / 86400)
    nums = []
    for i in range(len(thetas)):
        theta = thetas[i]
        # 去除最后一个参数1
        remains[i].pop()
        X = remains[i]
        number = []
        for d in range(big_period):
            X.append(1)
            TX = Matrix(X)
            R = TX * theta
            # 要考虑到count是无限大和无限小的情况
            count = int(round(R.get_row(0)[0]))
            if count < 0:
                count = 0
            number.append(count)
            X.pop()
            X.append(count)
            X.pop(0)
        nums.append(number)
    final = []
    for i in range(len(nums)):
        n = nums[i][len(nums[i]) - days:len(nums[i])]
        # if sum(n)>20:
        #     final.append(20)
        # else:
        final.append(sum(n))
    return final
# coding=utf-8

# 线性回归预测时间序列的累计数量
y = ax+b

import random
from matrix import Matrix
import time
# 线性回归
def linear_regression(data, flavor_list):
    raw = Matrix(data)
    alpha = 0.12
    ntheta = 5
    thetas = []
    remains = []
    for index in range(len(flavor_list)):
        n = int(flavor_list[index].name.split('flavor')[1])
        print('linear regression for flavor%s' % n)
        theta = []
        for i in range(ntheta+1):
            theta.append(random.uniform(0,1))
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
        thetas.append(gradient(X, Y, theta, alpha))
    return thetas, remains


# 梯度下降
def gradient(X, Y, theta, alpha):
    count = X.rows()
    n = theta.rows()
    lastJ = 0
    for i in range(200):
        temptheta = theta
        for j in range(n):
            MUL = X * temptheta
            SUB = MUL - Y
            SX = Matrix(X.get_col(j))
            SX.transposition()
            Z = SUB.dotmul(SX)
            SUM = 0
            for i in range(Z.rows()):
                SUM = SUM + Z.get_row(i)[0]
            theta.matrix[j][0] = theta.matrix[j][0] - alpha/count*SUM
        J = cost(X,Y,theta)
        if lastJ!=0 and lastJ<J:
            alpha = alpha/10
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
    # print 'cost is %s' % J
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
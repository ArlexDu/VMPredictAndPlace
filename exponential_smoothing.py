# coding=utf-8
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
import math
import numpy as np
import copy


# 一次指数平滑
def single(data, cal_mse):
    # 一次去先平滑的结果
    S1 = []
    # 初始值
    initial = []
    # 设置初始值
    for m in range(len(data)):
        x = 0
        for n in range(3):
            x = x + float(data[m][n])
        x = x / 3
        initial.append(x)
        S1.append([])
    # 平滑参数
    alpha = []
    # 误差
    MSE = []
    # 设置平滑参数
    for i in range(len(data)):
        alpha.append(0.6)
    # 一次平滑
    for i in range(len(data)):
        mse = 0
        for j in range(len(data[i])):
            if j == 0:
                smooth1 = float(alpha[i] * int(data[i][j]) + (1 - alpha[i]) * int(initial[i]))
            else:
                smooth1 = float(alpha[i] * int(data[i][j]) + (1 - alpha[i]) * float(S1[i][j - 1]))
            S1[i].append(smooth1)
            if cal_mse:
                mse = int(S1[i][j] - int(data[i][j])) ** 2 + mse
        if cal_mse:
            mse = mse ** (1 / 2) / int(len(data[i]))
            MSE.append(mse)
    return initial, alpha, S1


# 二次指数平滑 referred to as either Brown's linear exponential smoothing (LES) or Brown's double exponential smoothing
def linear_double(data, alpha):
    results = []
    # 一次去先平滑的结果
    S1 = []
    # 二次平滑的结果
    S2 = []
    # 设置初始值
    S1.append(data[0])
    S2.append(data[0])
    results.append(data[0])
    # 二次平滑
    for i in range(len(data)):
        smooth1 = float(alpha * int(data[i]) + (1 - alpha) * float(S1[-1]))
        smooth2 = float(alpha * float(smooth1) + (1 - alpha) * float(S2[-1]))
        S1.append(smooth1)
        S2.append(smooth2)
        result = float(S1[i]) * 2 - float(S2[i])
        results.append(result)
    return S1, S2, results


# 二次指数平滑 referred to as "Holt-Winters double exponential smoothing"
def holt_winters_double(data, alpha, beta):
    results = []
    # 一次指数平滑结果
    levels = []
    # 二次指数平滑结果
    trends = []
    # 默认的第一个level是数据集的第一个值
    levels.append(data[0])
    # 默认的第一个trend是数据集的前两个点的差值
    trends.append(data[1] - data[0])
    # 默认的第一个result是数据集的第一个值
    results.append(data[0])
    # 二次平滑
    for j in range(1, len(data)):
        level = alpha * data[j] + (1 - alpha) * (levels[-1] + trends[-1])
        level = max(level, levels[-1])
        trend = beta * (level - levels[-1]) + (1 - beta) * trends[-1]
        levels.append(level)
        trends.append(trend)
        result = level + trend * 0
        results.append(result)
    return levels, trends, results


# 外部调用预测接口
def predict(data, last_time, start_time, end_time):
    st = time.strptime(last_time + ' 0:00:00', '%Y-%m-%d %H:%M:%S')
    last_time = time.mktime(st) + 86400
    st = time.strptime(start_time + ' 0:00:00', '%Y-%m-%d %H:%M:%S')
    start_time = time.mktime(st)
    st = time.strptime(end_time + ' 0:00:00', '%Y-%m-%d %H:%M:%S')
    end_time = time.mktime(st)
    # 从训练集的最后一个日期到预测区间的结束日期
    big_period = int((end_time - last_time) / 86400)
    # 从训练集的最后一个日期到预测区间的开始日期
    small_period = int((start_time - last_time) / 86400)
    numbers = []
    # 线性二次平滑
    # alpha = []
    # for i in range(len(data)):
    #     alpha.append(0.5)
    # 线性平滑最优参数
    # alpha = linear_param_optimize(data, int((end_time - start_time) / 86400))
    # for i in range(len(data)):
    #     S1, S2, result = linear_double(data[i], alpha[i])
    #     number, values = linear_predict(data[i], S1, S2, alpha[i], big_period, small_period)
    #     numbers.append(number)
    # holt-winter二次平滑
    alpha, beta = holt_winters_param_optimize(data, int((end_time - start_time) / 86400))
    for i in range(len(data)):
        level, trend, result = holt_winters_double(data[i], alpha[i], beta[i])
        list, number = holt_winters_accumulative_predict(data[i], alpha[i], beta[i], level, trend, big_period,
                                                         small_period)
        numbers.append(number)
    return numbers


# 线性二次平滑预测
def linear_predict(data, S1, S2, alpha, big_period, small_period):
    # 直接相乘的方法
    # for i in range(len(data)):
    #     at = 2 * S1[i][-1] - S2[i][-1]
    #     bt = alpha[i] / (1 - alpha[i])
    #     number1 = at + bt * big_period
    #     number2 = at + bt * (small_period-1)
    #     numbers1.append(int(round(number1 - number2)))
    # 递推的方法
    sum = []
    sum.append(data[-1])
    for i in range(big_period):
        At = float(S1[-1]) * 2 - float(S2[-1])
        Bt = alpha / (1 - alpha) * (float(S1[-1]) - float(S2[-1]))
        y = At + Bt
        sum.append(y)
        new_s1 = alpha * y + (1 - alpha) * S1[-1]
        new_s2 = alpha * new_s1 + (1 - alpha) * S2[-1]
        S1.append(new_s1)
        S2.append(new_s2)
    number1 = sum[-1]
    number2 = sum[small_period]
    number = int(round(number1 - number2))
    sum.pop(0)
    return number, sum


# 二次平滑预测直接相乘法
def holt_winters_direct_predict(data, alpha, beta, levels, trends, big_period, small_period):
    # 直接相乘法
    level = alpha * data[-1] + (1 - alpha) * (levels[-1] + trends[-1])
    trend = beta * (level - levels[-1]) + (1 - beta) * trends[-1]
    number1 = level + trend * big_period
    if small_period == 0:
        number2 = data[-1]
    else:
        number2 = level + trend * small_period
    number = int(round(number1 - number2))
    return number1, number2


# 二次平滑预测累积法
def holt_winters_accumulative_predict(data, alpha, beta, levels, trends, big_period, small_period):
    # 递推的方法
    values = []
    values.append(data[-1])
    init = levels[-1]
    for j in range(big_period):
        level = alpha * values[-1] + (1 - alpha) * (levels[-1] + trends[-1])
        level = max(level, levels[-1])
        trend = beta * (level - levels[-1]) + (1 - beta) * trends[-1]
        levels.append(level)
        trends.append(trend)
        result = level + trend
        values.append(result)
    number1 = values[big_period]
    if small_period == 0:
        number2 = init
    else:
        number2 = values[small_period]
    number = int(math.ceil(number1 - number2))
    values.pop(0)
    return values, number


# 优化holt winters 参数模型 选取训练数据集中的前100个（如果样本个数多余100的话）作为训练样本优化参数模型
def holt_winters_param_optimize(data, validate_num):
    # 训练数量 90 测试数量 10
    train_num = 365
    if len(data[0]) < (train_num + validate_num):
        train_num = len(data[0]) - validate_num
    params = []
    raw = range(5, 105, 5)
    for i in raw:
        params.append(float(i) / 100)
    alpha = []
    beta = []
    bmse = []
    for i in range(len(data)):
        mses = []
        train_data = data[i][0:train_num]
        # target_number = data[i][train_num + validate_num - 1] - data[i][train_num - 1]
        target_data = data[i][train_num:train_num + validate_num]
        besta = 0.6
        bestb = 0.6
        bestMSE = float('inf')
        for b in params:
            m = []
            for a in params:
                # b = a
                level, trend, result = holt_winters_double(train_data, a, b)
                prds, number = holt_winters_accumulative_predict(train_data, a, b, level, trend, validate_num, 0)
                mse = 0
                for n in range(len(target_data)):
                    mse = (target_data[n] - prds[n]) ** 2 + mse
                mse = mse / validate_num
                m.append(mse)
                if mse < bestMSE:
                    besta = a
                    bestb = b
                    bestMSE = mse
            mses.append(m)
        alpha.append(besta)
        beta.append(bestb)
        bmse.append(bestMSE)
        # drawAlphaBetaDigram('Flavor ' + str(i+1)+' Parameters Determination', mses)
    # print('alpha:')
    # for i in alpha:
    #     print(str(i)+' ')
    # print('beta:')
    # for i in beta:
    #     print(str(i)+' ')
    # print('mse:')
    # for i in bmse:
    #     print(str(i)+' ')
    return alpha, beta


def drawAlphaBetaDigram(title, mses):
    fig = plt.figure()
    ax = Axes3D(fig)
    X = np.arange(0, 1, 0.05)
    Y = np.arange(0, 1, 0.05)
    X, Y = np.meshgrid(X, Y)  # x-y 平面的网格
    Z = np.array(mses)
    # height valu
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=plt.get_cmap('rainbow'))
    plt.title(title)
    ax.set_zlabel('MSE')  # 坐标轴
    ax.set_ylabel('Beta')
    ax.set_xlabel('Alpha')
    plt.show()


# 优化linear 参数模型 选取训练数据集中的前100个（如果样本个数多余100的话）作为训练样本优化参数模型
def linear_param_optimize(data, validate_num):
    # 训练数量 90 测试数量 10
    train_num = 90
    if len(data[0]) < (train_num + validate_num):
        train_num = len(data[0]) - validate_num
    params = range(1, 10, 1)
    params = map(lambda x: float(x) / 10, params)
    alpha = []
    for i in range(len(data)):
        train_data = data[i][0:train_num]
        target_data = data[i][train_num:(train_num + validate_num)]
        besta = 0
        bestSSE = float('inf')
        for a in params:
            S1, S2, result = linear_double(train_data, a)
            number, numbers = linear_predict(train_data, S1, S2, a, validate_num, 0)
            mse = 0
            for i in range(len(target_data)):
                mse = (target_data[i] - numbers[i]) ** 2 + mse
            if mse < bestSSE:
                besta = a
                bestSSE = mse
        alpha.append(besta)
    return alpha


# 外部调用预测接口
def predict_draw(data, test_data, last_time, start_time, end_time):
    st = time.strptime(last_time + ' 0:00:00', '%Y-%m-%d %H:%M:%S')
    last_time = time.mktime(st) + 86400
    st = time.strptime(start_time + ' 0:00:00', '%Y-%m-%d %H:%M:%S')
    start_time = time.mktime(st)
    st = time.strptime(end_time + ' 0:00:00', '%Y-%m-%d %H:%M:%S')
    end_time = time.mktime(st)
    # 从训练集的最后一个日期到预测区间的结束日期
    big_period = int((end_time - last_time) / 86400)
    # 从训练集的最后一个日期到预测区间的开始日期
    small_period = int((start_time - last_time) / 86400)
    numbers = []
    # 线性二次平滑
    # alpha = []
    # for i in range(len(data)):
    #     alpha.append(0.5)
    # 线性平滑最优参数
    # alpha = linear_param_optimize(data, int((end_time - start_time) / 86400))
    # for i in range(len(data)):
    #     S1, S2, result = linear_double(data[i], alpha[i])
    #     number, values = linear_predict(data[i], S1, S2, alpha[i], big_period, small_period)
    #     numbers.append(number)
    # holt-winter二次平滑
    day = 7
    # alpha, beta = holt_winters_param_optimize(data, int((end_time - start_time) / 86400))
    alpha, beta = holt_winters_param_optimize(data, day)
    # alpha, beta = holt_winters_param_optimize(data, 1)
    predict_mse = []
    for i in range(len(data)):
        level, trend, result = holt_winters_double(data[i], alpha[i], beta[i])
        # number1, number2 = holt_winters_direct_predict(data[i][:-day], alpha[i], beta[i], level, trend, day, 0)
        list, number = holt_winters_accumulative_predict(data[i], alpha[i], beta[i], level, trend, day,
                                                         0)
        mse = 0
        for k in range(day):
            mse = mse + ((list[k] - test_data[i][k]) / test_data[i][k]) ** 2
        predict_mse.append(mse / day)
        # numbers.append(number)
        # 画图
        # x = [i for i in range(len(data[i]))]
        # coor2 = []
        # for k in range(len(data[i]), len(data[i]) + len(list), 1):
        #     coor2.append(k)
        #
        # ax = plt.subplot()
        # ax.set_title(' Flavor '+str(i+1)+' Predict Diagram')  # give plot a title
        # ax.set_xlabel('Date')  # make axis labels
        # ax.set_ylabel('Accumulative Total')
        # # ax.set_ylim([0, 20])
        # # ax.set_xticklabels(x)
        #
        # plt.plot(x, data[i], label='origin')
        # plt.plot(x, result, label='matching')
        # plt.plot(coor2, list, label='predict')
        # plt.plot(coor2, test_data[i], label='test')
        # # plt.plot(coor3, [number2, number1], label='predict direct')
        #
        # plt.legend(loc='upper left')
        # plt.show()
    print('predict mape:')
    sum = 0
    for i in predict_mse:
        sum = sum + i
        print(str(i) + ' ')
    avg = sum / len(predict_mse)
    ax = plt.subplot()
    ax.set_title('Final MAPE')  # give plot a title
    ax.set_xlabel('Flavor')  # make axis labels
    ax.set_ylabel('MAPE')
    # plt.axis([0, 60, 0, 30])

    plt.plot([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], predict_mse, label='MAPE')
    plt.plot([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
             [avg, avg, avg, avg, avg, avg, avg, avg, avg, avg, avg, avg, avg, avg, avg], '--', label='Average MAPE')
    plt.legend(loc='upper left')

    plt.show()
    return numbers


# 外部调用预测接口
def predict_all(data, last_time, start_time, end_time):
    st = time.strptime(last_time + ' 0:00:00', '%Y-%m-%d %H:%M:%S')
    last_time = time.mktime(st) + 86400
    st = time.strptime(start_time + ' 0:00:00', '%Y-%m-%d %H:%M:%S')
    start_time = time.mktime(st)
    st = time.strptime(end_time + ' 0:00:00', '%Y-%m-%d %H:%M:%S')
    end_time = time.mktime(st)
    # 从训练集的最后一个日期到预测区间的结束日期
    big_period = int((end_time - last_time) / 86400)
    # 从训练集的最后一个日期到预测区间的开始日期
    small_period = int((start_time - last_time) / 86400)
    results = []
    # 线性二次平滑
    # alpha = []
    # for i in range(len(data)):
    #     alpha.append(0.5)
    # 线性平滑最优参数
    # alpha = linear_param_optimize(data, int((end_time - start_time) / 86400))
    # for i in range(len(data)):
    #     S1, S2, result = linear_double(data[i], alpha[i])
    #     number, values = linear_predict(data[i], S1, S2, alpha[i], big_period, small_period)
    #     numbers.append(number)
    # holt-winter二次平滑
    alpha, beta = holt_winters_param_optimize(data, int((end_time - start_time) / 86400))
    for i in range(len(data)):
        level, trend, result = holt_winters_double(data[i], alpha[i], beta[i])
        list, number = holt_winters_accumulative_predict(data[i], alpha[i], beta[i], level, trend, big_period,
                                                         small_period)
        last = [data[i][-1]] + list[:-1]
        result = []
        for x, y in zip(list, last):
            result.append(max(int(round(x - y)),0))
        results.append(result)
    return results

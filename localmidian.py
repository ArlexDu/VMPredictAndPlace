# coding=utf-8
from matrix import Matrix
import math
import copy
from histogram import Histogram
from kdistance import KDistance
import matplotlib.pyplot as plt


def data_clean(data, flavor_list):
    data = Matrix(data)
    for index in range(len(flavor_list)):
        n = int(flavor_list[index].name.split('flavor')[1])
        some_flavor_train_list = data.get_col(n)
        m, diff = median_diff(some_flavor_train_list)
        mean = get_mean(some_flavor_train_list)
        if mean < 5:
            if diff < 15:
                continue
        if mean > 20:
            if diff < 40:
                continue
        # 获取阈值
        # threshold = 0.6666 * diff
        threshold = 0.75 * diff
        k = 14
        # k = period
        count = len(some_flavor_train_list)
        # remove outlier in list and update data
        for i in range(count):
            if i < k:
                start = 0
                end = i + k
                a = some_flavor_train_list[start:end]
                a.pop(i)
            elif i > (count - k):
                start = i - k
                a = some_flavor_train_list[start:]
                a.pop(-(count - i + 1))
            else:
                start = i - k
                end = i + k
                a = some_flavor_train_list[start:end]
                a.pop(k)
            middan, diff = median_diff(a)
            if some_flavor_train_list[i] > middan + threshold:
                data.matrix[i][n] = int(math.ceil(middan + threshold))

    # print 'data clean'
    return data.matrix


def median_diff(s):
    n = len(s)  # 计算列表内元素数量
    new_s = s[:]
    m = sorted(new_s)  # 排序一下
    diff = abs(m[n - 1] - s[0])
    if n == 1:  # 这个要非常注意，当元素只有一个的时候，直接取值
        return new_s[0], 0
    elif n % 2 != 0:  # 如果元素数量为奇数
        mid = m[(n - 1) // 2]  # 中间值等于元素总数量减一以后除以2，记得要用//
        return mid, diff
    else:
        mid = m[n // 2 - 1] + m[n // 2] / 2  # 如果是偶数，取元素数量//2后减一位的那个值，以及元素数量//2的那个值，记得最后要用float，不然没有小数点
        return mid, diff


def get_mean(a):
    return float(sum(a)) / len(a)


def get_kn(s, v, k):
    k_dis = [abs(s[i] - v) for i in range(len(s))]
    k_dis.sort()
    return k_dis[k]


# 混合方法去噪
def mix_clean(data, flavor_list):
    data = Matrix(data)
    for index in range(len(flavor_list)):
        n = int(flavor_list[index].name.split('flavor')[1])
        some_flavor_train_list = data.get_col(n)
        midian_list = copy.deepcopy(some_flavor_train_list)
        # mean = get_mean(midian_list)
        m, diff = median_diff(midian_list)
        if m < 5:
            if diff < 15:
                continue
        threshold_ratio = 0.75
        if m >= 5:
            threshold_ratio = 0.8
            flavor_list[index].high = 1
        # 获取阈值
        # threshold = 0.6666 * diff
        # threshold = 0.75 * diff 80.623
        # threshold = 0.6 * diff #227.945
        # drawDigram('Flavor'+str(n)+' Origin Data',data.get_col(0),data.get_col(n))
        threshold = threshold_ratio * diff

        k = 14

        # 中位数的方法获得去异常点后的序列
        # 中位数方法获得的异常点判断结果
        count = len(midian_list)
        midian_result = [0 for i in range(count)]
        for i in range(count):
            if i < k:
                start = 0
                end = i + k
                a = midian_list[start:end]
                a.pop(i)
            elif i > (count - k):
                start = i - k
                a = midian_list[start:]
                a.pop(-(count - i + 1))
            else:
                start = i - k
                end = i + k
                a = midian_list[start:end]
                a.pop(k)
            middan, diff = median_diff(a)
            if midian_list[i] > middan + threshold:
                midian_list[i] = int(math.ceil(middan + threshold))
                midian_result[i] = 1

        # 直方图方法获得的异常点判断结果
        histogram_list = []
        histogram_result = [0 for i in range(count)]
        for i in range(len(some_flavor_train_list)):
            flag = False
            for j in range(len(histogram_list)):
                if some_flavor_train_list[i] == histogram_list[j].number:
                    histogram_list[j].members.append(i)
                    flag = True
                    break
            if not flag:
                histogram_list.append(Histogram(some_flavor_train_list[i]))
                histogram_list[-1].members.append(i)
        for i in range(len(histogram_list)):
            histogram_list[i].ratio = float(len(histogram_list[i].members)) / len(some_flavor_train_list)
            if histogram_list[i].ratio < 0.1:
                for j in histogram_list[i].members:
                    histogram_result[j] = 1
        histogram_list.sort(key=lambda Histogram: Histogram.ratio, reverse=True)

        # k临近点的方法获取异常点的判断结果
        outlier_count = int(math.ceil(float(len(some_flavor_train_list)) * 0.1))
        kneighbors = int(math.ceil(float(count)/10))
        kn_result = [0 for i in range(count)]
        kn_list = []
        for i in range(count):
            k_dis = get_kn(some_flavor_train_list, some_flavor_train_list[i], kneighbors)
            kn_list.append(KDistance(i, k_dis))
        kn_list.sort(key=lambda KDistance: KDistance.distance, reverse=True)
        for kdistance in kn_list[:outlier_count]:
            index = kdistance.number
            kn_result[index] = 1
        # 只有当三个条件都满足的时候才被认为是异常点
        for i in range(count):
            if midian_result[i] == 1 and histogram_result[i] == 1 and kn_result[i] == 1:
                # print 'flavor' + str(n) +':'+str(i)
                data.matrix[i][n] = midian_list[i]
        # drawDigram('Flavor' + str(n) + ' Remove Outlier Data', data.get_col(0), data.get_col(n))
    return data.matrix

def drawDigram(name,x,y):
    ax = plt.subplot()
    ax.set_title(name)  # give plot a title
    ax.set_xlabel('Date')  # make axis labels
    ax.set_ylabel('Number')
    ax.set_ylim([0, 30])
    # plt.axis([0, 60, 0, 30])

    plt.plot(x, y)
    plt.xticks((x[0],x[9],x[18],x[27],x[36],x[49]),rotation=0)
    # plt.legend(loc='upper left')

    plt.show()
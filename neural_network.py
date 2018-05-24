# coding=utf-8

import random
import math
from matrix import Matrix
import copy
import time
# 神经网络
class Network(object):

    def __init__(self,sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.weight_initial()
        self.activate_fun = sigmoid
        self.activate_fun_prime = sigmoid_prime

    # 设置平均值为1，标准差为1的高斯分布设置参数
    def weight_initial(self):
        bs = []
        ws = []
        for n in self.sizes[1:]:
            bias = [random.gauss(0,1) for i in range(n)]
            bias = Matrix(bias)
            bias.transposition()
            bs.append(bias)
        self.biases = bs
        for col,row in zip(self.sizes[:-1],self.sizes[1:]):
            w = []
            for i in range(row):
                line = []
                for j in range(col):
                    line.append(random.gauss(0,1)/math.sqrt(col)) # 加速训练
                w.append(line)
            w = Matrix(w)
            ws.append(w)
        self.weights = ws

    #向前传播预测
    def forwardprop(self,x):
        h = x
        for weight,bias in zip(self.weights[:-1],self.biases[:-1]):
            h = weight * h + bias
            h = self.activate_fun(h)
        w = self.weights[-1]
        b = self.biases[-1]
        b.transposition()
        a = w * h + b
        b.transposition()
        return a

    #训练函数
    def train(self,training_data,epochs,mini_batch_size,rate):
        n = len(training_data)
        for i in range(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0,n,mini_batch_size)
            ]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch,rate)
            print('epoch %s training complete' % i)
            cost = self.total_cost(training_data)
            print('cost is  %s' % cost)
            if cost < 0.001:
                return

    #更新每一个mini_batch
    def update_mini_batch(self,mini_batch,rate):
        theta_w = copy.deepcopy(self.weights)
        theta_b = copy.deepcopy(self.biases)
        #初始化weight和bias
        for i,attr in enumerate(theta_w):
            theta_w[i] = attr - attr
        for i, attr in enumerate(theta_b):
            theta_b[i] = attr - attr
        for i in range(len(mini_batch)):
            x = Matrix(mini_batch[i][:-1])
            x.transposition()
            #这里的y基本就是一个数字，为了可以转化为Matrix所以把它变为一维一项的数组
            y = Matrix([mini_batch[i][-1]])
            delta_w,delta_b = self.backprop(x,y)
            theta_w = [ nw+dnw for nw, dnw in zip(theta_w, delta_w)]
            theta_b = [ nb+dnb for nb, dnb in zip(theta_b, delta_b)]
        self.weights = [w -  nw.dotmul(rate / len(mini_batch))
                        for w, nw in zip(self.weights, theta_w)]
        self.biases = [b - nb.dotmul(rate / len(mini_batch))
                       for b, nb in zip(self.biases, theta_b)]

    #反向传播 传入的x,y都是Matrix类型
    def backprop(self,x,y):
        delta_w = copy.deepcopy(self.weights)
        delta_b= copy.deepcopy(self.biases)
        zs = []
        a = x
        acs = []
        acs.append(x)
        for weight,bias in zip(self.weights[:-1],self.biases[:-1]):
            z = weight * a + bias
            zs.append(z)
            a = self.activate_fun(z)
            acs.append(a)
        w = self.weights[-1]
        b = self.biases[-1]
        b.transposition()
        a = w * a + b
        zs.append(a)
        acs.append(a)
        delta = mse(a, y)
        delta_b[-1] = delta
        f = acs[-2]
        f.transposition()
        delta_w[-1] = delta * f
        for l in range(2,self.num_layers):
            z = zs[-l]
            sp = self.activate_fun_prime(z)
            last_w = self.weights[-l + 1]
            last_w.transposition()
            delta = last_w * delta
            #因为python对于list的操作是引用方式，所以这里要把weight再转置回来
            last_w.transposition()
            delta = delta.dotmul(sp)
            delta_b[-l] = delta
            last_a = acs[-l-1]
            last_a.transposition()
            #矩阵相乘可能出错
            delta_w[-l] = delta * last_a
        return delta_w, delta_b

    # 获取误差
    def total_cost(self, data):
        cost = 0.0
        for i in range(len(data)):
            x = Matrix(data[i][:-1])
            x.transposition()
            y = data[i][-1]
            a = self.forwardprop(x)
            cost += (a.matrix[0][0]-y)*(a.matrix[0][0]-y)
        return cost/len(data)


#sigmoid激活函数 返回为Matrix对象
def sigmoid(z):
    a = copy.deepcopy(z)
    for i in range(z.rows()):
        if z.matrix[i][0] == float('nan') or z.matrix[i][0] == float('inf') or z.matrix[i][0] == float('-inf'):
            z.matrix[i][0] = 0
        a.matrix[i][0] = 1.0/(1.0+math.exp(-z.matrix[i][0]))
    return a

#sigmod激活函数导数 返回的为Matrix对象
def sigmoid_prime(z):
    one = []
    for k in range(z.rows()):
        one.append(1)
    one = Matrix(one)
    one.transposition()
    return sigmoid(z).dotmul(one-sigmoid(z))

def relu(z):
    a = copy.deepcopy(z)
    for i in range(z.rows()):
        a.matrix[i][0] = max(z.matrix[i][0],0)
    return a

def relu_prime(z):
    a = copy.deepcopy(z)
    for i in range(z.rows()):
        if z.matrix[i][0] >= 0:
            a.matrix[i][0] = 1
        else:
            a.matrix[i][0] = 0
    return a

#cost function 返回的为Matrix对象
def mse(z,y):
    return (z-y).dotmul(2)

# 构造数据样本 返回list
def predict(data,flavor_list,firstln,start_time, end_time):

    #计算预测的时间段
    last_time = data[len(data) - 1][0]
    st = time.strptime(last_time + ' 0:00:00', '%Y-%m-%d %H:%M:%S')
    last_time = time.mktime(st)
    st = time.strptime(start_time + ' 0:00:00', '%Y-%m-%d %H:%M:%S')
    start_time = time.mktime(st)
    st = time.strptime(end_time + ' 0:00:00', '%Y-%m-%d %H:%M:%S')
    end_time = time.mktime(st)
    big_period = int((end_time - last_time) / 86400)
    days = int((end_time - start_time) / 86400)
    #初始化权重
    # weights = [0.1,0.2,0.5,0.7,0.9,1]
    # for i in range(firstln):
    #     numerator = (-1) * (i - firstln) * (i - firstln)
    #     denominator = 2 * firstln * firstln
    #     s = float(numerator) / float(denominator)
    #     weights.append(math.exp(s))
    # weights.append(1)
    raw = Matrix(data)
    if raw.rows() < firstln:
        firstln = raw.rows()
    final = []
    for index in range(len(flavor_list)):
        n = int(flavor_list[index].name.split('flavor')[1])
        print('neural network for flavor%s' % n)
        sample= []
        for i in range(firstln, len(data)):
            line = raw.get_col(n)[i - firstln:i + 1]
            # line = [ l*w for l,w in zip(line,weights)]
            sample.append(line)
        net = Network([firstln, 10, 5, 1])
        net.train(sample, 30, 20, 0.01)
        X = sample[len(sample)-1][1:]
        sums = []
        for d in range(big_period):
            TX = Matrix(X)
            TX.transposition()
            count = net.forwardprop(TX)
            count = int(round(count.matrix[0][0]))
            if count < 0:
                count = 0
            sums.append(count)
            X.append(count)
            X.pop(0)
        sums = sums[len(sums) - days:len(sums)]
        final.append(sum(sums))
    return final










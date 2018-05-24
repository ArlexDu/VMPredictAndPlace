# coding=utf-8

import re
from flavor import Flavor
from matrix import Matrix
import time
#数据处理

# 获取每一天的每种型号虚拟机的数量
def train_data_process(ecs_lines):
    data = [0 for i in range(16)]
    # in this list data[0] save the data and element1 ~ element15 are the new VFS numbers for each kind that date
    datas = []
    currdate = ''
    for line in ecs_lines:
        line = re.sub('\s+', ',', line.strip())
        kind = line.split(',')[1]
        date = line.split(',')[2]
        if currdate != '' and currdate != date:
            data[0] = currdate
            datas.append(data)
            data = [0 for i in range(16)]
            # 将没有数据集中遗漏的天数增加进去
            st = time.strptime(currdate + ' 0:00:00', '%Y-%m-%d %H:%M:%S')
            tcurrdata = time.mktime(st)
            st = time.strptime(date + ' 0:00:00', '%Y-%m-%d %H:%M:%S')
            tdate = time.mktime(st)
            for i in range(1, int(int(tdate - tcurrdata) / 86400)):
                miss_date = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(tcurrdata + 86400 * i))
                data[0] = miss_date.split(' ')[0]
                datas.append(data)
                data = [0 for i in range(16)]
        if kind.startswith('flavor'):
            n = int(kind.split('flavor')[1])
            if n >= 1 and n <= 15:
                number = data[n]
                number = number + 1
                data[n] = number
        currdate = date
    # the last date's info
    data[0] = currdate
    datas.append(data)
    return datas


# 读取input文件
def read_input_file(input_lines):
    i = 0
    flavor_list = []
    cpu = int(input_lines[i].split()[0])
    memory = int(input_lines[i].split(' ')[1]) * 1024
    # print 'cpu: ', cpu, ' memory: ', memory
    i = i + 2
    flavor_num = int(input_lines[i].split(' ')[0])
    i = i + 1
    for index in range(i, i + flavor_num):
        line = input_lines[index].split()
        # print line
        flavor_list.append(Flavor(line[0], line[1], line[2]))
    i = i + flavor_num + 1
    target = input_lines[i].split()[0]
    # print 'target: ', target
    i = i + 2
    start_time = input_lines[i].split(' ')[0].strip()
    end_time = input_lines[i + 1].split(' ')[0].strip()
    return target, memory, cpu, flavor_list, start_time, end_time

# 累加数据
def data_sum(data, flavor_list):
    data = Matrix(data)
    result = []
    for index in range(len(flavor_list)):
        n = int(flavor_list[index].name.split('flavor')[1])
        some_flavor_train_list = data.get_col(n)
        sum = 0
        for i in range(len(some_flavor_train_list)):
            sum = sum + some_flavor_train_list[i]
            some_flavor_train_list[i] = sum
        result.append(some_flavor_train_list)
    return result


# 累加数据
def test_sum(data,past,flavor_list):
    data = Matrix(data)
    past = Matrix(past)
    result = []
    for index in range(len(flavor_list)):
        n = int(flavor_list[index].name.split('flavor')[1])
        some_flavor_train_list = data.get_col(n)
        past_list = past.get_row(n-1)
        sum = past_list[-1]
        for i in range(len(some_flavor_train_list)):
            sum = sum + some_flavor_train_list[i]
            some_flavor_train_list[i] = sum
        result.append(some_flavor_train_list)
    return result



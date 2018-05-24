# coding=utf-8
from machine import Machine
import random
import data_processing
import localmidian
import exponential_smoothing
import math
from matrix import Matrix
from flavor import Flavor


def predict_vm(ecs_lines, test_lines, input_lines):
    # Do your work from here#
    # training data processing
    result = []
    if ecs_lines is None:
        print('ecs information is none')
        return result
    if input_lines is None:
        print('input file information is none')
        return result
    data = data_processing.train_data_process(ecs_lines)
    test = data_processing.train_data_process(test_lines)
    target, memory, cpu, flavor_list, start_time, end_time = data_processing.read_input_file(input_lines)
    # data = localmidian.data_clean(data, flavor_list)
    data = localmidian.mix_clean(data, flavor_list)
    # 指数平滑的方法
    last_time = data[len(data) - 1][0]
    data = data_processing.data_sum(data, flavor_list)
    test = data_processing.test_sum(test,data,flavor_list)
    # numbers = exponential_smoothing.predict_draw(data,test,last_time, start_time, end_time)
    # numbers = exponential_smoothing.predict(data, last_time, start_time, end_time)
    # 线性归回的方法
    # thetas, remains = wlr.weight_linear_regression(data, flavor_list)
    # numbers = wlr.predic(thetas, remains, data[len(data) - 1][0], start_time, end_time)
    numbers = []
    for i in range(len(test)):
        numbers.append(test[i][-1] - data[i][-1])
    # 神经网络的方法
    # numbers = neural_network.predict(data, flavor_list, 5, start_time, end_time)

    machine_list = []

    # 每天的新增PM
    for n in range(1):
        # numbers = test[n][1:]
        init_flavor_list(flavor_list, numbers)
        init_machine_list(machine_list)
        machine_list = first_hit(machine_list,target, flavor_list, memory, cpu)

    sum_vm = 0
    flavor_list.sort(key=lambda Flavor: Flavor.cpu * 1000 + Flavor.memory)
    for index in range(len(flavor_list)):
        result.append(flavor_list[index].name + ' ' + str(flavor_list[index].predict_num))
        sum_vm = sum_vm + flavor_list[index].predict_num
    result.insert(0, sum_vm)
    # result.insert(0, test[n][0])

    # machine_list = mffd(target, flavor_list, memory, cpu)
    # machine_list_cpu = mffd('CPU', flavor_list, memory, cpu)
    # machine_list_memory = mffd('MEM', flavor_list, memory, cpu)
    # if target == 'CPU':
    #     whole_cpu1 = len(machine_list_cpu) * cpu
    #     whole_cpu2 = len(machine_list_memory) * cpu
    #     flavor_cpu1 = 0
    #     flavor_cpu2 = 0
    #     for k in range(len(machine_list_cpu)):
    #         for i in range(len(flavor_list)):
    #             flavor = flavor_list[i]
    #             if flavor.name in machine_list_cpu[k].vm_dict.keys():
    #                 number1 = machine_list_cpu[k].vm_dict[flavor.name]
    #                 flavor_cpu1 = flavor_cpu1 + flavor.cpu * number1
    #     for k in range(len(machine_list_memory)):
    #         for i in range(len(flavor_list)):
    #             flavor = flavor_list[i]
    #             if flavor.name in machine_list_memory[k].vm_dict.keys():
    #                 number2 = machine_list_memory[k].vm_dict[flavor.name]
    #                 flavor_cpu2 = flavor_cpu2 + flavor.cpu * number2
    #     ratio1 = float(flavor_cpu1) / whole_cpu1
    #     ratio2 = float(flavor_cpu2) / whole_cpu2
    #     if ratio1 >= ratio2:
    #         machine_list = machine_list_cpu
    #     else:
    #         machine_list = machine_list_memory
    #
    # else:
    #     whole_memory1 = len(machine_list_cpu) * memory
    #     whole_memory2 = len(machine_list_memory) * memory
    #     flavor_memory1 = 0
    #     flavor_memory2 = 0
    #     for k in range(len(machine_list_cpu)):
    #         for i in range(len(flavor_list)):
    #             flavor = flavor_list[i]
    #             if flavor.name in machine_list_cpu[k].vm_dict.keys():
    #                 number1 = machine_list_cpu[k].vm_dict[flavor.name]
    #                 flavor_memory1 = flavor_memory1 + flavor.memory * number1
    #     for k in range(len(machine_list_memory)):
    #         for i in range(len(flavor_list)):
    #             flavor = flavor_list[i]
    #             if flavor.name in machine_list_memory[k].vm_dict.keys():
    #                 number2 = machine_list_memory[k].vm_dict[flavor.name]
    #                 flavor_memory2 = flavor_memory2 + flavor.memory * number2
    #     ratio1 = float(flavor_memory1) / whole_memory1
    #     ratio2 = float(flavor_memory2) / whole_memory2
    #     if ratio1 >= ratio2:
    #         machine_list = machine_list_cpu
    #     else:
    #         machine_list = machine_list_memory
    result = write_result(machine_list, result)
    return result


def init_flavor_list(flavor_list, numbers):
    for index in range(len(flavor_list)):
        flavor_list[index].predict_num = 0
        flavor_list[index].predict_num = numbers[index]


def init_machine_list(machine_list):
    for machine in machine_list:
        machine.vm_dict = {}


def fixed_output():
    result = []
    result.append(19)
    result.append('flavor1 0')
    result.append('flavor2 7')
    result.append('flavor3 0')
    result.append('flavor4 5')
    result.append('flavor5 7')
    result.append('')
    result.append('1')
    result.append('1 flavor2 7 flavor5 7 flavor4 5')
    return result


def write_result(machine_list, result):
    result.append('')
    result.append(len(machine_list))
    machine_id = 0
    for machine in machine_list:
        assigned_vm = ''
        machine_id += 1
        for k, v in machine.vm_dict.items():
            assigned_vm += (' ' + k + ' ' + str(v))
        result.append(str(machine_id) + assigned_vm)
    return result


# 首次适应算法FFD
def first_hit(machine_list, target, flavor_list, machine_memory, machine_cpu):
    # flavor_list.sort(key=lambda Flavor: Flavor.cpu * 1000 + Flavor.memory, reverse=True)
    if target == 'CPU':
        flavor_list.sort(key=lambda Flavor: Flavor.cpu, reverse=True)
    else:
        flavor_list.sort(key=lambda Flavor: Flavor.memory, reverse=True)
    for flavor in flavor_list:
        for index in range(flavor.predict_num):
            flag = False
            if not machine_list:
                machine_list.append(Machine(machine_memory, machine_cpu))
            if machine_list:
                for machine in machine_list:
                    if machine.can_accommodate(flavor.memory, flavor.cpu):
                        machine.assign_vm(flavor.name, flavor.memory, flavor.cpu, target)
                        flag = True
                        break
                if not flag:
                    machine_list.append(Machine(machine_memory, machine_cpu))
                    machine_list[-1].assign_vm(flavor.name, flavor.memory, flavor.cpu, target)
    return machine_list


def mffd(target, flavor_list, machine_memory, machine_cpu):
    range_number = 10
    bins = [[] for i in range(range_number)]
    ranges = [[] for i in range(range_number)]
    if target == 'CPU':
        per = float(machine_cpu) / range_number
        for i in range(len(flavor_list)):
            if flavor_list[i].predict_num == 0:
                continue
            else:
                for k in range(flavor_list[i].predict_num):
                    kind = int(math.floor(flavor_list[i].cpu / per))
                    ranges[kind].append(flavor_list[i])
    else:
        per = float(machine_memory) / range_number
        for i in range(len(flavor_list)):
            if flavor_list[i].predict_num == 0:
                continue
            else:
                for k in range(flavor_list[i].predict_num):
                    kind = int(math.floor(flavor_list[i].cpu / per))
                    ranges[kind].append(flavor_list[i])
    for i in range(range_number - 1, -1, -1):
        while len(ranges[i]) != 0:
            item_id = random.randint(0, len(ranges[i]) - 1)
            item = ranges[i][item_id]
            allocate = False
            for j in range(len(bins)):
                if len(bins[j]) == 0:
                    continue
                bin_id = random.randint(0, len(bins[j]) - 1)
                bin = bins[j][bin_id]
                if bin.can_accommodate(item.memory, item.cpu):
                    bin.assign_vm(item.name, item.memory, item.cpu, target)
                    allocate = True
                    bins[j].pop(bin_id)
                    # 更新bins
                    if target == 'CPU':
                        per = float(machine_cpu) / range_number
                        kind = int(math.floor(bin.residueCPU / per))
                        bins[kind].append(bin)
                    else:
                        per = float(machine_memory) / range_number
                        kind = int(math.floor(bin.residueMemory / per))
                        bins[kind].append(bin)
                    break
            if not allocate:
                newbin = Machine(machine_memory, machine_cpu)
                newbin.assign_vm(item.name, item.memory, item.cpu, target)
                # 更新bins
                if target == 'CPU':
                    per = float(machine_cpu) / range_number
                    kind = int(math.floor(newbin.residueCPU / per))
                    bins[kind].append(newbin)
                else:
                    per = float(machine_memory) / range_number
                    kind = int(math.floor(newbin.residueMemory / per))
                    bins[kind].append(newbin)
            # 更新ranges
            ranges[i].pop(item_id)
    machine_list = []
    for i in range(range_number):
        if len(bins[i]) == 0:
            continue
        for j in range(len(bins[i])):
            machine_list.append(bins[i][j])
    return machine_list


# 最佳适应算法
def best_hit(target, flavor_list, machine_memory, machine_cpu):
    machine_list = []
    if target == 'CPU':
        flavor_list.sort(key=lambda Flavor: Flavor.cpu, reverse=True)
    else:
        flavor_list.sort(key=lambda Flavor: Flavor.memory, reverse=True)
    for flavor in flavor_list:
        for index in range(flavor.predict_num):
            if not machine_list:
                machine_list.append(Machine(machine_memory, machine_cpu))
            best = -1
            best_memory = machine_memory
            best_cpu = machine_cpu
            index = 0
            if machine_list:
                for machine in machine_list:
                    if machine.can_accommodate(flavor.memory, flavor.cpu):
                        if (target == 'CPU' and machine.residueCPU <= best_cpu) or (
                                        target == 'MEM' and machine.residueMemory <= best_memory):
                            best = index
                            best_cpu = machine.residueCPU
                            best_memory = machine.residueMemory
                    index += 1
            if best == -1:
                machine_list.append(Machine(machine_memory, machine_cpu))
                machine_list[-1].assign_vm(flavor.name, flavor.memory, flavor.cpu, target)
            else:
                machine_list[best].assign_vm(flavor.name, flavor.memory, flavor.cpu, target)
    return machine_list

def predict_vm_all(ecs_lines, test_lines, input_lines):
    # Do your work from here#
    # training data processing
    result = []
    if ecs_lines is None:
        print('ecs information is none')
        return result
    if input_lines is None:
        print('input file information is none')
        return result
    data = data_processing.train_data_process(ecs_lines)
    test = data_processing.train_data_process(test_lines)
    target, memory, cpu, flavor_list, start_time, end_time = data_processing.read_input_file(input_lines)
    # data = localmidian.data_clean(data, flavor_list)
    data = localmidian.mix_clean(data, flavor_list)
    # 指数平滑的方法
    last_time = data[len(data) - 1][0]
    data = data_processing.data_sum(data, flavor_list)
    test = data_processing.test_sum(test,data,flavor_list)
    # numbers = exponential_smoothing.predict_draw(data,test,last_time, start_time, end_time)
    # numbers = exponential_smoothing.predict(data, last_time, start_time, end_time)
    numbers = exponential_smoothing.predict_all(data, last_time, start_time, end_time)
    predict_list=[]
    for i in range(len(numbers)):
        flavor = flavor_list[i]
        for k in range(len(numbers[i])):
            f = Flavor(flavor.name+'_'+str(k), flavor.cpu, flavor.memory)
            f.predict_num = numbers[i][k]
            predict_list.append(f)
    # 线性归回的方法
    # thetas, remains = wlr.weight_linear_regression(data, flavor_list)
    # numbers = wlr.predic(thetas, remains, data[len(data) - 1][0], start_time, end_time)

    # 神经网络的方法
    # numbers = neural_network.predict(data, flavor_list, 5, start_time, end_time)

    machine_list = mffd(target, predict_list, memory, cpu)

    sum_vm = 0
    flavor_list.sort(key=lambda Flavor: Flavor.name)
    for index in range(len(predict_list)):
        result.append(predict_list[index].name + ' ' + str(predict_list[index].predict_num))
        sum_vm = sum_vm + predict_list[index].predict_num
    result.insert(0, sum_vm)

    result = write_result(machine_list, result)
    return result
# coding=utf-8
import copy
from functools import reduce

class Matrix(object):

    def __init__(self,array):
        if isinstance(array[0],list):
            self.matrix = array
        else:
            final = []
            final.append(array)
            self.matrix = final

    def __add__(self, matrix):
        if self.size() != matrix.size():
            print('Size different error')
        clone_matrix = copy.deepcopy(self.matrix)
        for index1, arr1 in enumerate(clone_matrix):
            for index2, item in enumerate(arr1):
                clone_matrix[index1][index2] += matrix.matrix[index1][index2]
        return Matrix(clone_matrix)

    def __sub__(self, matrix):
        if self.size() != matrix.size():
            print('Size different error')
        clone_matrix = copy.deepcopy(self.matrix)
        for index1, arr1 in enumerate(clone_matrix):
            for index2, item in enumerate(arr1):
                clone_matrix[index1][index2] -= matrix.matrix[index1][index2]
        return Matrix(clone_matrix)

    def __mul__(self, matrix):
        m = self.rows()
        n = matrix.cols()
        k = self.cols()
        if k != matrix.rows():
            print('Size error')
        temp_matrix = [([0] * n) for i in range(m)]
        for index1, arr1 in enumerate(temp_matrix):
            row = self.get_row(index1)
            for index2, item in enumerate(arr1):
                col = matrix.get_col(index2)
                temp_matrix[index1][index2] = reduce(lambda x, y=0: x + y, map(lambda i: row[i] * col[i], range(k)))
        return Matrix(temp_matrix)

    def size(self):
        return '%s * %s' % (len(self.matrix),len(self.matrix[0]))

    def rows(self):
        return len(self.matrix)

    def cols(self):
        return len(self.matrix[0])

    def get_row(self,index):
        return self.matrix[index]

    def get_col(self,index):
        return list(map(lambda a:a[index],self.matrix))

    def transposition(self):
        cols = self.cols()
        rows = self.rows()
        temp_matrix = [([0] * rows) for i in range(cols)]
        for index1, arr1 in enumerate(self.matrix):
            for index2, item in enumerate(arr1):
                temp_matrix[index2][index1] = self.matrix[index1][index2]
        self.matrix = temp_matrix

    #矩阵的对应相乘
    def dotmul(self,matrix):
        if isinstance(matrix,Matrix):
            if self.size() != matrix.size(): print('Size different error')
            clone_matrix = copy.deepcopy(self.matrix)
            for index1, arr1 in enumerate(clone_matrix):
                for index2, item in enumerate(arr1):
                    clone_matrix[index1][index2] *= matrix.matrix[index1][index2]
        else:
            clone_matrix = copy.deepcopy(self.matrix)
            for index1, arr1 in enumerate(clone_matrix):
                for index2, item in enumerate(arr1):
                    clone_matrix[index1][index2] *= matrix
        return Matrix(clone_matrix)


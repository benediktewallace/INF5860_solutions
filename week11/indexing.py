import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt


def index1(mat):
    """
    Return second row
    """
    return mat[1,:]


def index2(mat):
    """
    Return second column
    """
    return mat[:,1]


def index3(mat):
    """
    Return 2x2 square, starting at index (1, 1)
    """
    return mat[1:3,1:3]


def index4(mat):
    """
    Return second and forth column
    """
    return mat[:,1:4:2]


def index5(mat):
    """
    Return second and forth column, but in reversed order
    """
    return mat[::-1,1:4:2]


def index6(mat):
    """
    Return every other row in mat. Row with index 0, 2, 4, 6, 8...
    """
    return mat[::2,:]


def index7(mat):
    """
    Return true where mat-values are higher than 3
    """
    return mat>3


def index8(mat):
    """
    Return array of all values higher than 3
    """
    return tf.boolean_mask(mat, mat>3)


def index9(mat):
    """
    Return the number of values higher than 3
    """
    return tf.reduce_sum(tf.to_float(mat>3))


if __name__ == '__main__':
    m0 = np.arange(16).reshape((4,4))
    print("Original",m0)
    m1 = index1(np.arange(16).reshape((4,4)))
    print("Second row",m1)
    m2 = index2(np.arange(16).reshape((4,4)))
    print("Second col",m2)
    m3 = index3(np.arange(16).reshape((4,4)))
    print("square",m3)
    m = index4(np.arange(16).reshape((4,4)))
    print("2nd and 4th column",m)
    m = index5(np.arange(16).reshape((4,4)))
    print("2nd and 4th row reversed",m)
    m = index6(np.arange(16).reshape((4,4)))
    print("every other row in mat",m)
    m = index7(np.arange(16).reshape((4,4)))
    print("boolean matvalues higher than 3",m)
    m = index8(np.arange(16).reshape((4,4)))
    print("mat-values are higher than 3",m)
    m = index9(np.arange(16).reshape((4,4)))
    print("number of values higher than 3",m)



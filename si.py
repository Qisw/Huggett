from multiprocessing import Process, Lock
from multiprocessing.sharedctypes import Value, Array, RawArray
from ctypes import Structure, c_double
from numpy import random, zeros, array

class Point(Structure):
    _fields_ = [('x', c_double), ('y', c_double)]

def modify(x,c):
    # n.value **= 2
    mu = zeros(8).reshape(2,2,2)
    for i in range(2):
        for j in range(2):
            for k in range(2):
                mu[i,j,k] = i+j+c
    mu = mu.reshape(8)
    for i in range(len(x)):
        x[i] = mu[i]
    # for i in range(len(x)):
    #     x[i] **= 2

if __name__ == '__main__':
    lock = Lock()
    # n = [Value('i', 7) for t in range(8)]
    xx = [RawArray(c_double, 8) for t in range(8)]
    # xx = [Array(c_double, zeros(8), lock=False) for t in range(8)]
    # xx = [Array(c_double, random.random(4), lock=False) for t in range(8)]
    # s = Array('c', 'hello world', lock=lock)
    # # A = Array(Point, [[1.875,-6.25], [-5.75,2.0], [2.375,9.5]], lock=lock)
    # A = Array(Point, [(1.875,-6.25), (-5.75,2.0), (2.375,9.5)], lock=lock)
    for x in xx:
        p = Process(target=modify, args=(x,5))
        p.start()
        p.join()

    # print n.value
    print '\n\n'
    for x in xx:
        print array(x).reshape(2,2,2)
    # print s.value
    # print [(a.x, a.y) for a in A]

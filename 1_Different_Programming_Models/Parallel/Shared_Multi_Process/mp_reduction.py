import multiprocessing
from multiprocessing import Process, Lock, Value
import numpy as np
import sys
import timeit
from timeit import default_timer as timer

def solver(l, n, i, array, s):
    private_sum = np.sum(array[i*n:(i+1)*n]*array[i*n:(i+1)*n])
    l.acquire()
    s.value += private_sum
    l.release()


if __name__ == '__main__':
    if len(sys.argv) != 2 :
        print("\nNeed number of threads.\n")
        exit()

    SIZE = (1 << 30)
    nthreads = int(sys.argv[1])
    mysum = Value('d', 0.0)

    a = np.ones(SIZE)
    lock = Lock()
    p = []

    start = timer()
    for i in range(1, nthreads):
        p.append(Process(target=solver, args=(lock, int(SIZE/nthreads), i, a, mysum)))
        p[i-1].start()

    private_sum = np.sum(a[0:int(SIZE/nthreads)]*a[0:int(SIZE/nthreads)])
    lock.acquire()
    mysum.value += private_sum
    lock.release()

    for i in range(1, nthreads):
        p[i-1].join()

    end = timer()

    print("SUM: ", mysum.value)
    print("Time in microseconds: ", (end - start)*1000000)
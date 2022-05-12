import pycuda.autoinit
import pycuda.driver as drv
import numpy as np
import sys
from pycuda import gpuarray
from pycuda.compiler import SourceModule
from pycuda.autoinit import context
import timeit
from timeit import default_timer as timer

ker = SourceModule("""
__global__ void reduction_kernel (float *a, int span, double *target)
{
    int i;
    int id = threadIdx.x + blockIdx.x*blockDim.x;
    target[id] = 0;

    for (i=span*id; i<span*(id+1); i++) {
        target[id] += (a[i]*a[i]);
    }
}
""")

reduction_gpu = ker.get_function("reduction_kernel")

if len(sys.argv) != 2 :
    print("\nNeed number of threads.\n")
    exit()

SIZE = (1 << 30)
nthreads = int(sys.argv[1])

a = drv.managed_empty(shape=SIZE, dtype=np.float32, mem_flags=drv.mem_attach_flags.GLOBAL)
a.fill(1)

private_sum = drv.managed_empty(shape=nthreads, dtype=np.double, mem_flags=drv.mem_attach_flags.GLOBAL)

start = timer()
if nthreads < 16 :
    reduction_gpu(a, np.intc(SIZE/nthreads), private_sum, block=(nthreads, 1, 1), grid=(1, 1, 1))
else :
    reduction_gpu(a, np.intc(SIZE/nthreads), private_sum, block=(16, 1, 1), grid=(int(nthreads/16), 1, 1))
context.synchronize()

mysum = np.sum(private_sum)

end = timer()

print("SUM: ", mysum)
print("Time in microseconds: ", (end - start)*1000000)
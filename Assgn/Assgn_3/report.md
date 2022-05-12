## 1
Assumption: n and t are large, (at least 32 each, so warps are populated).
We have defined each block to have 32 threads, so we can use tree reduction. Initialization has been done using cuda's random function
Tree reduction is done simply by using shuffle functions.
Overall, the memory overhead seems high, as we are unable to use large number of threads. The improvement over best performing cpu(8 threads) is minor as well


We had tried shared memory implementations as well, but the group launch was causing problems, only running for small t and n. As well as, it was not converging and we were unable to find the problem. 
The advantage of tree reduction is also minor for this program

n = 2048
```
Threads     GPU(without Opt)              GPU(with Tree)     Threads(8 Threads)     Pthread  
256             3.72 (seconds)              3.53                  1                   10.98
512             2.22                        1.96                  2                   5.32  
1024            1.95                        1.62                  4                   2.99
2048            4.72                        3.87                  8                   1.74
4096            5.51                        4.41                 16                   2.18
```



First and second question's codes have been tested on separate machines.
## 2
NOTE : I assume warp size to be 32 and number of threads and matrix size to be atleast 32 and 32*32 respectively.

I have written the file matrixVectorProduct3.cu which takes in input size of matrix and number of threads. The only constraint I've put is that the number of threads should be greater than 32, due to the way in which I've implemented my multiplication kernel. Based on which condition the number of threads and size of matrix satisfy, I initialise my kernel in a way that no more than the given number of threads are active at a given point of time. I use multiple invocation of the same kernel to ensure complete initialisation when my number of threads isn't enough for complete initialisation of the arrays at once (since I use one thread to initialise one value in the kernel) 

After having initialised the kernels, I move towards the multiplication part where the multiplication happens in a way that first both arrays are loaded into a block level shared memory and then each thread performs single multiplication operation. The multiplication is performed in a way that consecutive elements of a row in shared memory are given to the threads in a warp, after this, I use __shfl_down_sync() to fetch data in other register back to (threadIdx%warpSize == 0) register after which it updates the value in the result array after performing all such operations on the matrix. 

Results (Best) for some matrix Sizes (time in microseconds) - 
```
Matrix Size      GPU(without Opt)                   GPU                       OpenMP              Pthreads
1024*1024            1106                  60 (with 65536 threads)   1258 (4 threads)     340 (8 threads)
2048*2048            3077                 186 (131072 threads)       2742 (4 threads)     795 (8 threads)
4096*4096           25674                 508 (131072 threads)       6194 (8 threads)    3944 (12 threads)
8192*8192           92356                2051 (131072 threads)      19624 (8 threads)   12271 (12 threads)
16384*16384        387477                7244 (262144 threads)      50835 (8 threads)   47548 (12 threads)
32768*32768       1678984               27710 (524288 threads)     206152 (8 threads)  162483 (12 threads)
```

for fixed n = 2048
(in Microseconds)
```
Threads      GPU(without Opt)                   GPU           Threads            OpenMP              Pthreads
2048            28132                           3601            1                 24411                16671
4096            17565                           1882            2                 13310                10388
8192             9405                           1040            4                 42010                11147
16384           13454                           648             8                128569                11362 
32768           16654                           677             16               246368                6923
```


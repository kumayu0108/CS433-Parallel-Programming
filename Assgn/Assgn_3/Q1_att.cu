#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <sys/time.h>
#include <cuda.h>
#include <cooperative_groups.h>
#include <curand_kernel.h>
#include <curand.h>
namespace cg = cooperative_groups;
 
#define TOL 1e-5
#define ITER_LIMIT 1000
#define THREADS_PER_BLOCK 1024
 
__managed__ float diff = 0;
__managed__ int done = 0;
__managed__ int iter = 0;
 
__global__ void init_kernel (float *a, int n, int t, int span, curandState *states)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    curand_init(id, id, 0, &states[id]);
    for (int i = id * span; i < (id + 1) * span; i++)
    {
        a[i] = curand_uniform(&states[id]);
    }
    if (threadIdx.x == blockDim.x - 1 && blockIdx.x == gridDim.x - 1)
    {
        printf("in last thread, %d of %d threads, %d of %d blocks\n", threadIdx.x + 1, blockDim.x, blockIdx.x + 1, gridDim.x);
        printf("filling from %d to %d\n", (id + 1) * span, (n + 2) * (n + 2) - 1);
        for (int i = (id + 1) * span; i < (n + 2) * (n + 2); i++)
        {
            a[i] = curand_uniform(&states[id]);
        }
    }
 
}
 
__global__ void solver(float *a, int n, int x_tile, int y_tile)
{
    float local_diff = 0.0, temp;
    __shared__ float atile[68][68];
    int x_id, y_id;
    int ij, ipj, ijm, imj, ijp;
    cg::grid_group grid = cg::this_grid();
    while (!done)
    {
        local_diff = 0;
        x_id = blockIdx.x * blockDim.x + threadIdx.x;
        y_id = blockIdx.y * blockDim.y + threadIdx.y;
        if(x_id == 0 && y_id == 0)
        {
            diff = 0;
        }
        grid.sync();
 
        for (int i = x_id * x_tile; i < (x_id + 1) * x_tile; i++)
        {
            for (int j = y_id * y_tile; j < (y_id + 1) * y_tile; j++)
            {
                int tidx = i % x_tile;
                int tidy = j % y_tile;
                ij = (i + 1) * (n + 2) + (j + 1);
                ipj = (i + 2) * (n + 2) + (j + 1);
                imj = (i) * (n + 2) + (j + 1);
                ijp = (i + 1) * (n + 2) + (j + 2);
                ijm = (i + 1) * (n + 2) + (j);
                atile[tidx + 1][tidy + 1] = a[ij];
                atile[tidx + 1][tidy + 2] = a[ijp];
                atile[tidx + 2][tidy + 1] = a[ipj];
                atile[tidx][tidy + 1] = a[imj];
                atile[tidx + 1][tidy] = a[ijm];
                temp = a[ij];
                a[ij] = 0.2 * (atile[tidx + 1][tidy + 1] + atile[tidx + 1][tidy + 2] + atile[tidx + 2][tidy + 1] + atile[tidx][tidy + 1] + atile[tidx + 1][tidy]);
                local_diff+=fabs(a[ij] - temp);
                // printf("[iter: %d] %d, %d, %d, %d, %d %f\n", iter, ij, ipj, imj, ijp, ijm, local_diff);
            }
        }
        atomicAdd(&diff, local_diff);
        if(x_id == 0 && y_id == 0)
        {
            iter++;
            // printf("[iter: %5d] diff: %6f, local: %6f\n", iter, diff/(n * n), local_diff);
        }
        grid.sync();
        if((diff / (n * n)) < TOL || (iter == ITER_LIMIT))
        {
            done = 1;
        }
        grid.sync();
    }
}
 
int main (int argc, char *argv[])
{
 
    float *a;
    int n, t, left, init_span, total_span;
    int x_tile, y_tile;
    int num_thread_blocks_x, num_thread_blocks_y;
    int num_threads_per_block_x = 1, num_threads_per_block_y = 1;
    struct timeval tv0, tv2;
    struct timezone tz0, tz2;
 
 
 
    n = atoi(argv[1]);
    t = atoi(argv[2]);
    assert((t  & (t - 1)) == 0);
    if(n * n < t)
    {
        t = n * n;
        printf("more threads than number of elts, reducing them; nthreads now %d\n", t);
    }
    total_span = ((n) * (n)) / t;
    int log_t_2 = (int)log2((float)t);
    int log_span_2 = (int)log2((float)total_span);
    if(log_t_2 < 10) // less than THREADS_PER_BLOCK
    {
        int half = log_t_2 / 2;
        left = log_t_2 - half;
        num_threads_per_block_x = (1<<left);
        num_threads_per_block_y = (1<<half);
    }
    else
    {
        num_threads_per_block_x = 32;
        num_threads_per_block_y = 32;
        left = log_t_2 - 10;
    }
 
    int half = log_span_2 / 2;
    left = log_span_2 - half;
    x_tile = (1<<half);
    y_tile = (1<<left);
 
    num_thread_blocks_x = n / (x_tile * num_threads_per_block_x);
    num_thread_blocks_y = n / (y_tile * num_threads_per_block_y);
 
    printf("choosing gridDims: %d, %d\n", num_thread_blocks_x, num_thread_blocks_y);
    printf("choosing blockDims: %d, %d\n", num_threads_per_block_x, num_threads_per_block_y);
    printf("choosing tiles: %d, %d\n", x_tile, y_tile);
 
    assert(num_thread_blocks_x * num_threads_per_block_x * num_thread_blocks_y * num_threads_per_block_y == t);
    assert(x_tile * y_tile == total_span);
    assert(x_tile * num_threads_per_block_x * num_thread_blocks_x == n);
    assert(y_tile * num_threads_per_block_y * num_thread_blocks_y == n);
 
 
 
    cudaMallocManaged((void**)&a, sizeof(float) * (n + 2) * (n + 2));
 
    int device = -1;
    cudaGetDevice(&device);
    cudaMemAdvise(a, sizeof(float) * (n + 2) * (n + 2), cudaMemAdviseSetPreferredLocation, device);
 
    if((n + 2) * (n + 2) >= THREADS_PER_BLOCK)
    {
        int num_blocks = ((n + 2) * (n + 2)) / THREADS_PER_BLOCK;
        int tot_threads = num_blocks * THREADS_PER_BLOCK;
        init_span = ((n + 2) * (n + 2))/ tot_threads;
        curandState *dev_random;
        cudaMalloc((void**)&dev_random, tot_threads * sizeof(curandState));
        init_kernel<<<((n + 2) * (n + 2)) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(a, n, t, init_span, dev_random);
    }
    else
    {
        init_span = 1;
        int tot_threads = (n + 2) * (n + 2);
        curandState *dev_random;
        cudaMalloc((void**)&dev_random, tot_threads * sizeof(curandState));
        init_kernel<<<1, (n + 2) * (n + 2)>>>(a, n, t, init_span, dev_random);
    }
    // printf("choosing init span: %d, normal span: %d\n\n", init_span, total_span);
 
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("Failed to launch kernel (error code: %s)!\n", cudaGetErrorString(err));
    }
    cudaDeviceSynchronize();
 
    // for(int i = 0; i < 32; i++)
    // {
    //     printf("%f ", a[i]);
    // }
    // printf("\n\n");
 
    int supportsCoopLaunch = 0;
    cudaDeviceGetAttribute(&supportsCoopLaunch, cudaDevAttrCooperativeLaunch, device);
    if(!supportsCoopLaunch)
    {
        printf("\nno cooperative groups launch, exiting\n");
        exit(1);
    }
 
    void* kernelArgs[] = {(void*)&a, (void*)&n, (void*)&x_tile, (void*)&y_tile};
    dim3 dimBlock(num_threads_per_block_x, num_threads_per_block_y);
    dim3 dimGrid(num_thread_blocks_x, num_thread_blocks_y);
 
    gettimeofday(&tv0, &tz0);
    cudaLaunchCooperativeKernel((void*)solver, dimGrid, dimBlock, kernelArgs);
 
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("Failed to launch kernel (error code: %s)!\n", cudaGetErrorString(err));
    }
    cudaDeviceSynchronize();
    gettimeofday(&tv2, &tz2);
 
    printf("time: %ld microseconds\n", (tv2.tv_sec-tv0.tv_sec)*1000000+(tv2.tv_usec-tv0.tv_usec));
    return 0;
}
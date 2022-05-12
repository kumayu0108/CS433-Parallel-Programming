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
#define TILE_X 16
#define TILE_Y 16
#define THREADS_PER_BLOCK 1024
#define TPBX TILE_X
#define TPBY TILE_Y

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
        for (int i = (id + 1) * span; i < (n + 2) * (n + 2); i++)
        {
            a[i] = curand_uniform(&states[id]);
        }
    }

}

__global__ void solver(float *a, int n, int dim_span_x, int dim_span_y)
{
    float local_diff = 0.0, temp;
    int x_id, y_id;
    int ij, ipj, ijm, imj, ijp;
    cg::grid_group grid = cg::this_grid();
    while (!done)
    {
        __shared__ float atile[TILE_X + 2][TILE_Y + 2];
        local_diff = 0;
        x_id = blockIdx.x * blockDim.x + threadIdx.x;
        y_id = blockIdx.y * blockDim.y + threadIdx.y;
        if(x_id == 0 && y_id == 0)
        {
            diff = 0;
        }
        grid.sync();

        for (int m = 0; m < dim_span_x; m++)
        {
            for (int n = 0; n < dim_span_y; n++)
            {
                int tidx = threadIdx.x;
                int tidy = threadIdx.y;

                int i = (blockIdx.x * dim_span_x + m) * TILE_X + threadIdx.x;
                int j = (blockIdx.y * dim_span_y + n) * TILE_Y + threadIdx.y;
                // printf("%d %d\n", i, j);

                ij = (i + 1) * (n + 2) + (j + 1);
                ipj = (i + 2) * (n + 2) + (j + 1);
                imj = (i) * (n + 2) + (j + 1);
                ijp = (i + 1) * (n + 2) + (j + 2);
                ijm = (i + 1) * (n + 2) + (j);
                __syncthreads();
                atile[tidx + 1][tidy + 1] = a[ij];
                atile[tidx][tidy + 1] = a[imj];
                atile[tidx + 2][tidy + 1] = a[ipj];
                atile[tidx + 1][tidy] = a[ijm];
                atile[tidx + 1][tidy + 2] = a[ijp ];
                __syncthreads();

                temp = a[ij];
                a[ij] = 0.2 * (atile[tidx + 1][tidy + 1] + atile[tidx][tidy + 1] + atile[tidx + 2][tidy + 1] + atile[tidx + 1][tidy] + atile[tidx + 1][tidy + 2]);
                local_diff+=fabs(a[ij] - temp);
                __syncthreads();
            }
        }
        atomicAdd(&diff, local_diff);
        if(x_id == 0 && y_id == 0)
        {
            iter++;
        }
        grid.sync();
        if((diff / (n * n)) < TOL || (iter == ITER_LIMIT))
        {
            if(x_id == 0 && y_id == 0)
                printf("Thread {0, 0} : Iterations %d  Error %.7f\n", iter, diff/(n*n));
            done = 1;
        }
        grid.sync();
    }
}

int main (int argc, char *argv[])
{

	float *a;
    int n, t, init_span, total_span, pt_span_x, pt_span_y;
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
    int log_span = (int)log2((float)t);
    int half = log_span / 2;
    int left = log_span - half;
    // everything is a square, so calculated in one dimension
    num_thread_blocks_x = (1<<left) / TILE_X;
    num_thread_blocks_y = (1<<half) / TILE_Y;
    num_threads_per_block_x = TILE_X;
    num_threads_per_block_y = TILE_Y;
    total_span = (n * n) / t;
    half = ((int)log2((float)total_span)) / 2;
    left = ((int)log2((float)total_span)) - half;
    pt_span_x = (1<<half);
    pt_span_y = (1<<left);

    assert(num_thread_blocks_x * num_threads_per_block_x * num_thread_blocks_y * num_threads_per_block_y == t);

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

    int supportsCoopLaunch = 0;
    cudaDeviceGetAttribute(&supportsCoopLaunch, cudaDevAttrCooperativeLaunch, device);
    if(!supportsCoopLaunch)
    {
        printf("\nno cooperative groups launch, exiting\n");
        exit(1);
    }

    void* kernelArgs[] = {(void*)&a, (void*)&n, (void*)&pt_span_x, (void*)&pt_span_y};
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

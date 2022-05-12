#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <cuda.h>
#include <sys/time.h>
// #include <cuPrintf.h>

#define TILE_SIZE 32

__global__ void init_kernel (float *a, int N)
{
	int id = threadIdx.x + blockIdx.x*blockDim.x;
	a[id] = (float)id/(N*N);
}

__global__ void matvecmult_kernel (float *a, float *b, float *c, int N) 	
{
	int c_row = blockIdx.y*blockDim.y + threadIdx.y;
	unsigned mask = 0xffffffff;
	float x = 0;
	__shared__ float as[TILE_SIZE][TILE_SIZE];
	__shared__ float bs[TILE_SIZE];
	
	for (int i=0; i<N/TILE_SIZE; i++) {
		as[threadIdx.y][threadIdx.x] = a[c_row*N + i*TILE_SIZE + threadIdx.x];
		bs[threadIdx.x] = b[(i*TILE_SIZE) + threadIdx.x];
		
		__syncthreads();
        if((threadIdx.x % warpSize) == 0)x += as[threadIdx.y][threadIdx.x] * bs[threadIdx.x];
		else x = as[threadIdx.y][threadIdx.x] * bs[threadIdx.x];

		__syncthreads();
		for (int j=warpSize/2; j>0; j=j/2) x += __shfl_down_sync(mask, x, j);
		__syncthreads();
	}

	if ((threadIdx.x % warpSize) == 0) c[c_row] = x;
	__syncthreads();
}

int main(int argc, char *argv[]){
    float *a, *b, *c;
	struct timeval tv0, tv2;
	struct timezone tz0, tz2;
	srand(time(0));

	if (argc != 3) {
		printf ("Need size of matrix and number of threads.\n");
		exit(1);
	}

	int N, numThr;
    cudaMallocManaged((void**)&N, sizeof(int));
	N = atoi(argv[1]);
	numThr = atoi(argv[2]);
    int device = -1;
        cudaGetDevice(&device);
    cudaMallocManaged((void**)&a, sizeof(float)*N*N);
	cudaMemPrefetchAsync(a, sizeof(float)*N*N, device, NULL);
	cudaMallocManaged((void**)&b, sizeof(float)*N);
	cudaMemPrefetchAsync(b, sizeof(float)*N, device, NULL);
	cudaMallocManaged((void**)&c, sizeof(float)*N);

	cudaMemAdvise(c, sizeof(float)*N, cudaMemAdviseSetPreferredLocation, device);
	cudaMemPrefetchAsync(c, sizeof(float)*N, device, NULL);


    init_kernel<<<N*N/1024, 1024>>>(a, N);	
	cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("Failed to launch 1st init kernel (error code: %s)!\n", cudaGetErrorString(err));
    }
	init_kernel<<<N/1024, 1024>>>(b, N);
	err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("Failed to launch 2nd init kernel (error code: %s)!\n", cudaGetErrorString(err));
    }

    cudaDeviceSynchronize();

	gettimeofday(&tv0, &tz0);

    dim3 dimBlock(TILE_SIZE, TILE_SIZE);
	dim3 dimGrid(1, N/TILE_SIZE);

    matvecmult_kernel<<<dimGrid, dimBlock>>>(a, b, c, N);
	if (err != cudaSuccess)
    {
        printf("Failed to launch matvecmul kernel (error code: %s)!\n", cudaGetErrorString(err));
    }
    cudaDeviceSynchronize();
	gettimeofday(&tv2, &tz2);

	int rowC = random() % N;

	float x = 0;

	for (int i=0; i<N; i++) x += a[rowC*N + i]*b[i];
	float error = fabs(c[rowC] - x);
	printf("Error: %0.12f, computed value: %0.12f, actual value: %0.12f, time: %ld microseconds\n", error, c[rowC], x, (tv2.tv_sec-tv0.tv_sec)*1000000+(tv2.tv_usec-tv0.tv_usec));
    return 0;
}

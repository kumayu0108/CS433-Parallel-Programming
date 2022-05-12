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
	int c_row = blockIdx.x*blockDim.y + threadIdx.y;
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


    cudaError_t err;
	if(numThr < 1024){
		if(numThr < N*N){
			for(int i = 0; i < N*N; i += numThr){
				init_kernel<<<1, numThr>>>(a + i, N);
				err = cudaGetLastError();
				if (err != cudaSuccess)
				{
					printf("Failed to launch init kernel (error code: %s)!\n", cudaGetErrorString(err));
				}
				cudaDeviceSynchronize();
			}
		}
		else {
			init_kernel<<<1, N*N>>>(a, N);	
			err = cudaGetLastError();
			if (err != cudaSuccess)
			{
				printf("Failed to launch init kernel (error code: %s)!\n", cudaGetErrorString(err));
			}
			cudaDeviceSynchronize();
		}
		if(numThr < N){
			for(int i = 0; i < N; i += numThr){
				init_kernel<<<1, numThr>>>(b + i, N);
				err = cudaGetLastError();
				if (err != cudaSuccess)
				{
					printf("Failed to launch init kernel (error code: %s)!\n", cudaGetErrorString(err));
				}
				cudaDeviceSynchronize();
			}
		}
		else {
			init_kernel<<<1, N>>>(b, N);	
			err = cudaGetLastError();
			if (err != cudaSuccess)
			{
				printf("Failed to launch init kernel (error code: %s)!\n", cudaGetErrorString(err));
			}
			cudaDeviceSynchronize();
		}
	}
	else {
		if(numThr < N*N){
			for(int i = 0; i < N*N; i += numThr){
				init_kernel<<<numThr/1024, 1024>>>(a + i, N);	
				err = cudaGetLastError();
				if (err != cudaSuccess)
				{
					printf("Failed to launch init kernel (error code: %s)!\n", cudaGetErrorString(err));
				}
				cudaDeviceSynchronize();
			}
		}
		else {
			init_kernel<<<N*N/1024, 1024>>>(a, N);	
			err = cudaGetLastError();
			if (err != cudaSuccess)
			{
				printf("Failed to launch init kernel (error code: %s)!\n", cudaGetErrorString(err));
			}
			cudaDeviceSynchronize();
		}
		if(numThr < N){
			for(int i = 0; i < N; i += numThr){
				init_kernel<<<numThr/1024, 1024>>>(b + i, N);
				err = cudaGetLastError();
				if (err != cudaSuccess)
				{
					printf("Failed to launch init kernel (error code: %s)!\n", cudaGetErrorString(err));
				}
				cudaDeviceSynchronize();
			}
		}
		else {
			init_kernel<<<N/1024, 1024>>>(b, N);
			err = cudaGetLastError();
			if (err != cudaSuccess)
			{
				printf("Failed to launch init kernel (error code: %s)!\n", cudaGetErrorString(err));
			}	
			cudaDeviceSynchronize();
		}
	}

	gettimeofday(&tv0, &tz0);

    if(numThr > 1024)
		if(numThr > N * TILE_SIZE){
    		dim3 dimBlock(TILE_SIZE, TILE_SIZE);
			dim3 dimGrid(N/TILE_SIZE);
    		matvecmult_kernel<<<dimGrid, dimBlock>>>(a, b, c, N);
			err = cudaGetLastError();
			if (err != cudaSuccess)
			{
				printf("Failed to launch matvecmul kernel (error code: %s)!\n", cudaGetErrorString(err));
			}
    		cudaDeviceSynchronize();
		}
		else {
	    	dim3 dimBlock(TILE_SIZE, TILE_SIZE);
			dim3 dimGrid(numThr/(TILE_SIZE * TILE_SIZE));
			for(int i = 0; i < N; i += (numThr/(TILE_SIZE * TILE_SIZE)) * TILE_SIZE){
    			matvecmult_kernel<<<dimGrid, dimBlock>>>(a + i*N, b, c + i, N);
				err = cudaGetLastError();
				if (err != cudaSuccess)
				{
					printf("Failed to launch matvecmul kernel (error code: %s)!\n", cudaGetErrorString(err));
				}
    			cudaDeviceSynchronize();
			}
		}
	else {
		int tile = TILE_SIZE;
		while(TILE_SIZE * tile > numThr){
			tile /= 2;
		}
		// printf("Block Size %d  %d\n", TILE_SIZE, tile);
    	dim3 dimBlock(TILE_SIZE, tile);
		if(numThr > (N / tile)* (TILE_SIZE * tile)){
			dim3 dimGrid(N/tile);
    		matvecmult_kernel<<<dimGrid, dimBlock>>>(a, b, c, N);
			err = cudaGetLastError();
			if (err != cudaSuccess)
			{
				printf("Failed to launch matvecmul kernel (error code: %s)!\n", cudaGetErrorString(err));
			}
    		cudaDeviceSynchronize();
		}
		else {
			// printf("Grid Size : %d\n", numThr/(TILE_SIZE * tile));
			// printf("Number of times loop executed : %d\n", N/((numThr/(TILE_SIZE * tile)) * tile));
			dim3 dimGrid(numThr/(TILE_SIZE * tile));
			for(int i = 0; i < N; i += (numThr/(TILE_SIZE * tile)) * tile){
    			matvecmult_kernel<<<dimGrid, dimBlock>>>(a + i*N, b, c + i, N);
				err = cudaGetLastError();
				if (err != cudaSuccess)
				{
					printf("Failed to launch matvecmul kernel (error code: %s)!\n", cudaGetErrorString(err));
				}
    			cudaDeviceSynchronize();
			}
		}
	}
	gettimeofday(&tv2, &tz2);

	// srand(time(0));
	// int rowC = random() % N;

	float x = 0, error = 0;
	for(int rowC = 0; rowC < N; rowC++){
		x = 0;
		for (int i=0; i<N; i++) x += a[rowC*N + i]*b[i];
		error += fabs(c[rowC] - x);
	}
	error /= N;
	// float error = fabs(c[rowC] - x);
	printf("Error: %0.12f time: %ld microseconds\n", error, (tv2.tv_sec-tv0.tv_sec)*1000000+(tv2.tv_usec-tv0.tv_usec));
    return 0;
}

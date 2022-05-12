#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <cuda.h>
#include <sys/time.h>
// #include <cuPrintf.h>

// #define ROWS_A  (1<<13)
// #define COLS_A  (1<<13)
#define TILE_SIZE 32
// int N = 64;
// int numThr = 1;

__global__ void init_kernel (float *a, int N)
{
	int id = threadIdx.x + blockIdx.x*blockDim.x;

	a[id] = (float)id/(N*N);
	// printf("{%d}", id);
}

__global__ void matvecmult_kernel (float *a, float *b, float *c, float *d, float *e, int N) 	
{
	int c_row = blockIdx.x*blockDim.y + threadIdx.y;
	// int c_col = threadIdx.x;
	float x = 0;
	unsigned mask = 0xffffffff;
	__shared__ float as[TILE_SIZE][TILE_SIZE];
	__shared__ float bs[TILE_SIZE];
	
	for (int i=0; i<N/TILE_SIZE; i++) {
		as[threadIdx.y][threadIdx.x] = a[c_row*N + i*TILE_SIZE + threadIdx.x];
		bs[threadIdx.x] = b[(i*TILE_SIZE) + threadIdx.x];
		d[(blockIdx.x*blockDim.y + threadIdx.y)*N + (threadIdx.x + (i*TILE_SIZE))] = blockDim.y;//as[threadIdx.y][threadIdx.x];
		// e[(blockIdx.x*blockDim.y + threadIdx.y)*N + (threadIdx.x + (i*TILE_SIZE))] = bs[threadIdx.x];

		// d[threadIdx.y*TILE_SIZE + threadIdx.x] = as[threadIdx.y][threadIdx.x];
		__syncthreads();
		// for (j=0; j<TILE_SIZE; j++) {
		// 	x += (as[threadIdx.y][j]*bs[j]);
		// 	// d[0] = as[threadIdx.y][0]; d[1] = bs[0]; d[2] = as[threadIdx.y][1]; d[3] = bs[1]; 
		// }
		float y = 0;
        // if((threadIdx.x % warpSize) == 0)x += as[threadIdx.y][threadIdx.x] * bs[threadIdx.x];
		// else 
			y = as[threadIdx.y][threadIdx.x] * bs[threadIdx.x];
		// if(blockIdx.x == 0 && blockIdx.y == 0)d[0] = as[0][0]; d[1] = bs[0]; d[2] = as[0][1]; d[3] = bs[1]; d[4] = c_row;
		// d[(blockIdx.y*blockDim.y + threadIdx.y)*N + (blockIdx.x*blockDim.x + threadIdx.x) + 1] = bs[threadIdx.x];
		
		__syncthreads();
		for (int j=warpSize/2; j>0; j=j/2) y += __shfl_down_sync(mask, y, j);
		if((threadIdx.x % warpSize) == 0){
			//  d[c_row*N + i] = y; 
			 x += y; 
			//  e[c_row*N + i] = x;
		}
		// a[c_row*COLS_A + i*TILE_SIZE + threadIdx.x] = as[threadIdx.y][threadIdx.x];
		__syncthreads();
	}
    // float val = 0;
	__syncthreads();
	if ((threadIdx.x % warpSize) == 0) c[c_row] = x;

}


int main(int argc, char *argv[]){
    float *a, *b, *c;
	float *d, *e;
	struct timeval tv0, tv2;
	struct timezone tz0, tz2;
	int N, numThr;
    cudaMallocManaged((void**)&N, sizeof(int));
	N = atoi(argv[1]);
	numThr = atoi(argv[2]);
	// printf("N : %d\n", N);

    cudaMallocManaged((void**)&a, sizeof(float)*N*N);
	cudaMallocManaged((void**)&b, sizeof(float)*N);
	cudaMallocManaged((void**)&c, sizeof(float)*N);
	cudaMallocManaged((void**)&d, sizeof(float)*N*N);
	cudaMallocManaged((void**)&e, sizeof(float)*N*N);

    int device = -1;
        cudaGetDevice(&device);
	cudaMemAdvise(c, sizeof(float)*N, cudaMemAdviseSetPreferredLocation, device);
	cudaMemPrefetchAsync(a, sizeof(float)*N*N, device, NULL);
	cudaMemPrefetchAsync(b, sizeof(float)*N, device, NULL);
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
	// for(int i = 2048*2048; i < 2048*2048 + 10; i++){printf("%0.12f ", a[i]);} printf("\n");
    // init_kernel<<<N*N/1024, 1024>>>(a, N);	
	// init_kernel<<<N/1024, 1024>>>(b, N);

	// init_kernel<<<N*N/1024, 1024>>>(a, N);	
	// init_kernel<<<N/1024, 1024>>>(b, N);


	// err = cudaGetLastError();
    // if (err != cudaSuccess)
    // {
    //     printf("Failed to launch init kernel (error code: %s)!\n", cudaGetErrorString(err));
    // }

    // cudaDeviceSynchronize();
	
	srand(time(0));
	int rowC = random() % N;
	

	// for(int i = 0; i < 1; i++){
	// 	for(int j = 0; j < 32; j++){
    //     	printf("%0.2f ", a[rowC*N + i*N + j]);
	// 	}
    //     printf("\n");
    // }
	// printf("\n_____________________________\n");
    // for(int i = 0; i < 10; i++){
	// 	for(int j = 0; j < 10; j++){
    //     	printf("%0.2f ", b[i*N + j]);
	// 	}
    //     printf("\n");
    // }
	// printf("\n_____________________________\n");

	gettimeofday(&tv0, &tz0);

	if(numThr > 1024)
		if(numThr > N * TILE_SIZE){
    		dim3 dimBlock(TILE_SIZE, TILE_SIZE);
			dim3 dimGrid(N/TILE_SIZE);
    		matvecmult_kernel<<<dimGrid, dimBlock>>>(a, b, c, d, e, N);
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
    			matvecmult_kernel<<<dimGrid, dimBlock>>>(a + i*N, b, c + i, d + i*N, e + i*N, N);
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
    		matvecmult_kernel<<<dimGrid, dimBlock>>>(a, b, c, d, e, N);
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
    			matvecmult_kernel<<<dimGrid, dimBlock>>>(a + i*N, b, c + i, d + i*N, e + i*N, N);
				err = cudaGetLastError();
				if (err != cudaSuccess)
				{
					printf("Failed to launch matvecmul kernel (error code: %s)!\n", cudaGetErrorString(err));
				}
    			cudaDeviceSynchronize();
			}
		}
	}
	

	// dim3 dimGrid(N/TILE_SIZE);

    // matvecmult_kernel<<<dimGrid, dimBlock>>>(a, b, c, N);
	// err = cudaGetLastError();
	// if (err != cudaSuccess)
    // {
    //     printf("Failed to launch matvecmul kernel (error code: %s)!\n", cudaGetErrorString(err));
    // }
    // cudaDeviceSynchronize();
	gettimeofday(&tv2, &tz2);

	// rowC = 500;
	// printf("Random Row : %d\n", rowC);
	float x = 0;
	for (int i=0; i<N; i++) x += a[rowC*N + i]*b[i];
	float error = fabs(c[rowC] - x);
	printf("Error: %0.12f, computed value: %0.12f, actual value: %0.12f, time: %ld microseconds\n", error, c[rowC], x, (tv2.tv_sec-tv0.tv_sec)*1000000+(tv2.tv_usec-tv0.tv_usec));


	int start_i = 0, end_i = 71, start_j = 0, end_j = 32;
	printf("###### d ######\n");
    for(int i = start_i; i < end_i; i++){
		for(int j = start_j; j < end_j; j++){
        	printf("%0.12f ", d[rowC*N + i*N + j]);
		}
		// if(d[])
        printf("\n");
    }
	// printf("\n_____________________________\n");
	// for(int i = 0; i < N/TILE_SIZE; i++){
	// 	float bla = 0;
	// 	for(int j = 0; j < TILE_SIZE; j++){
	// 		bla += a[rowC*N + i*TILE_SIZE + j]*b[i*TILE_SIZE + j];
	// 	}
	// 	printf("%0.2f ", bla);
	// }
	// float bla2 = 0;
	// printf("\n________________cpu_____________\n");
	// for(int i = 0; i < N/TILE_SIZE; i++){
	// 	float tmp2 = 0;
	// 	for(int j = 0; j < TILE_SIZE; j++){
	// 		tmp2 += a[rowC*N + i*TILE_SIZE + j]*b[i*TILE_SIZE + j];
	// 	}
	// 	bla2 = bla2 + tmp2;
	// 	printf("%0.2f ", bla2);
	// }
	// printf("\n###### e ######\n");
    // for(int i = start_i; i < end_i; i++){
	// 	for(int j = start_j; j < end_j; j++){
    //     	printf("%0.2f ", e[rowC*N + i*N + j]);
	// 	}
    //     printf("\n");
    // }
	// printf("\n_____________________________\n");

	// float tmp = 0;
	// for(int i = start_i; i < end_i; i++){
	// 	for(int j = start_j; j < end_j; j++){
	// 		if(e[i*N+j] != b[j]){printf("%d %d\n", i, j); break;}
    //     	// tmp += d[i*N+j] * e[i*N+j];
	// 	}
    // }
	// printf("tmp = %0.2f\n", tmp);

	// for(int i = 0; i < COLS_A; i++){
    //     if(i%ROWS_A == 0)printf("\n");
    //     printf("%f ", b[i]);
    // }
	printf("\n_____________________________\n");
    for(int i = start_i; i < end_i; i++){
		// for(int j = start_j; j < end_j; j++){
        	printf("%0.12f ", c[i]);
		// }
        // printf("\n");
    }
	// printf("\n_____________________________\n");

	// for(int i = 0; i < 2; i++){for(int j = 0; j < 2; j++){for(int k = 0; k < 2; k++){for(int l = 0; l < 2; l++){printf("{%d %d} ", )}} }}

    return 0;
}

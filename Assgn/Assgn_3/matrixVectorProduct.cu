#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <cuda.h>
#include <sys/time.h>
// #include <cuPrintf.h>

// #define ROWS_A  (1<<13)
// #define COLS_A  (1<<13)
#define TILE_SIZE 16
// int N = 64;
int numThr = 1;

__global__ void init_kernel (float *a, int N)
{
	int id = threadIdx.x + blockIdx.x*blockDim.x;

	a[id] = (float)id/(N*N);
	// printf("{%d}", id);
}

__global__ void matvecmult_kernel (float *a, float *b, float *c, int N) 	
{
	int c_row = blockIdx.y*blockDim.y + threadIdx.y;
	// int c_col = threadIdx.x;
	int i, j;
	float x = 0;
	__shared__ float as[TILE_SIZE][TILE_SIZE];
	__shared__ float bs[TILE_SIZE];
	
	for (i=0; i<N/TILE_SIZE; i++) {
		as[threadIdx.y][threadIdx.x] = a[c_row*N + i*TILE_SIZE + threadIdx.x];
		bs[threadIdx.x] = b[(i*TILE_SIZE) + threadIdx.x];
		// d[threadIdx.y*TILE_SIZE + threadIdx.x] = as[threadIdx.y][threadIdx.x];
		__syncthreads();
		for (j=0; j<TILE_SIZE; j++) {
			x += (as[threadIdx.y][j]*bs[j]);
			// d[0] = as[threadIdx.y][0]; d[1] = bs[0]; d[2] = as[threadIdx.y][1]; d[3] = bs[1]; 
		}
		// a[c_row*COLS_A + i*TILE_SIZE + threadIdx.x] = as[threadIdx.y][threadIdx.x];
		__syncthreads();
	}

	c[c_row] = x;
}


int main(int argc, char *argv[]){
    float *a, *b, *c;
	struct timeval tv0, tv2;
	struct timezone tz0, tz2;
	srand(time(0));
	// float *d;
    // cudaMallocManaged((void**)&d, sizeof(float)*TILE_SIZE*TILE_SIZE);
	int N;
    cudaMallocManaged((void**)&N, sizeof(int));
	N = atoi(argv[1]);

    cudaMallocManaged((void**)&a, sizeof(float)*N*N);
	cudaMallocManaged((void**)&b, sizeof(float)*N*1);
	cudaMallocManaged((void**)&c, sizeof(float)*N*1);

    int device = -1;
        cudaGetDevice(&device);
	cudaMemAdvise(c, sizeof(float)*N, cudaMemAdviseSetPreferredLocation, device);

    init_kernel<<<N*N/1024, 1024>>>(a, N);	
	init_kernel<<<N/1024, 1024>>>(b, N);
	cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("Failed to launch kernel (error code: %s)!\n", cudaGetErrorString(err));
    }

    cudaDeviceSynchronize();
	
	for(int i = 0; i < 5; i++){
        if(i%N == 0){printf("\n");}
        printf("%0.12f ", a[i]);
    }
	printf("\n_____________________________\n");
    // for(int i = 0; i < 5; i++){
    //     if(i%N == 0)printf("\n");
    //     printf("%0.12f ", b[i]);
    // }
	// printf("\n_____________________________\n");

	gettimeofday(&tv0, &tz0);

    dim3 dimBlock(TILE_SIZE, TILE_SIZE);
	dim3 dimGrid(N/TILE_SIZE, N/TILE_SIZE);

    matvecmult_kernel<<<dimGrid, dimBlock>>>(a, b, c, N);
    cudaDeviceSynchronize();
	gettimeofday(&tv2, &tz2);

	int rowC = random() % N;

	float x = 0;

	for (int i=0; i<N; i++) x += a[rowC*N + i]*b[i];
	float error = fabs(c[rowC] - x);
	printf("Error: %0.12f, computed value: %0.12f, actual value: %0.12f, time: %ld microseconds\n", error, c[rowC], x, (tv2.tv_sec-tv0.tv_sec)*1000000+(tv2.tv_usec-tv0.tv_usec));

    // for(int i = 0; i < TILE_SIZE*TILE_SIZE; i++){
    //     if(i%TILE_SIZE == 0)printf("\n");
    //     printf("%f ", d[i]);
    // }
	// printf("\n_____________________________\n");
    // for(int i = 0; i < COLS_A; i++){
    //     if(i%ROWS_A == 0)printf("\n");
    //     printf("%f ", b[i]);
    // }
	// printf("\n_____________________________\n");
    // for(int i = 0; i < COLS_A; i++){
    //     if(i%ROWS_A == 0)printf("\n");
    //     printf("%f ", c[i]);
    // }

    return 0;
}

// #include <stdio.h>
// #include <stdlib.h>
// #include <assert.h>
// #include <cuda.h>
// #include <sys/time.h>

// #define ROWS_A (1 << 6)
// #define COLS_A (1 << 6)
// #define ROWS_B COLS_A
// #define COLS_B (1 << 6)
// #define ROWS_C ROWS_A
// #define COLS_C COLS_B

// __global__ void init_kernel (int *a)
// {
// 	int id = threadIdx.x + blockIdx.x*blockDim.x;
// 	a[id] = id;
// }


// int main (int argc, char *argv[])
// {
// 	// int i;
// 	int *a, *b, *c;

// 	cudaMallocManaged((void**)&a, sizeof(float)*ROWS_A*COLS_A);

// 	init_kernel<<<ROWS_A*COLS_A/1024, 1024>>>(a);
// 	cudaError_t err = cudaGetLastError();
//     if (err != cudaSuccess)
//     {
//         printf("Failed to launch add kernel (error code: %s)!\n", cudaGetErrorString(err));
//     }

// 	cudaDeviceSynchronize();

// 	for(int i = 0; i < 10; i++){
// 		printf("%d ", a[i]);
// 		printf("\n");
// 	}
// 	printf("\n%f\n", (float)ROWS_C*COLS_C);
// 	return 0;
// }
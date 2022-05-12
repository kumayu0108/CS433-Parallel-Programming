#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <cuda.h>
#include <sys/time.h>
#include <omp.h>

#ifdef GPU_MEMORY_FITTING
#define SIZE (1<<29)
#else
#define SIZE (1<<30)
#endif
#define CPU_LOAD_FACTOR_NUMER 1
#define CPU_LOAD_FACTOR_DENOM 2

__global__ void init (float *a, float *b, int span, int start)
{
        int i;
        int id = threadIdx.x + blockIdx.x*blockDim.x;

        for (i=span*id+start; i<span*(id+1)+start; i++) {
                a[i] = 1;
		b[i] = 1.5;
        }
}

__global__ void vadd_kernel (float *a, float *b, float *c, int span, int start)
{
	int i;
	int id = threadIdx.x + blockIdx.x*blockDim.x;
	
	for (i=span*id+start; i<span*(id+1)+start; i++) {
		c[i] = a[i] + b[i];
	}
}	

int main (int argc, char *argv[])
{
	float *a, *b, *c;
	int nthreads_cpu, nthreads_gpu, i;
	struct timeval tv0, tv2;
	struct timezone tz0, tz2;

	if (argc != 3) {
		printf ("Need number of CPU and GPU threads (in that order).\n");
		exit(1);
	}
	nthreads_cpu = atoi(argv[1]);
	assert((nthreads_cpu & (nthreads_cpu - 1)) == 0);
	nthreads_gpu = atoi(argv[2]);
        assert((nthreads_gpu & (nthreads_gpu - 1)) == 0);

	assert((SIZE % CPU_LOAD_FACTOR_DENOM) == 0);

	cudaMallocManaged((void**)&a, sizeof(float)*SIZE);
	cudaMallocManaged((void**)&b, sizeof(float)*SIZE);
	cudaMallocManaged((void**)&c, sizeof(float)*SIZE);

	if (nthreads_cpu && nthreads_gpu) {
		for (i=0; i<CPU_LOAD_FACTOR_NUMER*(SIZE/CPU_LOAD_FACTOR_DENOM); i++) {
			a[i] = 1;
			b[i] = 1.5;
		}

		if (nthreads_gpu < 16) {
                	init<<<1, nthreads_gpu>>>(a, b, (CPU_LOAD_FACTOR_DENOM-CPU_LOAD_FACTOR_NUMER)*(SIZE/(nthreads_gpu*CPU_LOAD_FACTOR_DENOM)), CPU_LOAD_FACTOR_NUMER*(SIZE/CPU_LOAD_FACTOR_DENOM));
		}
		else {
                	init<<<nthreads_gpu/8, 8>>>(a, b, (CPU_LOAD_FACTOR_DENOM-CPU_LOAD_FACTOR_NUMER)*(SIZE/(nthreads_gpu*CPU_LOAD_FACTOR_DENOM)), CPU_LOAD_FACTOR_NUMER*(SIZE/CPU_LOAD_FACTOR_DENOM));
		}
		cudaDeviceSynchronize();
	}
	else if (nthreads_gpu) {
		if (nthreads_gpu < 16) {
                        init<<<1, nthreads_gpu>>>(a, b, SIZE/nthreads_gpu, 0);
                }
                else {
                        init<<<nthreads_gpu/8, 8>>>(a, b, SIZE/nthreads_gpu, 0);
                }
		cudaDeviceSynchronize();
	}
	else if (nthreads_cpu) {
		for (i=0; i<SIZE; i++) {
                        a[i] = 1;
                        b[i] = 1.5;
                }
	}
	else {
		printf("Both CPU and GPU threads cannot be zero. Abprting...\n");
		exit(0);
	}

	gettimeofday(&tv0, &tz0);

	if (nthreads_cpu && nthreads_gpu) {
		if (nthreads_gpu < 16) {
			vadd_kernel<<<1, nthreads_gpu>>>(a, b, c, (CPU_LOAD_FACTOR_DENOM-CPU_LOAD_FACTOR_NUMER)*(SIZE/(nthreads_gpu*CPU_LOAD_FACTOR_DENOM)), CPU_LOAD_FACTOR_NUMER*(SIZE/CPU_LOAD_FACTOR_DENOM));
		}
		else {
			vadd_kernel<<<nthreads_gpu/8, 8>>>(a, b, c, (CPU_LOAD_FACTOR_DENOM-CPU_LOAD_FACTOR_NUMER)*(SIZE/(nthreads_gpu*CPU_LOAD_FACTOR_DENOM)), CPU_LOAD_FACTOR_NUMER*(SIZE/CPU_LOAD_FACTOR_DENOM));
		}

#pragma omp parallel for num_threads(nthreads_cpu)
		for (i=0; i<CPU_LOAD_FACTOR_NUMER*(SIZE/CPU_LOAD_FACTOR_DENOM); i++) c[i] = a[i] + b[i];

		cudaDeviceSynchronize();
	}
	else if (nthreads_gpu) {
		if (nthreads_gpu < 16) {
                        vadd_kernel<<<1, nthreads_gpu>>>(a, b, c, SIZE/nthreads_gpu, 0);
                }
                else {
                        vadd_kernel<<<nthreads_gpu/8, 8>>>(a, b, c, SIZE/nthreads_gpu, 0);
                }
		cudaDeviceSynchronize();
	}
	else if (nthreads_cpu) {
#pragma omp parallel for num_threads(nthreads_cpu)
                for (i=0; i<SIZE; i++) c[i] = a[i] + b[i];
	}
	else {
                printf("Both CPU and GPU threads cannot be zero. Abprting...\n");
                exit(0);
        }

	gettimeofday(&tv2, &tz2);
	
	printf("Random element: %lf, time: %ld microseconds\n", c[random() % SIZE], (tv2.tv_sec-tv0.tv_sec)*1000000+(tv2.tv_usec-tv0.tv_usec));
	return 0;
}

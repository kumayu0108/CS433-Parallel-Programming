#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <sys/time.h>
#include <assert.h>

#define SIZE (1<<30)

double sum = 0;
float *a;
int nthreads;
double *private_sum;

void solver (void)
{
	int i, id = omp_get_thread_num();
	private_sum[id] = 0;
	
	for (i=(SIZE/nthreads)*id; i<(SIZE/nthreads)*(id+1); i++) {
		private_sum[id] += (a[i]*a[i]);
	}

#pragma omp critical
	sum += private_sum[id];
}	

int main (int argc, char *argv[])
{
	int i;
	struct timeval tv0, tv1;
	struct timezone tz0, tz1;

	if (argc != 2) {
		printf ("Need number of threads.\n");
		exit(1);
	}
	nthreads = atoi(argv[1]);
	private_sum = (double*)malloc(nthreads*sizeof(double));

	a = (float*)malloc(sizeof(float)*SIZE);
	assert(a != NULL);
	for (i=0; i<SIZE; i++) a[i] = 1;

	gettimeofday(&tv0, &tz0);

# pragma omp parallel num_threads (nthreads)
        solver();
   	
	gettimeofday(&tv1, &tz1);
	printf("SUM: %lf, time: %ld microseconds\n", sum, (tv1.tv_sec-tv0.tv_sec)*1000000+(tv1.tv_usec-tv0.tv_usec));
	return 0;
}
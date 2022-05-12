#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <sys/time.h>
#include <assert.h>

#define SIZE (1<<20)
#define ITERS 1000

double sum = 0;
float *a;
int nthreads;
struct timeval tv0, tv1;
struct timezone tz0, tz1;

void solver (void)
{
	int i, j, id = omp_get_thread_num();
	if (id == 0) {
		gettimeofday(&tv0, &tz0);
		for (i=0; i<ITERS; i++) {
			for (j=0; j<SIZE/2; j++) sum += a[j];
		}
		gettimeofday(&tv1, &tz1);
	}
	else {
		for (i=0; i<ITERS; i++) {
			for (j=SIZE/2; j<SIZE; j++) a[j]++;
		}
	}
}	

int main (int argc, char *argv[])
{
	int i;

	if (argc != 2) {
		printf ("Need number of threads.\n");
		exit(1);
	}
	nthreads = atoi(argv[1]);

	a = (float*)malloc(sizeof(float)*SIZE);
	assert(a != NULL);
	for (i=0; i<SIZE; i++) a[i] = 1;

# pragma omp parallel num_threads (nthreads)
        solver();
   	
	printf("SUM: %lf, time: %ld microseconds\n", sum, (tv1.tv_sec-tv0.tv_sec)*1000000+(tv1.tv_usec-tv0.tv_usec));
	return 0;
}
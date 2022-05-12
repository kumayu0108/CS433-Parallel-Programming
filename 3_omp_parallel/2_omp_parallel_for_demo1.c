#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <sys/time.h>
#include <assert.h>

#define SIZE (1<<30)

float *a, *b, *c;
int nthreads;

int main (int argc, char *argv[])
{
	int i;
	struct timeval tv0, tv1;
	struct timezone tz0, tz1;

	nthreads = omp_get_num_procs();
	omp_set_num_threads(nthreads);
	printf("nthreads: %d\n", nthreads);

	a = (float*)malloc(sizeof(float)*SIZE);
	assert(a != NULL);
	for (i=0; i<SIZE; i++) a[i] = 1;

	b = (float*)malloc(sizeof(float)*SIZE);
        assert(b != NULL);
        for (i=0; i<SIZE; i++) b[i] = 1;

	c = (float*)malloc(sizeof(float)*SIZE);
        assert(c != NULL);

	gettimeofday(&tv0, &tz0);

#pragma omp parallel for
        for (i=0; i<SIZE; i++) c[i] = a[i] + b[i];

	gettimeofday(&tv1, &tz1);
	printf("Random element: %f, time: %ld microseconds\n", c[random()%SIZE], (tv1.tv_sec-tv0.tv_sec)*1000000+(tv1.tv_usec-tv0.tv_usec));
	return 0;
}
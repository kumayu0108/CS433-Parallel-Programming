#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <sys/time.h>
#include <assert.h>

#define SIZE (1<<30)

float *a, *b, total=0;

int main (int argc, char *argv[])
{
	int i;
	struct timeval tv0, tv1;
	struct timezone tz0, tz1;

	a = (float*)malloc(sizeof(float)*SIZE);
	assert(a != NULL);
	for (i=0; i<SIZE; i++) a[i] = 1;

	b = (float*)malloc(sizeof(float)*SIZE);
        assert(b != NULL);
        for (i=0; i<SIZE; i++) b[i] = 1;

	gettimeofday(&tv0, &tz0);

#pragma omp simd reduction(+:total)
        for (i=0; i<SIZE; i++) total += (a[i] + b[i]);
   	
	gettimeofday(&tv1, &tz1);
	printf("Total: %f, time: %ld microseconds\n", total, (tv1.tv_sec-tv0.tv_sec)*1000000+(tv1.tv_usec-tv0.tv_usec));
	return 0;
}

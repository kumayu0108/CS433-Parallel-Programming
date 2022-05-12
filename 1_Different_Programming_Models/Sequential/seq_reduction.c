#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <assert.h>

#define SIZE (1<<30)

double sum = 0;
float *a;

int main (int argc, char *argv[])
{
	int i;
	struct timeval tv0, tv1;
	struct timezone tz0, tz1;

	a = (float*)malloc(sizeof(float)*SIZE);
	assert(a != NULL);
	for (i=0; i<SIZE; i++) a[i] = 1;

	gettimeofday(&tv0, &tz0);
	for (i=0; i<SIZE; i++) sum += (a[i]*a[i]);
	gettimeofday(&tv1, &tz1);

	printf("SUM: %lf, time: %ld microseconds\n", sum, (tv1.tv_sec-tv0.tv_sec)*1000000+(tv1.tv_usec-tv0.tv_usec));
	return 0;
}
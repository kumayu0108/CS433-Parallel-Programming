#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <sys/time.h>
#include <assert.h>

#define n (1 << 30)

int main (int argc, char *argv[])
{
	int i;
	struct timeval tv0, tv1;
	struct timezone tz0, tz1;
	int nthreads;
	unsigned long long fib[64];

	if (argc != 2) {
		printf ("Need number of threads.\n");
		exit(1);
	}
	nthreads = atoi(argv[1]);

	gettimeofday(&tv0, &tz0);

	double factor;
	double sum = 0;

#pragma omp parallel for num_threads(nthreads) private(factor) schedule(static,1) reduction(+:sum)
	for (i=0; i<n; i++) {
		factor = (i % 2) ? -1.0 : 1.0;
   		sum += factor/(2*i+1);
	}
	double pi = 4*sum;

	gettimeofday(&tv1, &tz1);

	printf("Pi=%.15lf, Time: %ld microseconds\n", pi, (tv1.tv_sec-tv0.tv_sec)*1000000+(tv1.tv_usec-tv0.tv_usec));
	return 0;
}

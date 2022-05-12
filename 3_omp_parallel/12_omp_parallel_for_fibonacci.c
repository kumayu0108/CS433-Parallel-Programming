#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <sys/time.h>
#include <assert.h>

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
	fib[0] = fib[1] = 1;
	//for (i=2; i<64; i++) fib[i] = 0;
#pragma omp parallel for num_threads(nthreads)
	for (i=2; i<64; i++) fib[i] = fib[i-1]+fib[i-2];

	gettimeofday(&tv1, &tz1);

	for (i=0; i<64; i++) printf("%llu ", fib[i]);
	printf("\nTime: %ld microseconds\n", (tv1.tv_sec-tv0.tv_sec)*1000000+(tv1.tv_usec-tv0.tv_usec));
	return 0;
}

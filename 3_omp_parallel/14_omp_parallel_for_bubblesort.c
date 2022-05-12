#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <sys/time.h>
#include <assert.h>

#define SIZE (1 << 6)

int main (int argc, char *argv[])
{
	int i, temp;
	struct timeval tv0, tv1;
	struct timezone tz0, tz1;
	int nthreads;
	int *a = (int*)malloc(SIZE*sizeof(int));
	assert(a != NULL);

	if (argc != 2) {
		printf ("Need number of threads.\n");
		exit(1);
	}
	nthreads = atoi(argv[1]);

	for (i=0; i<SIZE; i++) a[i] = SIZE - i;

	gettimeofday(&tv0, &tz0);
#pragma omp parallel for num_threads(nthreads) private(i,temp)
	for (int length=SIZE; length>=2; length--) {
        	for (i=0; i<length-1; i++) {
      			if (a[i] > a[i+1]) {    	// S1
         			temp = a[i];       	// S2
         			a[i] = a[i+1];      	// S3
         			a[i+1] = temp;    	// S4
      			}
   		}
	}

	gettimeofday(&tv1, &tz1);

	for (i=0; i<SIZE; i++) printf("%d ", a[i]);
	printf("\nTime: %ld microseconds\n", (tv1.tv_sec-tv0.tv_sec)*1000000+(tv1.tv_usec-tv0.tv_usec));
	return 0;
}

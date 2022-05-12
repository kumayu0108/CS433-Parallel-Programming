#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <sys/time.h>
#include <assert.h>

#define SIZE (1<<30)
#define SEGMENT_SIZE (1<<20)
#define INFINITY 0x7fffffffffffffffULL

int minindex;
int *a;
int nthreads;
long long globalmin = INFINITY;

void solver (void)
{
   int i, id = omp_get_thread_num();
   long long firstsum = 0, localmin = INFINITY;
   int localminindex;
   int ntasks = SIZE - SEGMENT_SIZE;  // Number of tasks is actually SIZE - SEGMENT_SIZE + 1
   int nstasksPerThread = ntasks/nthreads;
   int start = id*nstasksPerThread;
   int end = (id+1)*nstasksPerThread;
   if (!id) end++;  // Let T0 do one extra task to take care of the left-over
   else {           // Adjust the other threads' start and end tasks
      start++;
      end++;
   }

   for (i=start; i<start+SEGMENT_SIZE; i++) firstsum += a[i];  // Sum for first task

   if (firstsum < localmin) {
      localmin = firstsum;
      localminindex = start;
   }

   // Explore the remaining tasks
   int startptr = start+1;
   int endptr = startptr+SEGMENT_SIZE-1;
   while (startptr < end) {
      assert(endptr < SIZE);
      firstsum = firstsum - a[startptr-1] + a[endptr];
      if (firstsum < localmin) {
         localmin = firstsum;
         localminindex = startptr;
      }
      startptr++;
      endptr++;
   }
   if (id == (nthreads-1)) assert(endptr == SIZE);

   if (localmin < globalmin) {
#pragma omp critical
      {
         if (localmin < globalmin) {
	    globalmin = localmin;
	    minindex = localminindex;
	 }
      }
   }
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

	a = (int*)malloc(sizeof(int)*SIZE);
	assert(a != NULL);
	for (i=0; i<SIZE; i++) {
	   a[i] = (random() % 10) - 5;
	}

	gettimeofday(&tv0, &tz0);

# pragma omp parallel num_threads (nthreads)
        solver();
   	
	gettimeofday(&tv1, &tz1);
	printf("Min sum: %lld, start index: %d, time: %ld microseconds\n", globalmin, minindex, (tv1.tv_sec-tv0.tv_sec)*1000000+(tv1.tv_usec-tv0.tv_usec));
	return 0;
}
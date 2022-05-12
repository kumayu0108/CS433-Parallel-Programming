#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <sys/time.h>
#include <assert.h>
#include <math.h>

#define TOL 1e-5
#define ITER_LIMIT 1000

int P, n;
float **A, diff = 0.0;
pthread_barrier_t barrier;
pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;

typedef struct param_s {
   FILE *fp;
   int id;
} param_t;

void Initialize (float **X)
{
   for (int i=0; i<n+2; i++) for (int j=0; j<n+2; j++) X[i][j] = ((float)(random() % 100)/100.0);
}

void* Solve (void *param)
{
   FILE *fp = ((param_t*)param)->fp;
   int pid = ((param_t*)param)->id;
   int done = 0, iters = 0;

   float temp, local_diff;

   while (!done) {
      local_diff = 0.0;
      if (!pid) diff = 0.0;
      pthread_barrier_wait (&barrier);
      for (int i = pid*(n/P)+1; i < ((pid+1)*(n/P))+1; i++) {
         for (int j = 1; j < n+1; j++) {
            temp = A[i][j];
            A[i][j] = 0.2*(A[i][j] + A[i][j-1] + A[i][j+1] + A[i+1][j] + A[i-1][j]);
	    local_diff += fabs(A[i][j] - temp);
	 }
      }
      pthread_mutex_lock(&mutex);
      diff += local_diff;
      pthread_mutex_unlock(&mutex);
      pthread_barrier_wait (&barrier);
      iters++;
      if ((diff/(n*n) < TOL) || (iters == ITER_LIMIT)) done = 1;
      pthread_barrier_wait (&barrier);

      if (!pid) fprintf(fp, "[%d] diff = %.10f\n", iters, diff/(n*n));
   }
}

int main (int argc, char **argv)
{
   struct timeval tv0, tv1;
   struct timezone tz0, tz1;
   char buffer[64];

   pthread_t *tid;
   pthread_attr_t attr;

   param_t *param_array;

   if (argc != 3) {
      printf("Need grid size (n) and number of threads (P).\nAborting...\n");
      exit(1);
   }

   n = atoi(argv[1]);
   P = atoi(argv[2]);

   A = (float**)malloc((n+2)*sizeof(float*));
   assert(A != NULL);
   for (int i=0; i<n+2; i++) {
      A[i] = (float*)malloc((n+2)*sizeof(float));
      assert(A[i] != NULL);
   }

   Initialize(A);

   sprintf(buffer, "pthread_gsblockrow_outfile_%d_%d.txt", n, P);
   FILE *fp = fopen(buffer, "w");

   pthread_barrier_init (&barrier, NULL, P);
   tid = (pthread_t*)malloc(P*sizeof(pthread_t));
   param_array = (param_t*)malloc(P*sizeof(param_t));
   for (int i=0; i<P; i++) {
      param_array[i].fp = fp;
      param_array[i].id = i;
   }

   pthread_attr_init(&attr);

   gettimeofday(&tv0, &tz0);

   for (int i=1; i<P; i++) {
      pthread_create(&tid[i], &attr, Solve, &param_array[i]);
   }

   Solve(&param_array[0]);

   for (int i=1; i<P; i++) {
      pthread_join(tid[i], NULL);
   }

   gettimeofday(&tv1, &tz1);

   fclose(fp);
   printf("Time: %ld microseconds\n", (tv1.tv_sec-tv0.tv_sec)*1000000+(tv1.tv_usec-tv0.tv_usec));

   return 0;
}
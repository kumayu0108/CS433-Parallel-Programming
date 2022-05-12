#include <mpi.h>
#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <assert.h>
#include <sys/time.h>
#include <math.h>

#define TOL 1e-5
#define ITER_LIMIT 1000

#define ROW 99
#define DIFF 98
#define DONE 97

void Initialize (int n, int P, float **X)
{
   for (int j=0; j<n+2; j++) X[0][j] = (j % 100)/100.0;
   for (int j=0; j<n+2; j++) X[n/P+1][j] = (j % 100)/100.0;
   for (int i=1; i<n/P+1; i++) for (int j=0; j<n+2; j++) X[i][j] = ((float)(random() % 100)/100.0);
}

int main(int argc, char ** argv)
{
   int pid, P, done, n, iters = 0;
   float /*tempdiff,*/ local_diff, temp, **A, total_diff;
   FILE *fp;
   char buffer[64];

   char hostname[256];
   gethostname(hostname, 256);

   struct timeval tv0, tv1;
   struct timezone tz0, tz1;

   MPI_Init(&argc, &argv);

   MPI_Comm_rank(MPI_COMM_WORLD, &pid);
   MPI_Comm_size(MPI_COMM_WORLD, &P);

   n = atoi(argv[1]);
   //tempdiff = 0.0;
   done = 0;

   A = (float **) malloc ((n/P+2) * sizeof(float *));
   for (int i=0; i < n/P+2; i++) {
       A[i] = (float *) malloc (sizeof(float) * (n+2));
   }

   Initialize(n, P, A);

   if (!pid) {
      sprintf(buffer, "mpi_gsblockrow_allreduce_outfile_%d_%d.txt", n, P);
      fp = fopen(buffer, "w");
      gettimeofday(&tv0, &tz0);
   }

   // Uncomment the following line to see the placement of processes
   //printf("%s: Hi, I am %d about to start computation.\n", hostname, pid);

   while (!done) {
      local_diff = 0.0;
      /* MPI_CHAR means raw byte format */
      if (pid) {  /* send my first row up */
         MPI_Send(&A[1][1], n*sizeof(float), MPI_CHAR, pid-1, ROW, MPI_COMM_WORLD);
      }
      if (pid != P-1) {  /* recv last row */
         MPI_Recv(&A[n/P+1][1], n*sizeof(float), MPI_CHAR, pid+1, ROW, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      }
      if (pid != P-1) {  /* send last row down */
         MPI_Send(&A[n/P][1], n*sizeof(float), MPI_CHAR, pid+1, ROW, MPI_COMM_WORLD);
      }
      if (pid) {  /* recv first row from above */
         MPI_Recv(&A[0][1], n*sizeof(float), MPI_CHAR, pid-1, ROW, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      }
      for (int i=1; i <= n/P; i++) for (int j=1; j <= n; j++) {
         temp = A[i][j];
         A[i][j] = 0.2 * (A[i][j] + A[i][j-1] + A[i-1][j] + A[i][j+1] + A[i+1][j]);
         local_diff += fabs(A[i][j] - temp);
      }
      MPI_Allreduce(&local_diff, &total_diff, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
      iters++;
      if ((total_diff/(n*n) < TOL) ||
          (iters == ITER_LIMIT)) done = 1;
      if (!pid) fprintf(fp, "[%d] diff = %.10f\n", iters, total_diff/(n*n));
#if 0
      if (pid) {  /* tell P0 my diff */
         MPI_Send(&local_diff, sizeof(float), MPI_CHAR, 0, DIFF, MPI_COMM_WORLD);
         MPI_Recv(&done, sizeof(int), MPI_CHAR, 0, DONE, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      } 
      else {  /* recv from all and add up */
         for (int i=1; i < P; i++) {
            MPI_Recv(&tempdiff, sizeof(float), MPI_CHAR, MPI_ANY_SOURCE, DIFF, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            local_diff += tempdiff;
         } 
         iters++;
         if ((local_diff/(n*n) < TOL) ||
             (iters == ITER_LIMIT)) done = 1;
         for (int i=1; i < P; i++) { 
            /* tell all if done */
            MPI_Send(&done, sizeof(int), MPI_CHAR, i, DONE, MPI_COMM_WORLD);
         }
	 fprintf(fp, "[%d] diff = %.10f\n", iters, local_diff/(n*n));
      }
#endif
   }
   if (!pid) {
      gettimeofday(&tv1, &tz1);
      fclose(fp);
      printf("Time: %ld microseconds.\n", (tv1.tv_sec-tv0.tv_sec)*1000000+(tv1.tv_usec-tv0.tv_usec));
   }
   MPI_Finalize();
   return 0;
}

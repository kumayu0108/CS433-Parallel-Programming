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
#define GATHER 96

void Initialize (int n, int P, float *X)
{
   for (int j=0; j<n+2; j++) X[j] = (j % 100)/100.0;
   for (int j=0; j<n+2; j++) X[(n/P+1)*(n+2)+j] = (j % 100)/100.0;
   for (int i=1; i<n/P+1; i++) for (int j=0; j<n+2; j++) X[i*(n+2)+j] = ((float)(random() % 100)/100.0);
}

int main(int argc, char ** argv)
{
   int pid, P, done, n, iters = 0;
   float local_diff, temp, *A, total_diff, *result;
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
   done = 0;

   A = (float *) malloc ((n/P+2) * (n+2) * sizeof(float));
   Initialize(n, P, A);

   if (!pid) {
   	result = (float *) malloc ((n+2) * (n+2) * sizeof(float));
   	Initialize(n, 1, result);
   }

   if (!pid) {
      sprintf(buffer, "mpi_gsblockrow_allreduce_gather_outfile_%d_%d.txt", n, P);
      fp = fopen(buffer, "w");
      gettimeofday(&tv0, &tz0);
   }

   // Uncomment the following line to see the placement of processes
   //printf("%s: Hi, I am %d about to start computation.\n", hostname, pid);

   while (!done) {
      local_diff = 0.0;
      /* MPI_CHAR means raw byte format */
      if (pid) {  /* send my first row up */
         MPI_Send(&A[n+3], n*sizeof(float), MPI_CHAR, pid-1, ROW, MPI_COMM_WORLD);
      }
      if (pid != P-1) {  /* recv last row */
         MPI_Recv(&A[(n/P+1)*(n+2)+1], n*sizeof(float), MPI_CHAR, pid+1, ROW, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      }
      if (pid != P-1) {  /* send last row down */
         MPI_Send(&A[(n/P)*(n+2)+1], n*sizeof(float), MPI_CHAR, pid+1, ROW, MPI_COMM_WORLD);
      }
      if (pid) {  /* recv first row from above */
         MPI_Recv(&A[1], n*sizeof(float), MPI_CHAR, pid-1, ROW, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      }
      for (int i=1; i <= n/P; i++) for (int j=1; j <= n; j++) {
         temp = A[i*(n+2)+j];
         A[i*(n+2)+j] = 0.2 * (A[i*(n+2)+j] + A[i*(n+2)+j-1] + A[(i-1)*(n+2)+j] + A[i*(n+2)+j+1] + A[(i+1)*(n+2)+j]);
         local_diff += fabs(A[i*(n+2)+j] - temp);
      }
      MPI_Allreduce(&local_diff, &total_diff, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
      iters++;
      if ((total_diff/(n*n) < TOL) ||
          (iters == ITER_LIMIT)) done = 1;
      if (!pid) fprintf(fp, "[%d] diff = %.10f\n", iters, total_diff/(n*n));
   }
   //MPI_Gather (&A[n+2], (n/P)*(n+2), MPI_FLOAT, &result[n+2], (n/P)*(n+2), MPI_FLOAT, 0, MPI_COMM_WORLD);
   if (pid) MPI_Send(&A[n+2], (n/P)*(n+2), MPI_FLOAT, 0, DONE, MPI_COMM_WORLD);
   else {
      for (int i=1; i<P; i++) {
         MPI_Recv(&result[n+2+i*(n/P)*(n+2)], (n/P)*(n+2), MPI_FLOAT, i, DONE, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      }
      for (int i=0; i<(n/P)*(n+2); i++) result[n+2+i] = A[n+2+i];
      gettimeofday(&tv1, &tz1);
      fclose(fp);
      total_diff = 0;
      for (int i=1; i <=n; i++) for (int j=1; j <= n; j++) {
         temp = 0.2 * (result[i*(n+2)+j] + result[i*(n+2)+j-1] + result[(i-1)*(n+2)+j] + result[i*(n+2)+j+1] + result[(i+1)*(n+2)+j]);
         total_diff += fabs(result[i*(n+2)+j] - temp);
      }

      printf("Error: %.10f, Time: %ld microseconds.\n", total_diff/(n*n), (tv1.tv_sec-tv0.tv_sec)*1000000+(tv1.tv_usec-tv0.tv_usec));
   }
   MPI_Finalize();
   return 0;
}

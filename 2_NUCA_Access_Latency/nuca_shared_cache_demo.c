/* Author: Mainak Chaudhuri */

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <unistd.h>
#include <signal.h>
#include <sched.h>
#include <string.h>
#include <errno.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <assert.h>

#define IA32_PERFEVTSEL0_MSR_ADDR 0x186
#define IA32_PERFEVTSEL1_MSR_ADDR 0x187
#define IA32_PERFEVTSEL2_MSR_ADDR 0x188

#define IA32_PERF_GLOBAL_CTRL_MSR_ADDR 0x38f

#define FATAL(fmt,args...) do {               \
   PRINT(fmt, ##args);                        \
   exit(1);                                   \
} while (0)
 
#define PRINT(fmt,args...) \
   fprintf(stderr, fmt, ##args)
 
#define openmsr_for_rw(cpu_id) do {                                \
   sprintf(filepath, "/dev/cpu/%d/msr", cpu_id);                   \
   rwmsr_fd[cpu_id] = open(filepath, O_RDWR);                      \
   if (rwmsr_fd[cpu_id] < 0) printf("Failed to open msr file.\n"); \
} while (0)

#define rdmsr(fd,msr_addr,value) do {                 \
   retval = pread(fd, &value, 8, msr_addr);           \
   if (retval == -1) printf("Failed to read msr.\n"); \
} while (0)

#define wrmsr(fd,msr_addr,value) do {                     \
   retval = pwrite(fd, &value, 8, msr_addr);              \
   if (retval == -1) printf("Failed to write to msr.\n"); \
} while (0)

#define rdpmc(counter,low,high)     \
     __asm__ __volatile__("rdpmc"   \
        : "=a" (low), "=d" (high)   \
        : "c" (counter))

#ifdef L2_CACHE_FITTING
#define ARRAY_SIZE ((512*1024)/sizeof(int))
#else
#define ARRAY_SIZE ((2*1024*1024)/sizeof(int))
#endif
#define ITERS 100
 
int cpu, nr_cpus;

int *rwmsr_fd;
char filepath[64];

size_t retval;
 
void handle (int sig)
{
   PRINT("Please run check_rdpmc and check_rdmsr to find out what is wrong.\n");
   FATAL("cpu %d: caught %d\n", cpu, sig);
}
 
int main (int argc, char *argv[])
{
   nr_cpus = sysconf(_SC_NPROCESSORS_ONLN);

   PRINT("CPU count: %d\n", nr_cpus);

   rwmsr_fd = (int*)malloc(nr_cpus*sizeof(int));
   assert(rwmsr_fd != NULL);

   cpu = sched_getcpu();
 
   signal(SIGSEGV, &handle);

   openmsr_for_rw(cpu);

   unsigned long long value;

   rdmsr(rwmsr_fd[cpu],IA32_PERF_GLOBAL_CTRL_MSR_ADDR,value);
   PRINT("cpu %d: value of IA32_PERF_GLOBAL_CTRL_MSR_ADDR [MSR %#x] = %#llx\n", cpu, IA32_PERF_GLOBAL_CTRL_MSR_ADDR, value);
   if ((value != 0x00000007000000ffULL) && (value != 0x000000070000000fULL)) {
      PRINT("Set IA32_PERF_GLOBAL_CTRL_MSR value to 0x000000070000000f if hyperthreading is enabled; else to 0x00000007000000ff using wrmsr.\n");
      assert((value == 0x00000007000000ffULL) || (value == 0x000000070000000fULL));
   }

   // The next four rdmsr's are unnecessary and can be removed. They are there just to examine the current values of the MSRs.

   rdmsr(rwmsr_fd[cpu],IA32_PERFEVTSEL0_MSR_ADDR,value);
   PRINT("cpu %d: value of IA32_PERFEVTSEL0_MSR_ADDR [MSR %#x] = %#llx\n", cpu, IA32_PERFEVTSEL0_MSR_ADDR, value);

   rdmsr(rwmsr_fd[cpu],IA32_PERFEVTSEL1_MSR_ADDR,value);
   PRINT("cpu %d: value of IA32_PERFEVTSEL1_MSR_ADDR [MSR %#x] = %#llx\n", cpu, IA32_PERFEVTSEL1_MSR_ADDR, value);

   rdmsr(rwmsr_fd[cpu],IA32_PERFEVTSEL2_MSR_ADDR,value);
   PRINT("cpu %d: value of IA32_PERFEVTSEL2_MSR_ADDR [MSR %#x] = %#llx\n", cpu, IA32_PERFEVTSEL2_MSR_ADDR, value);

   PRINT("\n");

   // First event: unhalted core cycles: 0x0043003c
   value = 0x000000000043003cULL;
   wrmsr(rwmsr_fd[cpu],IA32_PERFEVTSEL0_MSR_ADDR,value);
   rdmsr(rwmsr_fd[cpu],IA32_PERFEVTSEL0_MSR_ADDR,value);
   PRINT("cpu %d: new value of IA32_PERFEVTSEL0_MSR_ADDR [MSR %#x] = %#llx\n", cpu, IA32_PERFEVTSEL0_MSR_ADDR, value);

   // Second event: LLC references: 0x00434f2e
   value = 0x0000000000434f2eULL;
   wrmsr(rwmsr_fd[cpu],IA32_PERFEVTSEL1_MSR_ADDR,value);
   rdmsr(rwmsr_fd[cpu],IA32_PERFEVTSEL1_MSR_ADDR,value);
   PRINT("cpu %d: new value of IA32_PERFEVTSEL1_MSR_ADDR [MSR %#x] = %#llx\n", cpu, IA32_PERFEVTSEL1_MSR_ADDR, value);

   // Third event: LLC misses: 0x0043412e
   value = 0x000000000043412eULL;
   wrmsr(rwmsr_fd[cpu],IA32_PERFEVTSEL2_MSR_ADDR,value);
   rdmsr(rwmsr_fd[cpu],IA32_PERFEVTSEL2_MSR_ADDR,value);
   PRINT("cpu %d: new value of IA32_PERFEVTSEL2_MSR_ADDR [MSR %#x] = %#llx\n", cpu, IA32_PERFEVTSEL2_MSR_ADDR, value);

   unsigned long long start_value_pmc0, start_value_pmc1, start_value_pmc2;
   unsigned int start_value_pmc0_low, start_value_pmc0_high;
   unsigned int start_value_pmc1_low, start_value_pmc1_high;
   unsigned int start_value_pmc2_low, start_value_pmc2_high;

   unsigned long long end_value_pmc0, end_value_pmc1, end_value_pmc2;
   unsigned int end_value_pmc0_low, end_value_pmc0_high;
   unsigned int end_value_pmc1_low, end_value_pmc1_high;
   unsigned int end_value_pmc2_low, end_value_pmc2_high;

   register int i, j, k, x=0, prevj = -1;
   register unsigned long long LLCref=0, LLCmiss=0, totalcycles=0;
   int *array = (int*)malloc(ARRAY_SIZE*sizeof(int));
   assert(array != NULL);
   for (k=0; k<1024; k++) for (j=k; j<ARRAY_SIZE; j+=1024) {
      if (prevj != -1) {
         array[prevj] = j;
      }
      prevj = j;
   }
   assert(prevj == ARRAY_SIZE-1);
   array[prevj] = 0;

   for (i=0; i<ITERS; i++) {
      k = 0;
      j = 0;
      while (k != ARRAY_SIZE) {
	 // Insert serializing code
         asm ("movl $0x2, %%eax\n\t"
              "cpuid"
              :
              :
              : "eax","ebx","ecx","edx");

	 // Read initial values from performance counters
         rdpmc(2,start_value_pmc2_low,start_value_pmc2_high);
         rdpmc(1,start_value_pmc1_low,start_value_pmc1_high);
         rdpmc(0,start_value_pmc0_low,start_value_pmc0_high);

         // Insert serializing code
         asm ("movl $0x2, %%eax\n\t"
              "cpuid"
              :
              :
              : "eax","ebx","ecx","edx");

	 // Insert code to be monitored
#ifdef PREFETCH_FRIENDLY
	 prevj = j;
	 x += array[j];
	 j++;
#else
	 prevj = j;
         j = array[j];
         x += j;
#endif
	 // Insert serializing code
         asm ("movl $0x2, %%eax\n\t"
              "cpuid"
              :
              :
              : "eax","ebx","ecx","edx");

         // Read final values from performance counters
         rdpmc(0,end_value_pmc0_low,end_value_pmc0_high);
         rdpmc(1,end_value_pmc1_low,end_value_pmc1_high);
         rdpmc(2,end_value_pmc2_low,end_value_pmc2_high);
	 // Insert serializing code
         asm ("movl $0x2, %%eax\n\t"
              "cpuid"
              :
              :
              : "eax","ebx","ecx","edx");

         start_value_pmc0 = ((unsigned long long)start_value_pmc0_high << 32) | start_value_pmc0_low;
         start_value_pmc1 = ((unsigned long long)start_value_pmc1_high << 32) | start_value_pmc1_low;
         start_value_pmc2 = ((unsigned long long)start_value_pmc2_high << 32) | start_value_pmc2_low;

         end_value_pmc0 = ((unsigned long long)end_value_pmc0_high << 32) | end_value_pmc0_low;
         end_value_pmc1 = ((unsigned long long)end_value_pmc1_high << 32) | end_value_pmc1_low;
         end_value_pmc2 = ((unsigned long long)end_value_pmc2_high << 32) | end_value_pmc2_low;

 	 printf("%d %llu\n", prevj, (end_value_pmc0 - start_value_pmc0));
 	 totalcycles += (end_value_pmc0 - start_value_pmc0);
	 LLCref += (end_value_pmc1 - start_value_pmc1);
	 LLCmiss += (end_value_pmc2 - start_value_pmc2);

         k++;
      }
   }

   PRINT("cpu %d: Unhalted core cycles: %llu (total), %lf (average per iteration), %lf (average per access)\n", cpu, totalcycles, (double)totalcycles/ITERS, (double)totalcycles/(ITERS*ARRAY_SIZE));
   PRINT("cpu %d: LLC references: %llu (total), %lf (average per iteration), %lf (average per access)\n", cpu, LLCref, (double)LLCref/ITERS, (double)LLCref/(ITERS*ARRAY_SIZE));
   PRINT("cpu %d: LLC misses: %llu (total), %lf (average per iteration), %lf (average per access)\n", cpu, LLCmiss, (double)LLCmiss/ITERS, (double)LLCmiss/(ITERS*ARRAY_SIZE));

   PRINT("%d\n", x);
   return 0;
}
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

int main() {
  printf("Parallel region execution.\n");
#pragma omp parallel
  {
    int tid = omp_get_thread_num();
    int num_threads = omp_get_num_threads();
    printf("Hello from thread %d of %d\n", tid, num_threads);
  }

  printf("Parallel region finished.\n");

  return 0;
}

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main(int argc, char *argv[]) {
  if (argc < 2) {
    printf("Usage: %s N\n", argv[0]);
    return 1;
  }
  int N = atoi(argv[1]);
  int *A = (int *)malloc(N * sizeof(int));
  int *B = (int *)malloc(N * sizeof(int));
  srand(time(NULL));

  printf("Initializing arrays A and B with size %d\n", N);

#pragma omp parallel
  {
#pragma omp single
    {
      for (int i = 0; i < N; i++) {
#pragma omp task depend(inout : A[i])
        {
          A[i] = i;
          printf("Task %d - 0: Initializing A[%d] = %d\n", i, i, A[i]);
        }

        if (i == 0) {
#pragma omp task depend(in : A[i]) depend(inout : B[i])
          {
            printf("Task %d - 1 -> Input: A[%d] = %d\n", i, i, A[i]);
            B[i] = A[i] + 5;
            printf("Task %d - 1: Computing B[%d] = %d\n", i, i, B[i]);
          }
        } else {
#pragma omp task depend(in : A[i]) depend(in : B[i - 1]) depend(inout : B[i])
          {
            printf("Task %d - 2 -> Input: A[%d] = %d, B[%d] = %d\n", i, i, A[i],
                   i - 1, B[i - 1]);
            B[i] = A[i] + B[i - 1] + 5;
            printf("Task %d - 2: Computing B[%d] = %d\n", i, i, B[i]);
          }
        }
      }
    }
  }

  printf("Final arrays:\n");
  for (int i = 0; i < N; i++)
    printf("A[%d] = %d, B[%d] = %d\n", i, A[i], i, B[i]);
  return 0;
}

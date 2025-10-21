#include <omp.h>
#include <stdio.h>

int fib(int n) {
  int i, j;
  if (n < 2)
    return n;
  else {
#pragma omp task shared(i)
    i = fib(n - 1);

#pragma omp task shared(j)
    j = fib(n - 2);

#pragma omp taskwait
    return i + j;
  }
}

int main(int argc, char *argv[]) {
  int n = 5;
  if (argc > 1) {
    n = atoi(argv[1]);
  }

  int result = 0;
  double start = omp_get_wtime();
#pragma omp parallel firstprivate(n)
#pragma omp single
  result = fib(n);
  double end = omp_get_wtime();
  printf("Finished in %f seconds\n", end - start);

  // Sequential for correctness
  int seq = 0, a = 0, b = 1;
  if (n == 0)
    seq = 0;
  else if (n == 1)
    seq = 1;
  else {
    for (int i = 2; i <= n; ++i) {
      seq = a + b;
      a = b;
      b = seq;
    }
  }
  
  printf("RESULTS: ");
  printf(" - Sequential result: %d\n", seq);
  printf(" - Parallel result: %d\n", result);

  /// Verify results
  if (result == seq)
    printf("Result: CORRECT\n");
  else
    printf("Result: INCORRECT\n");
}

#include <omp.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {
  if (argc < 2) {
    printf("Usage: %s <size>\n", argv[0]);
    return 1;
  }
  const int N = atoi(argv[1]);
  if (N <= 0) {
    printf("Error: N must be a positive integer.\n");
    return 1;
  }

  int a[N], b[N], c[N];

  // Initialize arrays
  for (int i = 0; i < N; i++) {
    a[i] = i;
    b[i] = i * 2;
  }

  // Start timer
  double t_start = omp_get_wtime();

// Parallel loop for array addition
#pragma omp parallel for
  for (int i = 0; i < N; i++)
    c[i] = a[i] + b[i];

  // Stop timer
  double t_end = omp_get_wtime();
  printf("Finished in %f seconds\n", t_end - t_start);

  // Verify results
  bool success = true;
  for (int i = 0; i < N; i++) {
    if (c[i] != a[i] + b[i]) {
      success = false;
      break;
    }
  }
  // Output correctness in a script-parseable format
  if (success)
    printf("Result: CORRECT\n");
  else
    printf("Result: INCORRECT\n");

  fflush(stdout);
  return 0;
}

#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {
  if (argc < 2) {
    printf("Usage: %s <size>\n", argv[0]);
    return 1;
  }
  int N = atoi(argv[1]);
  int *a = (int *)malloc(N * sizeof(int));
  int *b = (int *)malloc(N * sizeof(int));
  int *c = (int *)malloc(N * sizeof(int));

  // Initialize arrays
  for (int i = 0; i < N; i++) {
    a[i] = i;
    b[i] = i * 2;
  }
  /// Print initial values
  printf("Initial values:\n");
  for (int i = 0; i < N; i++)
    printf("a[%d] = %d, b[%d] = %d\n", i, a[i], i, b[i]);

  /// Parallel region with worksharing loop - vector addition
  printf("Parallel region:\n");
#pragma omp parallel for schedule(static, 4)
  for (int i = 0; i < N; i++)
    c[i] = a[i] + b[i];

  printf("Parallel region completed\n");

  /// Verify results
  int sum = 0;
#pragma omp parallel
  {
#pragma omp for reduction(+ : sum)
    for (int i = 0; i < N; i++)
      sum += c[i];
  }

  printf("Sum of results: %d\n", sum);
  printf("Expected sum: %d\n", N * (N - 1) / 2 + N * (N - 1));

  free(a);
  free(b);
  free(c);

  return 0;
}

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main(int argc, char *argv[]) {
  if (argc < 2) {
    printf("Usage: %s N\n", argv[0]);
    exit(EXIT_FAILURE);
  }
  const int N = atoi(argv[1]);
  double **A = (double **)malloc(N * sizeof(double *));
  double **B = (double **)malloc(N * sizeof(double *));
  for (int i = 0; i < N; i++) {
    A[i] = (double *)malloc(N * sizeof(double));
    B[i] = (double *)malloc(N * sizeof(double));
  }

  srand(time(NULL));
  const int random = rand() % 11;
  printf("\nRandom number: %d\n", random);
#pragma omp parallel
  {
#pragma omp single
    {
      /// Compute the A matrix
      for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
/// Each task computes one A[i][j].
#pragma omp task firstprivate(i, j) depend(out: A[i][j])
          {
            A[i][j] = i * 1.0 + random;
            printf("Task 0: Initializing A[%d][%d] = %.2f\n", i, j, A[i][j]);
          }
        }
      }

      /// Compute the B matrix using A.
      /// For row zero, B[0][j] only depends on A[0][j]
      for (int j = 0; j < N; j++) {
#pragma omp task firstprivate(j) depend(in: A[0][j]) depend(out: B[0][j])
        {
          B[0][j] = A[0][j];
          printf("Task 1: Computing B[0][%d] = %.2f\n", j, B[0][j]);
        }
      }

      /// For rows 1..N-1, B[i][j] depends on A[i][j] and A[i-1][j].
      for (int i = 1; i < N; i++) {
        for (int j = 0; j < N; j++) {
#pragma omp task firstprivate(i, j) depend(in: A[i][j], A[i-1][j]) depend(out: B[i][j])
          {
            B[i][j] = A[i][j] + A[i - 1][j];
            printf("Task 2: Computing B[%d][%d] = %.2f\n", i, j, B[i][j]);
          }
        }
      }
    }
  }

  /// Print the computed matrices
  printf("Matrix A:\n");
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      printf("%6.2f ", A[i][j]);
    }
    printf("\n");
  }

  printf("\nMatrix B:\n");
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      printf("%6.2f ", B[i][j]);
    }
    printf("\n");
  }

  /// Verification of the result.
  int correct = 1;
  for (int j = 0; j < N; j++) {
    if (B[0][j] != A[0][j]) {
      correct = 0;
    }
  }
  for (int i = 1; i < N; i++) {
    for (int j = 0; j < N; j++) {
      if (B[i][j] != A[i][j] + A[i - 1][j]) {
        correct = 0;
      }
    }
  }
  if (correct) {
    printf("\nVerification: PASSED\n");
  } else {
    printf("\nVerification: FAILED\n");
  }

  return 0;
}

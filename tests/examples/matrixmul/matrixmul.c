#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#define TOLERANCE 1e-6

int main(int argc, char *argv[]) {
  /// Check for the correct number of arguments
  if (argc != 3) {
    /// E.g. ./matrixmul 8 2
    printf("Usage: %s <matrix_size> <block_size>\n", argv[0]);
    return EXIT_FAILURE;
  }

  const int N = atoi(argv[1]);
  const int BS = atoi(argv[2]);
  if (N <= 0 || BS <= 0) {
    printf("Matrix size and block size must be positive integers.\n");
    return EXIT_FAILURE;
  }

  /// Allocate matrices
  float **A = (float **)malloc(N * sizeof(float *));
  float **B = (float **)malloc(N * sizeof(float *));
  float **C_parallel = (float **)malloc(N * sizeof(float *));
  float **C_sequential = (float **)malloc(N * sizeof(float *));
  for (int i = 0; i < N; i++) {
    A[i] = (float *)malloc(N * sizeof(float));
    B[i] = (float *)malloc(N * sizeof(float));
    C_parallel[i] = (float *)malloc(N * sizeof(float));
    C_sequential[i] = (float *)malloc(N * sizeof(float));
  }

  /// Initialize matrices
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      A[i][j] = (float)(i + j);
      B[i][j] = (float)(i - j);
      C_parallel[i][j] = 0.0f;
      C_sequential[i][j] = 0.0f;
    }
  }

  /// Print A and B
  printf("Matrix A:\n");
  for (int i = 0; i < N; i++) {
    printf("  ");
    for (int j = 0; j < N; j++)
      printf("%6.2f ", A[i][j]);
    printf("\n");
  }

  printf("\nMatrix B:\n");
  for (int i = 0; i < N; i++) {
    printf("  ");
    for (int j = 0; j < N; j++)
      printf("%6.2f ", B[i][j]);
    printf("\n");
  }

  /// Parallel computation
  double t_start = omp_get_wtime();
#pragma omp parallel
#pragma omp single
  {
    for (int i = 0; i < N; i += BS) {
      for (int j = 0; j < N; j += BS) {
        for (int k = 0; k < N; k += BS) {
#pragma omp task depend(in : A[i][k], B[k][j]) depend(inout : C_parallel[i][j])
          {
            printf(
                "Thread %d is working on block (%d, %d, %d) to (%d, %d, %d)\n",
                omp_get_thread_num(), i, j, k, i + BS, j + BS, k + BS);
            for (int ii = i; ii < i + BS; ii++)
              for (int jj = j; jj < j + BS; jj++)
                for (int kk = k; kk < k + BS; kk++)
                  C_parallel[ii][jj] += A[ii][kk] * B[kk][jj];
          }
        }
      }
    }
  }
  double t_end = omp_get_wtime();
  printf("Finished in %f seconds\n", t_end - t_start);

  /// Sequential computation
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      float sum = 0.0f;
      for (int k = 0; k < N; k++)
        sum += A[i][k] * B[k][j];
      C_sequential[i][j] = sum;
    }
  }

  /// Print C parallel
  printf("\nMatrix C (Parallel Result):\n");
  for (int i = 0; i < N; i++) {
    printf("  ");
    for (int j = 0; j < N; j++)
      printf("%8.2f ", C_parallel[i][j]);
    printf("\n");
  }

  /// Verification
  int verified = 1;
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      if (fabs(C_parallel[i][j] - C_sequential[i][j]) > TOLERANCE) {
        printf("Mismatch at (%d,%d): Parallel=%.2f, Sequential=%.2f\n", i, j,
               C_parallel[i][j], C_sequential[i][j]);
        verified = 0;
      }
    }
  }

  /// Clean up
  for (int i = 0; i < N; i++) {
    free(A[i]);
    free(B[i]);
    free(C_parallel[i]);
    free(C_sequential[i]);
  }
  free(A);
  free(B);
  free(C_parallel);
  free(C_sequential);

  if (verified)
    printf("Result: CORRECT\n");
  else
    printf("Result: INCORRECT\n");

  fflush(stdout);

  return 0;
}
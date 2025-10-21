#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define BLOCK_SIZE 16
#define TOLERANCE 1e-6
#define N 128

int main() {
  srand(time(NULL));

  double A[N * N];
  double L_parallel[N * N];
  double L_sequential[N * N];
  double L_orig[N * N];

  /// Generate a positive-definite matrix A = L_orig * L_orig^T
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      /// Avoid zeros on off-diagonal
      if (j <= i)
        L_orig[i * N + j] = (double)rand() / RAND_MAX + 0.01;
      /// L_orig is lower triangular
      else
        L_orig[i * N + j] = 0.0;
    }
    /// Strengthen diagonal for positive definiteness
    L_orig[i * N + i] += 1.0;
  }

  for (int i = 0; i < N; i++) {
    /// A is symmetric, compute its lower triangle
    for (int j = 0; j <= i; j++) {
      A[i * N + j] = 0.0;
      for (int k_sum = 0; k_sum <= j; k_sum++)
        A[i * N + j] += L_orig[i * N + k_sum] * L_orig[j * N + k_sum];
    }
  }

  /// Fill upper triangle of A by symmetry
  for (int i = 0; i < N; i++) {
    for (int j = i + 1; j < N; j++)
      A[i * N + j] = A[j * N + i];
  }

  /// Initialize matrices for Cholesky decomposition
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      L_parallel[i * N + j] = A[i * N + j];
      L_sequential[i * N + j] = A[i * N + j];
    }
  }

  /// Parallel Cholesky Decomposition
  double start_time = omp_get_wtime();
#pragma omp parallel
#pragma omp single
  {
    for (int k_block = 0; k_block < N; k_block += BLOCK_SIZE) {
      int kb = (k_block + BLOCK_SIZE > N) ? N - k_block : BLOCK_SIZE;

      /// Task 1: Factorize diagonal block (POTRF-like operation)
#pragma omp task depend(inout : L_parallel[k_block * N + k_block])
      {
        for (int i = k_block; i < k_block + kb; i++) {
          for (int j = k_block; j < i; j++) {
            L_parallel[i * N + i] -=
                L_parallel[i * N + j] * L_parallel[i * N + j];
          }
          /// Avoid sqrt of negative due to floating point inaccuracies
          if (L_parallel[i * N + i] < 0.0)
            L_parallel[i * N + i] = 0.0;
          L_parallel[i * N + i] = sqrt(L_parallel[i * N + i]);

          for (int j = i + 1; j < k_block + kb; j++) {
            for (int m = k_block; m < i; m++) {
              L_parallel[j * N + i] -=
                  L_parallel[j * N + m] * L_parallel[i * N + m];
            }
            /// Handle division by zero if diagonal element L_ii is zero
            if (L_parallel[i * N + i] != 0.0)
              L_parallel[j * N + i] /= L_parallel[i * N + i];
            /// Set to 0 to prevent NaN; indicates numerical issues
            else
              L_parallel[j * N + i] = 0.0;
          }
        }
      }

      /// Task 2: Update panel of blocks below diagonal (TRSM-like operation)
      for (int j_block = k_block + kb; j_block < N; j_block += BLOCK_SIZE) {
        int jb = (j_block + BLOCK_SIZE > N) ? N - j_block : BLOCK_SIZE;
#pragma omp task depend(in : L_parallel[k_block * N + k_block])                \
    depend(inout : L_parallel[j_block * N + k_block])
        {
          /// Column index in L_kk and panel L_jk
          for (int col_k = k_block; col_k < k_block + kb; col_k++) {
            /// Row index in panel L_jk
            for (int row_j = j_block; row_j < j_block + jb; row_j++) {
              for (int m = k_block; m < col_k; m++) {
                L_parallel[row_j * N + col_k] -=
                    L_parallel[row_j * N + m] * L_parallel[col_k * N + m];
              }
              /// Avoid division by zero
              if (L_parallel[col_k * N + col_k] != 0.0)
                L_parallel[row_j * N + col_k] /= L_parallel[col_k * N + col_k];
              /// Avoid NaN; indicates numerical issues
              else
                L_parallel[row_j * N + col_k] = 0.0;
            }
          }
        }
      }

      /// Task 3: Update trailing submatrix (SYRK/GEMM-like operation)
      for (int j_block_trail = k_block + kb; j_block_trail < N;
           j_block_trail += BLOCK_SIZE) {
        int jb_trail =
            (j_block_trail + BLOCK_SIZE > N) ? N - j_block_trail : BLOCK_SIZE;
        for (int i_block_trail = j_block_trail; i_block_trail < N;
             i_block_trail += BLOCK_SIZE) {
          int ib_trail =
              (i_block_trail + BLOCK_SIZE > N) ? N - i_block_trail : BLOCK_SIZE;
#pragma omp task depend(in : L_parallel[i_block_trail * N + k_block],          \
                            L_parallel[j_block_trail * N + k_block])           \
    depend(inout : L_parallel[i_block_trail * N + j_block_trail])
          {
            for (int row_target = i_block_trail;
                 row_target < i_block_trail + ib_trail; row_target++) {
              for (int col_target = j_block_trail;
                   col_target < j_block_trail + jb_trail; col_target++) {
                /// Update only lower triangular part of the target block
                if (row_target >= col_target) {
                  double sum_update = 0.0;
                  for (int m_sum = k_block; m_sum < k_block + kb; m_sum++) {
                    sum_update += L_parallel[row_target * N + m_sum] *
                                  L_parallel[col_target * N + m_sum];
                  }
                  L_parallel[row_target * N + col_target] -= sum_update;
                }
              }
            }
          }
        }
      }
    } /// End k_block loop
  } /// End OpenMP single / parallel region
  double parallel_time = omp_get_wtime() - start_time;
  printf("Finished in %f seconds\n", parallel_time);

  /// Sequential Cholesky Decomposition
  start_time = omp_get_wtime();
  for (int k = 0; k < N; k++) {
    /// Avoid sqrt of negative
    if (L_sequential[k * N + k] < 0.0)
      L_sequential[k * N + k] = 0.0;
    L_sequential[k * N + k] = sqrt(L_sequential[k * N + k]);

    if (L_sequential[k * N + k] != 0.0) {
      for (int i = k + 1; i < N; i++)
        L_sequential[i * N + k] /= L_sequential[k * N + k];
    } else {
      /// L_kk is zero, avoid division by zero for elements L_ik
      for (int i = k + 1; i < N; i++)
        L_sequential[i * N + k] = 0.0;
    }

    for (int j = k + 1; j < N; j++) {
      for (int i = j; i < N; i++) {
        L_sequential[i * N + j] -=
            L_sequential[i * N + k] * L_sequential[j * N + k];
      }
    }
  }
  double sequential_time = omp_get_wtime() - start_time;

  /// Verify results: Check if A = L * L^T
  int parallel_correct = 1;
  int sequential_correct = 1;
  double max_par_err = 0.0, max_seq_err = 0.0;

  for (int i = 0; i < N; i++) {
    /// Check lower triangle
    for (int j = 0; j <= i; j++) {
      double sum = 0.0;
      for (int k_sum = 0; k_sum <= j; k_sum++)
        sum += L_parallel[i * N + k_sum] * L_parallel[j * N + k_sum];
      double err = fabs(A[i * N + j] - sum);
      if (err > max_par_err)
        max_par_err = err;
      if (err > TOLERANCE * fabs(A[i * N + j]) && err > TOLERANCE)
        parallel_correct = 0;
    }
  }

  for (int i = 0; i < N; i++) {
    /// Check lower triangle
    for (int j = 0; j <= i; j++) {
      double sum = 0.0;
      for (int k_sum = 0; k_sum <= j; k_sum++)
        sum += L_sequential[i * N + k_sum] * L_sequential[j * N + k_sum];
      double err = fabs(A[i * N + j] - sum);
      if (err > max_seq_err)
        max_seq_err = err;
      if (err > TOLERANCE * fabs(A[i * N + j]) && err > TOLERANCE)
        sequential_correct = 0;
    }
  }

  printf("Parallel Cholesky %s! (Max error: %.2e)\n",
         parallel_correct ? "verification passed" : "verification FAILED",
         max_par_err);
  printf("Sequential Cholesky %s! (Max error: %.2e)\n",
         sequential_correct ? "verification passed" : "verification FAILED",
         max_seq_err);
  printf("Parallel time: %.4fs\nSequential time: %.4fs\n", parallel_time,
         sequential_time);

  /// Verify results
  if (parallel_correct)
    printf("Result: CORRECT\n");
  else
    printf("Result: INCORRECT\n");

  return 0;
}
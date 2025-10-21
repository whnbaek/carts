#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MATCH_SCORE 2
#define MISMATCH_PENALTY -1
#define GAP_PENALTY -2

int main() {
  const char seq1[] = "AGTACGCATGACCTGATCGTACGATCGATGCA";
  const char seq2[] = "TATGCGCTAGCTAGGCTATGCGATCGTAGCGA";
  int len1 = strlen(seq1);
  int len2 = strlen(seq2);

  int score_parallel[len1 + 1][len2 + 1];
  int score_sequential[len1 + 1][len2 + 1];

  /// Initialize matrices to 0
  for (int i = 0; i <= len1; i++) {
    for (int j = 0; j <= len2; j++) {
      score_parallel[i][j] = 0;
      score_sequential[i][j] = 0;
    }
  }

  double t_start = omp_get_wtime();
/// Parallel Smith-Waterman inline
#pragma omp parallel
#pragma omp single
  {
    /// Print sequences
    printf("seq1: %s\n", seq1);
    printf("seq2: %s\n", seq2);
    for (int i = 1; i <= len1; i++) {
      for (int j = 1; j <= len2; j++) {
#pragma omp task depend(                                                       \
        in : score_parallel[i - 1][j], score_parallel[i][j - 1],               \
            score_parallel[i - 1][j - 1]) depend(out : score_parallel[i][j])
        {
          int match =
              score_parallel[i - 1][j - 1] +
              ((seq1[i - 1] == seq2[j - 1]) ? MATCH_SCORE : MISMATCH_PENALTY);
          int del = score_parallel[i - 1][j] + GAP_PENALTY;
          int ins = score_parallel[i][j - 1] + GAP_PENALTY;
          score_parallel[i][j] = (match > del) ? ((match > ins) ? match : ins)
                                               : ((del > ins) ? del : ins);
          if (score_parallel[i][j] < 0)
            score_parallel[i][j] = 0;
        }
      }
    }
  }
  double t_end = omp_get_wtime();
  printf("Finished in %f seconds\n", t_end - t_start);
  printf("Process has finished\n");

  /// Sequential Smith-Waterman inline
  for (int i = 1; i <= len1; i++) {
    for (int j = 1; j <= len2; j++) {
      int match =
          score_sequential[i - 1][j - 1] +
          ((seq1[i - 1] == seq2[j - 1]) ? MATCH_SCORE : MISMATCH_PENALTY);
      int del = score_sequential[i - 1][j] + GAP_PENALTY;
      int ins = score_sequential[i][j - 1] + GAP_PENALTY;
      score_sequential[i][j] = (match > del) ? ((match > ins) ? match : ins)
                                             : ((del > ins) ? del : ins);
      if (score_sequential[i][j] < 0)
        score_sequential[i][j] = 0;
    }
  }

  /// Verify the results inline
  int ok = 1;
  for (int i = 0; i <= len1; i++) {
    for (int j = 0; j <= len2; j++) {
      if (score_parallel[i][j] != score_sequential[i][j]) {
        printf("Mismatch at (%d, %d): Parallel=%d, Sequential=%d\n", i, j,
               score_parallel[i][j], score_sequential[i][j]);
        ok = 0;
      }
    }
  }
  if (ok)
    printf("Result: CORRECT\n");
  else
    printf("Result: INCORRECT\n");

  return 0;
}

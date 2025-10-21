#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main() {
  printf("-----------------\nMain function\n-----------------\n");
  int a = 0;
  int b = 0;
  srand(time(NULL));

#pragma omp parallel
  {
#pragma omp single
    {
      printf("Main thread: Creating tasks\n");
#pragma omp task
      {
        a = rand() % 100;
        printf("Task 1: Initializing a with value %d\n", a);
      }

#pragma omp task depend(in: a) depend(inout: b)
      {
        printf("Task 2: Reading a=%d and updating b\n", a);
        b = a + 5;
      }

#pragma omp task depend(in: b)
      {
        printf("Task 3: Final value of b=%d\n", b);
      }
    }
  }

  printf("Parallel region DONE\n");
  printf("A: %d, B: %d\n", a, b);
  printf("-----------------\nMain function DONE\n-----------------\n");
  return 0;
}

#ifndef MATRIX_IO_H
#define MATRIX_IO_H

#include <mpi.h>
#include <cstdio>

// Matrix initialization and IO functions
int readRow(FILE *fp, double *a, int len);
double partialInit(int s, int n, int i, int j);
void initMatrix(double *a, int n, int m, int procCols, int p, int k, int s, double (*f)(int, int, int, int));
int readMatrix(double *a, int n, int m, int p, int k, int procCols, const char *fileName, double *buf, double *block, MPI_Comm com);
void procInitUnitMatrix(int n, int m, int p, int k, int procCols, double *a);
void printMatrix(double *a, int n, int m, int p, int k, int procCols, double *block, int r, MPI_Comm com);

#endif // MATRIX_IO_H 
#ifndef MATRIX_OPERATIONS_H
#define MATRIX_OPERATIONS_H

#include <mpi.h>

// Block operations
void getBlock(int n, int m, int indv, int indh, int l, int h, double *a, double *block);
void putBlock(int n, int m, int indv, int indh, int l, int h, double *a, double *block);

// Matrix utility functions
void initSimpleUnitMatrix(int n, double *a);
double simpleFindNorm(int n, double *a);
double procFindNorm(int n, int procCol, double *a, MPI_Comm com);
double findCorEl(int n, int m, int procCols, int k, int p, double *a, MPI_Comm com, int s);

// Matrix multiplication
void multMatr(double *a, double *b, double *c, int rowNum1, int colNum1, int colNum2, int m);
void multSub(double *a, double *b, double *c, int rowNum1, int colNum1, int colNum2);
void procMultMatr(double *a, double *b, double *blockRow, double *blockRow1, double *block1, double *mlBlock, double *llBlock, int n, int procCols, int m, int p, int k, MPI_Comm com);

// Matrix operations
void subUnitMatr(int n, int m, int p, int k, int procCols, double *a);
double findRes(double *a, double *b, double *blockRow, double *blockRow1, double *block1, double *mlBlock, double *llBlock, int n, int procCols, int m, int p, int k, MPI_Comm com);

#endif // MATRIX_OPERATIONS_H 
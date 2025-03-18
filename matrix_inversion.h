#ifndef MATRIX_INVERSION_H
#define MATRIX_INVERSION_H

#include <mpi.h>

// Initialization for inversion
void initInvertedStatus(int p, int k, int locBlocks, int totalBlocks, int *a);

// Matrix inversion
int simpleInvert(int n, double *a, double *b, double corEl, int *indicesTable);
int blockInvert(int n, int m, int procCols, int p, int k, MPI_Comm com,
                double corEl, double *a, double *b, int *indicesTable,
                double *minInvertedNormList, double *minInvNormIndexTriple,
                int *invertedStatus, double *blockCol, double *block1,
                double *block2, double *block3, double *mlBlock,
                double *lmBlock, double *lmBlock1, double *llBlock,
                double *llBlock1, double *llBlock2);

#endif // MATRIX_INVERSION_H 
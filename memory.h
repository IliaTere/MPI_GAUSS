#ifndef MEMORY_H
#define MEMORY_H

// Memory management function
void freeAllMemory(double *matrix, double *invertedMatrix, double *blockStringBuf, double *blockRow,
                   double *block1, double *block2, double *block3, double *mlBlock, double *lmBlock,
                   double *lmBlock1, double *llBlock, double *llBlock1, double *llBlock2, int *indicesTable,
                   double *minInvertedNormList, double *minInvNormIndexTriple, int *invertedStatus);

#endif // MEMORY_H 
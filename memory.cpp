#include "memory.h"

void freeAllMemory(double *matrix, double *invertedMatrix, double *blockStringBuf, double *blockRow,
                   double *block1, double *block2, double *block3, double *mlBlock, double *lmBlock,
                   double *lmBlock1, double *llBlock, double *llBlock1, double *llBlock2, int *indicesTable,
                   double *minInvertedNormList, double *minInvNormIndexTriple, int *invertedStatus)
{
    if (matrix != nullptr)
        delete [] matrix;

    if (invertedMatrix != nullptr)
        delete [] invertedMatrix;

    if (invertedStatus != nullptr)
        delete [] invertedStatus;

    if (blockRow != nullptr)
        delete [] blockRow;

    if (mlBlock != nullptr)
        delete [] mlBlock;

    if (lmBlock != nullptr)
        delete [] lmBlock;

    if (lmBlock1 != nullptr)
        delete [] lmBlock1;

    if (llBlock != nullptr)
        delete [] llBlock;

    if (llBlock1 != nullptr)
        delete [] llBlock1;

    if (llBlock2 != nullptr)
        delete [] llBlock2;

    if (blockStringBuf != nullptr)
        delete [] blockStringBuf;

    if (block1 != nullptr)
        delete [] block1;

    if (block2 != nullptr)
        delete [] block2;

    if (block3 != nullptr)
        delete [] block3;

    if (indicesTable != nullptr)
        delete [] indicesTable;

    if (minInvertedNormList != nullptr)
        delete [] minInvertedNormList;

    if (minInvNormIndexTriple != nullptr)
        delete [] minInvNormIndexTriple;
} 
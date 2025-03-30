#include "memory.h"

void freeAllMemory(double *matrix, double *invertedMatrix, double *blockStringBuf, double *blockRow,
                   double *block1, double *block2, double *block3, double *mlBlock, double *lmBlock,
                   double *lmBlock1, double *llBlock, double *llBlock1, double *llBlock2, int *indicesTable,
                   double *minInvertedNormList, double *minInvNormIndexTriple, int *invertedStatus)
{
    auto safeDelete = [](double* &ptr) {
        if (ptr) {
            delete[] ptr;
            ptr = nullptr;
        }
    };
    
    auto safeDeleteInt = [](int* &ptr) {
        if (ptr) {
            delete[] ptr;
            ptr = nullptr;
        }
    };
    
    safeDelete(matrix);
    safeDelete(invertedMatrix);
    safeDelete(blockStringBuf);
    
    safeDelete(blockRow);
    safeDelete(block1);
    safeDelete(block2);
    safeDelete(block3);
    
    safeDelete(mlBlock);
    safeDelete(lmBlock);
    safeDelete(lmBlock1);
    
    safeDelete(llBlock);
    safeDelete(llBlock1);
    safeDelete(llBlock2);
    
    safeDeleteInt(indicesTable);
    safeDeleteInt(invertedStatus);
    
    safeDelete(minInvertedNormList);
    safeDelete(minInvNormIndexTriple);
} 
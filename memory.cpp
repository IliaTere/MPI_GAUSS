#include "memory.h"

void freeAllMemory(double *matrix, double *invertedMatrix, double *blockStringBuf, double *blockRow,
                   double *block1, double *block2, double *block3, double *mlBlock, double *lmBlock,
                   double *lmBlock1, double *llBlock, double *llBlock1, double *llBlock2, int *indicesTable,
                   double *minInvertedNormList, double *minInvNormIndexTriple, int *invertedStatus)
{
    // Helper function to safely delete any pointer
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
    
    // Free matrix data structures
    safeDelete(matrix);
    safeDelete(invertedMatrix);
    safeDelete(blockStringBuf);
    
    // Free block buffers
    safeDelete(blockRow);
    safeDelete(block1);
    safeDelete(block2);
    safeDelete(block3);
    
    // Free mixed-size blocks
    safeDelete(mlBlock);
    safeDelete(lmBlock);
    safeDelete(lmBlock1);
    
    // Free lower blocks
    safeDelete(llBlock);
    safeDelete(llBlock1);
    safeDelete(llBlock2);
    
    // Free auxiliary arrays
    safeDeleteInt(indicesTable);
    safeDeleteInt(invertedStatus);
    
    // Free norm data
    safeDelete(minInvertedNormList);
    safeDelete(minInvNormIndexTriple);
} 
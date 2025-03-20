#include "utils.h"
#include <cmath>

int min(int a, int b)
{
    if (a <= b) {
        return a;
    }
    return b;
}

double min(double a, double b)
{
    if (a <= b) {
        return a;
    }
    return b;
}

double max(double a, double b)
{
    if (a >= b) {
        return a;
    }
    return b;
}

int local_to_global(int m, int p, int k, int j_loc)
{
    int localBlock = j_loc / m;
    int globalBlock = localBlock * p + k;
    int localOffset = j_loc % m;
    return globalBlock * m + localOffset;
}

int global_to_local(int m, int p, int j_glob)
{
    int globalBlock = j_glob / m;
    int localBlock = globalBlock / p;
    int localOffset = j_glob % m;
    return localBlock * m + localOffset;
}

int numOfBlockColsInProc(int n, int m, int p, int k)
{
    int totalBlocks = (n + m - 1) / m;
    int baseBlocks = totalBlocks / p;
    int extraBlocks = (totalBlocks % p > k) ? 1 : 0;
    return baseBlocks + extraBlocks;
}

int numOfColsInProc(int n, int m, int p, int k)
{
    int fullBlockSize = m * p;
    int completeBlocks = n / fullBlockSize;
    int baseCols = completeBlocks * m;
    
    int remainingCols = n % fullBlockSize;
    int fullBlocksInRemainder = remainingCols / m;
    int partialBlockSize = remainingCols % m;
    
    if (k < fullBlocksInRemainder) {
        return baseCols + m;
    } else if (k == fullBlocksInRemainder) {
        return baseCols + partialBlockSize;
    }
    
    return baseCols;
}

int getNumOfProc(int m, int p, int j_glob)
{
    int globalBlock = j_glob / m;
    return globalBlock % p;
} 
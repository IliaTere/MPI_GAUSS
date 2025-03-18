#include "utils.h"
#include <cmath>

int min(int a, int b)
{
    return (a < b ? a : b);
}

double min(double a, double b)
{
    return (a < b ? a : b);
}

double max(double a, double b)
{
    return (a > b ? a : b);
}

int local_to_global(int m, int p, int k, int j_loc)
{
    int j_loc_m = j_loc / m;
    int j_glob_m = j_loc_m * p + k;
    return j_glob_m * m + j_loc % m;
}

int global_to_local(int m, int p, int j_glob)
{
    int j_glob_m = j_glob / m;
    int j_loc_m = j_glob_m / p;
    return j_loc_m * m + j_glob % m;
}

int numOfBlockColsInProc(int n, int m, int p, int k)
{
    int totalBlockNum = (n + m - 1) / m;
    return (((totalBlockNum % p) > k) ? (totalBlockNum / p + 1) : (totalBlockNum / p));
}

int numOfColsInProc(int n, int m, int p, int k)
{
    int fullColIter = m * p;
    int colNum = (n / fullColIter) * m;
    int remCols = n % fullColIter;
    int remFullBlocks = remCols / m;
    if (k < remFullBlocks) {
        colNum += m; 
    } else if (k == remFullBlocks) {
        colNum += (remCols % m);
    }

    return colNum;
}

int getNumOfProc(int m, int p, int j_glob)
{
    int j_glob_m = j_glob / m;
    return (j_glob_m % p);
} 
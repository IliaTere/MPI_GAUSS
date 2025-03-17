#include <mpi.h>
#include <stdio.h>
#include <cstdlib>
#include <cstring>
#include <math.h>
#include <ctime>

#define EPS 1e-15

int min(int a, int b);

double min(double a, double b);

double max(double a, double b);

void getBlock(int n, int m, int indv, int indh, int l,
                int h, double *a, double *block); // n - totalColNum in a, m - block size

void putBlock(int n, int m, int indv, int indh, int l,
                int h, double *a, double *block);

int local_to_global(int m, int p, int k, int j_local);

int global_to_local(int m, int p, int j_glob);

int numOfBlockColsInProc(int n, int m, int p, int k);

int numOfColsInProc(int n, int m, int p, int k);

int getNumOfProc(int m, int p, int j_glob); // получить номер процесса по номеру неблочного столбца

int readRow(FILE *fp, double *a, int len);

double partialInit(int s, int n, int i, int j);

void initMatrix(double *a, int n, int m, int procCols, int p,
                int k, int s, double (*f)(int, int, int, int));

int readMatrix(double *a, int n, int m, int p, int k,
                int procCols, const char *fileName, double *buf,
                double *block, MPI_Comm com);

void procInitUnitMatrix(int n, int m, int p, int k, int procCols,
                        double *a);

void printMatrix(double *a, int n, int m, int p, int k,
                    int procCols, double *block, int r, MPI_Comm com);

void initSimpleUnitMatrix(int n, double *a);

double simpleFindNorm(int n, double *a);

double procFindNorm(int n, int procCol, double *a, MPI_Comm com);

double findCorEl(int n, int m, int procCols, int k, int p, double *a,
                    MPI_Comm com, int s);

void initInvertedStatus(int p, int k, int locBlocks, int totalBlocks,
                        int *a);

int simpleInvert(int n, double *a, double *b, double corEl, int *indicesTable);

void multMatr(double *a, double *b, double *c,
                int rowNum1, int colNum1, int colNum2, int m);

void multSub(double *a, double *b, double *c,
                int rowNum1, int colNum1, int colNum2);

void procMultMatr(double *a, double *b, double *blockRow,
                    double *blockRow1, double *block1, double *mlBlock,
                    double *llBlock, int n, int procCols, int m, int p,
                    int k, MPI_Comm com); // result is put to a

int blockInvert(int n, int m, int procCols, int p, int k, MPI_Comm com,
                double corEl, double *a, double *b, int *indicesTable,
                double *minInvertedNormList, double *minInvNormIndexTriple,
                int *invertedStatus, double *blockCol, double *block1,
                double *block2, double *block3, double *mlBlock,
                double *lmBlock, double *lmBlock1, double *llBlock,
                double *llBlock1, double *llBlock2);

void subUnitMatr(int n, int m, int p, int k, int procCols,
                        double *a);

double findRes(double *a, double *b, double *blockRow,
                    double *blockRow1, double *block1, double *mlBlock,
                    double *llBlock, int n, int procCols, int m, int p,
                    int k, MPI_Comm com);

void freeAllMemory(double *matrix, double *invertedMatrix, double *blockStringBuf, double *blockRow,
                   double *block1, double *block2, double *block3, double *mlBlock, double *lmBlock,
                   double *lmBlock1, double *llBlock, double *llBlock1, double *llBlock2, int *indicesTable,
                   double *minInvertedNormList, double *minInvNormIndexTriple, int *invertedStatus/*,
                   int procCols, int remSize */);

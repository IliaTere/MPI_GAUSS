#include "matrix_io.h"
#include "utils.h"
#include "matrix_operations.h"
#include <cstring>
#include <cmath>

int readRow(FILE *fp, double *a, int len)
{
    int i;
    for (i = 0; i < len; ++i) {
        if (fscanf(fp, "%lf", a++) != 1) {
            return 1;
        }
    }

    return 0;
}

double partialInit(int s, int n, int i, int j)
{
    double elem = 0.0;
    switch (s) {
        case 1:
            elem = (n - max(i, j)) * 1.0;
            break;
        case 2:
            elem = max(i, j) + 1.0;
            break;
        case 3:
            elem = fabs(i - j) * 1.0;
            break;
        case 4:
            elem = 1.0 / (i + j + 1);
            break;
        default:
            break;
    }

    return elem;
}

void initMatrix(double *a, int n, int m, int procCols, int p,
                int k, int s, double (*f)(int, int, int, int))
{
    int i_loc, j_loc, i_glob, j_glob;
    //procCols = numOfColsInProc(n, m, p, k);
    for (i_loc = 0; i_loc < n; ++i_loc) {
        i_glob = i_loc;
        for (j_loc = 0; j_loc < procCols; ++j_loc) {
            j_glob = local_to_global(m, p, k, j_loc);
            a[i_loc * procCols + j_loc] = (*f)(s, n, i_glob, j_glob);
        }
    }
}

int readMatrix(double *a, int n, int m, int p, int k,
                int procCols, const char *fileName, double *buf,
                double *block, MPI_Comm com)
{
    int mainProc = 0;
    int blockNum = (n + m - 1) / m;
    int rowsInBlock;
    int colsInBlock;
    int ibv;
    int ibh;
    int ibhLoc;
    FILE *fp = nullptr;
    int err = 0;
    if (k == mainProc) {
        fp = fopen(fileName, "r");
        if (fp == nullptr) {
            err = 1;
        }
    }

    MPI_Bcast(&err, 1, MPI_INT, mainProc, com);
    if (err) {
        return err;
    }

    memset(buf, 0.0, n * m * sizeof(double));
    for (ibv = 0; ibv < blockNum; ++ibv) {
        rowsInBlock = (((ibv + 1) * m <= n) ? m : n - ibv * m);
        if (k == mainProc) {
            err += readRow(fp, buf, n * rowsInBlock);
        }

        MPI_Bcast(buf, n * rowsInBlock, MPI_DOUBLE, mainProc, com);
        for (ibh = k; ibh < blockNum; ibh += p) {
            colsInBlock = (((ibh + 1) * m <= n) ? m : n - ibh * m);
            ibhLoc = ibh / p;
            getBlock(n, m, 0, ibh, colsInBlock, rowsInBlock, buf, block);
            putBlock(procCols, m, ibv, ibhLoc, colsInBlock, rowsInBlock, a, block);
        }
    }

    if (k == mainProc) {
        fclose(fp);
        fp = nullptr;
    }

    MPI_Bcast(&err, 1, MPI_INT, mainProc, com);
    if (err) {
        return err;
    }

    return 0;
}

void procInitUnitMatrix(int n, int m, int p, int k, int procCols, double *a)
{
    int i, j_loc, j_glob;
    for (i = 0; i < n; ++i) {
        for (j_loc = 0; j_loc < procCols; ++j_loc) {
            j_glob = local_to_global(m, p, k, j_loc);
            a[i * procCols + j_loc] = (i == j_glob ? 1.0 : 0.0);
        }
    }
}

void printMatrix(double *a, int n, int m, int p, int k,
                    int procCols, double *block, int r, MPI_Comm com)
{
    int i;
    int mainProc = 0;
    int ownerProc;
    int ibh;
    int ibv;
    int locCurBlock;
    int colsToPrint;
    int printedCols = 0;
    int printSize = min(r, n);
    for (ibv = 0; ibv < printSize; ++ibv) {
        for (ibh = 0; ibh * m < printSize; ++ibh) {
            colsToPrint = ((ibh + 1) * m < printSize ? m : printSize - ibh * m);
            ownerProc = ibh % p;
            locCurBlock = ibh / p;
            if (k == mainProc) {
                if (ownerProc == mainProc) {
                    for (i = 0; i < colsToPrint; ++i) {
                        printf("%10.3e ", a[ibv * procCols + locCurBlock * m + i]);
                    }

                    printedCols += colsToPrint;
                    if (printedCols == printSize) {
                        printf("\n");
                        printedCols = 0;
                    }
                } else {
                    MPI_Status stat;
                    MPI_Recv(block, colsToPrint, MPI_DOUBLE, ownerProc,
                                0, com, &stat);
                    for (i = 0; i < colsToPrint; ++i) {
                        printf("%10.3e ", block[i]);
                    }

                    printedCols += colsToPrint;
                    if (printedCols == printSize) {
                        printf("\n");
                        printedCols = 0;
                    }
                }
            } else {
                if (k == ownerProc) {
                    MPI_Send(a + ibv * procCols + locCurBlock * m,
                                colsToPrint, MPI_DOUBLE, mainProc, 0, com);
                }
            }
        }
    }

    if (k == mainProc) {
        printf("\n\n");
    }
} 
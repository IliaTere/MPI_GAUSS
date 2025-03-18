#include "matrix_operations.h"
#include "utils.h"
#include <cstring>
#include <cmath>

void getBlock(int n, int m, int indv, int indh, int l,
                int h, double *a, double *block)
{
    int i;
    for (i = 0; i < h; ++i) {
        memcpy(block + i * l,
                a + indv * n * m + indh * m + i * n, l * sizeof(double));
    }
}

void putBlock(int n, int m, int indv, int indh, int l,
                int h, double *a, double *block)
{
    int i;
    for (i = 0; i < h; ++i) {
        memcpy(a + indv * n * m + indh * m + i * n,
                block + i * l, l * sizeof(double));
    }
}

void initSimpleUnitMatrix(int n, double *a)
{
    int i, j;
    for (i = 0; i < n; ++i) {
        for (j = 0; j < n; ++j) {
            a[i * n + j] = (i == j ? 1.0 : 0.0);
        }
    }
}

double simpleFindNorm(int n, double *a)
{
    double norm = 0.0;
    double curNorm;
    int i, j;
    for (i = 0; i < n; ++i) {
        curNorm = fabs(a[i]);
        for (j = 1; j < n; ++j) {
            curNorm += fabs(a[i + j * n]);
        }

        if (curNorm > norm) {
            norm = curNorm;
        }
    }

    return norm;
}

double procFindNorm(int n, int procCol, double *a, MPI_Comm com)
{
    double locNorm = 0.0;
    double curNorm;
    double globalNorm = 0.0;
    int i, j;
    for (i = 0; i < procCol; ++i) {
        curNorm = fabs(a[i]);
        //printf("curNorm %lf\n", curNorm);
        for (j = 1; j < n; ++j) {
            curNorm += fabs(a[i + j * procCol]);
        }

        if (curNorm > locNorm) {
            locNorm = curNorm;
        }
    }

    //printf("norm %lf\n", locNorm);
    MPI_Allreduce(&locNorm, &globalNorm, 1, MPI_DOUBLE, MPI_MAX, com);
    return globalNorm;
}

double findCorEl(int n, int m, int procCols, int k, int p, double *a,
                    MPI_Comm com, int s)
{
    double corEl = 0.0;
    int lastColProc = getNumOfProc(m, p, n - 1);
    if (k == lastColProc) {
        corEl = a[n * procCols - 1];
        //corEl *= corEl;
    }

    MPI_Bcast(&corEl, 1, MPI_DOUBLE, lastColProc, com);
    corEl = (s == 4 ? corEl * corEl : procFindNorm(n, procCols, a, com));
    return corEl;
}

void multMatr(double *a, double *b, double *c,
                int rowNum1, int colNum1, int colNum2, int m)
{
    double *pa = nullptr;
    double *pb = nullptr;
    double *pc = nullptr;
    int r, t, q, k;
    double s, s00, s01, s02, s10, s11, s12, s20, s21, s22;
    int rk1 = rowNum1 / m;
    int rl1 = rowNum1 - rk1 * m;
    int ck1 = colNum1 / m;
    int cl1 = colNum1 - ck1 * m;
    int ck2 = colNum2 / m;
    int cl2 = colNum2 - ck2 * m;
    int rowBl1 = (rl1 == 0 ? rk1 : rk1 + 1);
    int colBl1 = (cl1 == 0 ? ck1 : ck1 + 1);
    int colBl2 = (cl2 == 0 ? ck2 : ck2 + 1);
    for (int i = 0; i < rowBl1; ++i) {
        for (int j = 0; j < colBl2; ++j) {
            int vertSize = (i < rk1 ? m : rl1);
            int horSize = (j < ck2 ? m : cl2);
            pc = c + (i * colNum2 + j) * m;
            for (r = 0; r < vertSize; ++r) {
                for (t = 0; t < horSize; ++t) {
                    pc[r * colNum2 + t] = 0;
                }
            }

            for (k = 0; k < colBl1; ++k) {
                int curHorSize = (k < ck1 ? m : cl1);
                pa = a + (i * colNum1 + k) * m;
                pb = b + (k * colNum2 + j) * m;
                int vsRem = vertSize % 3;
                int hsRem = horSize % 3;
                for (r = 0; r < vsRem; ++r) {
                    for (t = 0; t < hsRem; ++t) {
                        s = 0.0;
                        for (q = 0; q < curHorSize; ++q) {
                            s += pa[r * colNum1 + q] * pb[q * colNum2 + t];
                        }

                        pc[r * colNum2 + t] += s;
                    }

                    for (; t < horSize; t += 3) {
                        s00 = 0.0;
                        s01 = 0.0;
                        s02 = 0.0;
                        for (q = 0; q < curHorSize; ++q) {
                            s00 += pa[r * colNum1 + q] * pb[q * colNum2 + t];
                            s01 += pa[r * colNum1 + q] * pb[q * colNum2 + t + 1];
                            s02 += pa[r * colNum1 + q] * pb[q * colNum2 + t + 2];
                        }

                        pc[r * colNum2 + t] += s00;
                        pc[r * colNum2 + t + 1] += s01;
                        pc[r * colNum2 + t + 2] += s02;
                    }
                }

                for (; r < vertSize; r += 3) {
                    for (t = 0; t < hsRem; ++t) {
                        s00 = 0.0;
                        s10 = 0.0;
                        s20 = 0.0;
                        for (q = 0; q < curHorSize; ++q) {
                            s00 += pa[r * colNum1 + q] * pb[q * colNum2 + t];
                            s10 += pa[(r + 1) * colNum1 + q] * pb[q * colNum2 + t];
                            s20 += pa[(r + 2) * colNum1 + q] * pb[q * colNum2 + t];
                        }

                        pc[r * colNum2 + t] += s00;
                        pc[(r + 1) * colNum2 + t] += s10;
                        pc[(r + 2) * colNum2 + t] += s20;
                    }

                    for (; t < horSize; t += 3) {
                        s00 = 0.0;
                        s01 = 0.0;
                        s02 = 0.0;
                        s10 = 0.0;
                        s11 = 0.0;
                        s12 = 0.0;
                        s20 = 0.0;
                        s21 = 0.0;
                        s22 = 0.0;
                        for (q = 0; q < curHorSize; ++q) {
                            s00 += pa[r * colNum1 + q] * pb[q * colNum2 + t];
                            s01 += pa[r * colNum1 + q] * pb[q * colNum2 + t + 1];
                            s02 += pa[r * colNum1 + q] * pb[q * colNum2 + t + 2];
                            s10 += pa[(r + 1) * colNum1 + q] * pb[q * colNum2 + t];
                            s11 += pa[(r + 1) * colNum1 + q] * pb[q * colNum2 + t + 1];
                            s12 += pa[(r + 1) * colNum1 + q] * pb[q * colNum2 + t + 2];
                            s20 += pa[(r + 2) * colNum1 + q] * pb[q * colNum2 + t];
                            s21 += pa[(r + 2) * colNum1 + q] * pb[q * colNum2 + t + 1];
                            s22 += pa[(r + 2) * colNum1 + q] * pb[q * colNum2 + t + 2];
                        }

                        pc[r * colNum2 + t] += s00;
                        pc[r * colNum2 + t + 1] += s01;
                        pc[r * colNum2 + t + 2] += s02;
                        pc[(r + 1) * colNum2 + t] += s10;
                        pc[(r + 1) * colNum2 + t + 1] += s11;
                        pc[(r + 1) * colNum2 + t + 2] += s12;
                        pc[(r + 2) * colNum2 + t] += s20;
                        pc[(r + 2) * colNum2 + t + 1] += s21;
                        pc[(r + 2) * colNum2 + t + 2] += s22;
                    }
                }
            }
        }
    }
}

void multSub(double *a, double *b, double *c,
                int rowNum1, int colNum1, int colNum2)
{
    int i, j, k;
    double s00, s01, s02, s10, s11, s12, s20, s21, s22;
    int verRem = rowNum1 % 3;
    int horRem = colNum2 % 3;
    for (i = 0; i < verRem; ++i) {
        for (j = 0; j < horRem; ++j) {
            s00 = 0.0;
            for (k = 0; k < colNum1; ++k) {
                s00 += a[i * colNum1 + k] * b[k * colNum2 + j];
            }

            c[i * colNum2 + j] -= s00;
        }

        for (; j < colNum2; j += 3) {
            s00 = 0.0;
            s01 = 0.0;
            s02 = 0.0;
            for (k = 0; k < colNum1; ++k) {
                s00 += a[i * colNum1 + k] * b[k * colNum2 + j];
                s01 += a[i * colNum1 + k] * b[k * colNum2 + j + 1];
                s02 += a[i * colNum1 + k] * b[k * colNum2 + j + 2];
            }

            c[i * colNum2 + j] -= s00;
            c[i * colNum2 + j + 1] -= s01;
            c[i * colNum2 + j + 2] -= s02;
        }
    }

    for (; i < rowNum1; i += 3) {
        for (j = 0; j < horRem; ++j) {
            s00 = 0.0;
            s10 = 0.0;
            s20 = 0.0;
            for (k = 0; k < colNum1; ++k) {
                s00 += a[i * colNum1 + k] * b[k * colNum2 + j];
                s10 += a[(i + 1) * colNum1 + k] * b[k * colNum2 + j];
                s20 += a[(i + 2) * colNum1 + k] * b[k * colNum2 + j];
            }

            c[i * colNum2 + j] -= s00;
            c[(i + 1) * colNum2 + j] -= s10;
            c[(i + 2) * colNum2 + j] -= s20;
        }

        for (; j < colNum2; j += 3) {
            s00 = 0.0;
            s01 = 0.0;
            s02 = 0.0;
            s10 = 0.0;
            s11 = 0.0;
            s12 = 0.0;
            s20 = 0.0;
            s21 = 0.0;
            s22 = 0.0;
            for (k = 0; k < colNum1; ++k) {
                s00 += a[i * colNum1 + k] * b[k * colNum2 + j];
                s01 += a[i * colNum1 + k] * b[k * colNum2 + j + 1];
                s02 += a[i * colNum1 + k] * b[k * colNum2 + j + 2];
                s10 += a[(i + 1) * colNum1 + k] * b[k * colNum2 + j];
                s11 += a[(i + 1) * colNum1 + k] * b[k * colNum2 + j + 1];
                s12 += a[(i + 1) * colNum1 + k] * b[k * colNum2 + j + 2];
                s20 += a[(i + 2) * colNum1 + k] * b[k * colNum2 + j];
                s21 += a[(i + 2) * colNum1 + k] * b[k * colNum2 + j + 1];
                s22 += a[(i + 2) * colNum1 + k] * b[k * colNum2 + j + 2];
            }

            c[i * colNum2 + j] -= s00;
            c[i * colNum2 + j + 1] -= s01;
            c[i * colNum2 + j + 2] -= s02;
            c[(i + 1) * colNum2 + j] -= s10;
            c[(i + 1) * colNum2 + j + 1] -= s11;
            c[(i + 1) * colNum2 + j + 2] -= s12;
            c[(i + 2) * colNum2 + j] -= s20;
            c[(i + 2) * colNum2 + j + 1] -= s21;
            c[(i + 2) * colNum2 + j + 2] -= s22;
        }
    }
}

void procMultMatr(double *a, double *b, double *blockRow,
                    double *blockRow1, double *block1, double *mlBlock,
                    double *llBlock, int n, int procCols, int m, int p,
                    int k, MPI_Comm com)
{
    int fullBlocks = n / m;
    int remRows = n % m;
    int blocks = fullBlocks + (n % m > 0 ? 1 : 0);
    int i, j;
    int ownerProc;
    int localBlock;
    //int procBlockNum = numOfBlockColsInProc(n, m, p, k);
    //int globColInd = local_to_global(1, p, k, procBlockNum - 1);
    MPI_Status stat;
    for (i = 0; i < fullBlocks; ++i) {
        // gather i-th row in 0 proc, bcast to all
        for (j = 0; j < blocks; ++j) {
            ownerProc = j % p;
            localBlock = j / p;
            if (k == 0) {
                if (ownerProc == 0) {
                    if (j == fullBlocks) {
                        getBlock(procCols, m, i, localBlock, remRows, m, a, mlBlock);
                        putBlock(n, m, 0, j, remRows, m, blockRow, mlBlock);
                    } else {
                        getBlock(procCols, m, i, localBlock, m, m, a, block1);
                        putBlock(n, m, 0, j, m, m, blockRow, block1);
                    }
                } else {
                    if (j == fullBlocks) {
                        MPI_Recv(mlBlock, m * remRows, MPI_DOUBLE, ownerProc,
                                    0, com, &stat);
                        putBlock(n, m, 0, j, remRows, m, blockRow, mlBlock);
                    } else {
                        MPI_Recv(block1, m * m, MPI_DOUBLE, ownerProc,
                                    0, com, &stat);
                        putBlock(n, m, 0, j, m, m, blockRow, block1);
                    }
                }
            } else {
                if (k == ownerProc) {
                    if (j == fullBlocks) {
                        getBlock(procCols, m, i, localBlock, remRows,
                                    m, a, mlBlock);
                        MPI_Send(mlBlock, m * remRows, MPI_DOUBLE,
                                    0, 0, com);
                    } else {
                        getBlock(procCols, m, i, localBlock, m, m,
                                    a, block1);
                        MPI_Send(block1, m * m, MPI_DOUBLE, 0, 0, com);
                    }
                }
            }
        } // 0 proc has i-th row

        // send i-th row to all
        MPI_Bcast(blockRow, m * n, MPI_DOUBLE, 0, com);
        multMatr(blockRow, b, blockRow1, m, n, procCols, m);
        memcpy(a + i * m * procCols, blockRow1, m * procCols * sizeof(double));
    }

    if (remRows > 0) {
        for (i = 0; i < blocks; ++i) {
            ownerProc = i % p;
            localBlock = i / p;
            if (k == 0) {
                if (ownerProc == 0) {
                    if (i == fullBlocks) {
                        getBlock(procCols, m, fullBlocks, localBlock,
                                    remRows, remRows, a, llBlock);
                        putBlock(n, m, 0, i, remRows, remRows,
                                    blockRow, llBlock);
                    } else {
                        getBlock(procCols, m, fullBlocks, localBlock,
                                    m, remRows, a, mlBlock);
                        putBlock(n, m, 0, i, m, remRows, blockRow, mlBlock);
                    }
                } else {
                    if (i == fullBlocks) {
                        MPI_Recv(llBlock, remRows * remRows, MPI_DOUBLE,
                                    ownerProc, 0, com, &stat);
                        putBlock(n, m, 0, i, remRows, remRows, blockRow,
                                    llBlock);
                    } else {
                        MPI_Recv(mlBlock, remRows * m, MPI_DOUBLE,
                                    ownerProc, 0, com, &stat);
                        putBlock(n, m, 0, i, m, remRows, blockRow, mlBlock);
                    }
                }
            } else {
                if (k == ownerProc) {
                    if (i == fullBlocks) {
                        getBlock(procCols, m, fullBlocks, localBlock,
                                    remRows, remRows, a, llBlock);
                        MPI_Send(llBlock, remRows * remRows, MPI_DOUBLE,
                                    0, 0, com);
                    } else {
                        getBlock(procCols, m, fullBlocks, localBlock,
                                    m, remRows, a, mlBlock);
                        MPI_Send(mlBlock, remRows * m, MPI_DOUBLE,
                                    0, 0, com);
                    }
                }
            }
        }

        MPI_Bcast(blockRow, n * remRows, MPI_DOUBLE, 0, com);
        multMatr(blockRow, b, blockRow1, remRows, n, procCols, remRows);
        memcpy(a + fullBlocks * m * procCols, blockRow1,
                remRows * procCols * sizeof(double));
    }
}

void subUnitMatr(int n, int m, int p, int k, int procCols, double *a)
{
    int i, j;
    int j_global;
    for (i = 0; i < n; ++i) {
        for (j = 0; j < procCols; ++j) {
            j_global = local_to_global(m, p, k, j);
            if (i == j_global) {
                a[i * procCols + j] -= 1;
            }
        }
    }
}

double findRes(double *a, double *b, double *blockRow,
                    double *blockRow1, double *block1, double *mlBlock,
                    double *llBlock, int n, int procCols, int m, int p,
                    int k, MPI_Comm com)
{
    double res = 0.0;
    if (n <= 11000) {
        procMultMatr(a, b, blockRow, blockRow1, block1, mlBlock, llBlock,
                        n, procCols, m, p, k, com);
        subUnitMatr(n, m, p, k, procCols, a);
        res = procFindNorm(n, procCols, a, com);
    }

    //MPI_Barrier(com);
    return res;
} 
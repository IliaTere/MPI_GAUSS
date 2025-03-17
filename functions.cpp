#include "functions.h"

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
    //int local_n = numOfColsInProc(n, m, p, k);
    //int ownerProc;
    int rowsInBlock;
    int colsInBlock;
    int ibv; // block iterator from 0 to blockNum (vertical)
    int ibh; // block iterator from 0 to blockNum (horizontal)
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
    int i, j_loc, j_glob; // i_loc = i_glob
    //int localCols = numOfColsInProc(n, m, p, k);
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
    //int blockNum = (n + m - 1) / m;
    int locCurBlock;
    int colsToPrint;
    int printedCols = 0;
    int printSize = min(r, n);
    //int local_n = numOfColsInProc(n, m, p, k);
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
                    //printf("Proc 0 recieved ststus error: %d\n", stat.MPI_ERROR);
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

void initInvertedStatus(int p, int k, int locBlocks, int totalBlocks,
                        int *a)
{
    int i, j_loc, j_glob; // i_loc = i_glob = i
    //int locBlocks = numOfBlockColsInProc(n, m, p, k);
    //int totalBlocks = n / m;
    //totalBlocks += (n % m > 0 ? 1 : 0);
    for (i = 0; i < totalBlocks; ++i) {
        for (j_loc = 0; j_loc < locBlocks; ++j_loc) {
            j_glob = local_to_global(1, p, k, j_loc);
            a[i * locBlocks + j_loc] = (i == j_glob ? 1 : 0);
        }
    }
}

int simpleInvert(int n, double *a, double *b, double corEl, int *indicesTable)
{
    int i;
    int j;
    int k;
    double maxColEl;
    double curColEl;
    double div;
    double multElem;
    int iInd;
    int replaceInd;
    int tmpInd;
    for (i = 0; i < n; ++i) {
        indicesTable[i] = i;
    }

    for (i = 0; i < n; ++i) { // начало прямого хода
        j = 0;
        while (indicesTable[j] != i) {
            ++j;
        }

        iInd = j;
        replaceInd = iInd;
        maxColEl = fabs(a[iInd * n + i]);
        for (j = 0; j < n; ++j) { // поиск максиммума по итому столбцу
            if (indicesTable[j] > i) {
                curColEl = fabs(a[j * n + i]);
                if (curColEl > maxColEl) {
                    replaceInd = j;
                    maxColEl = curColEl;
                }
            }
        }

        if (fabs(maxColEl) < (EPS * corEl)) { // проверка на 0 столбец
            //printf("Block cannot be inverted.\n");
            return 1;
        }

        if (replaceInd != iInd) { // замена строк в основной матрице(с наибольшим элем делаем итой)
            indicesTable[iInd] = indicesTable[replaceInd];
            indicesTable[replaceInd] = i;
        }

        div = a[replaceInd * n + i];
        for (j = i + 1; j < n; ++j) { // деление итой строки на первый элемент
            a[replaceInd * n + j] /= div; 
        }

        for (j = 0; j < n; ++j) { // деление итой строки в присоед матр
            b[replaceInd * n + j] /= div;
        }

        for (j = 0; j < n; ++j) { // null rows in a and b under the ith row
            if (indicesTable[j] > i) {
                multElem = a[j * n + i];
                for (k = i + 1; k < n; ++k) {
                    a[j * n + k] -= (a[replaceInd * n + k] * multElem);
                }

                for (k = 0; k < n; ++k) {
                    b[j * n + k] -= (b[replaceInd * n + k] * multElem);
                }
            }
        }
    } // конец прямого хода

    for (i = n - 1; i > 0; --i) { // начало обратного хода (справа налево)
        tmpInd = 0;
        while (indicesTable[tmpInd] != i) {
            ++tmpInd;
        }

        for (j = 0; j < n; ++j) { // (снизу вверх)
            if (indicesTable[j] < i) {
                multElem = a[j * n + i];
                for (k = 0; k < n; ++k) { // (вычитание в присоединенной)
                    b[j * n + k] -= (b[tmpInd * n + k] * multElem);
                }
            }
        }
    } // конец обратного хода

    for (i = 0; i < n; ++i) {
        tmpInd = indicesTable[i];
        for (j = 0; j < n; ++j) {
            a[tmpInd * n + j] = b[i * n + j];
        }
    }

    return 0;
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
                        // for (int u = 0; u < m; ++u) {
                        //     for (int v = 0; v < remRows; ++v) {
                        //         printf("%lf ", mlBlock[u * remRows + v]);
                        //     }
                        //     printf("\n");
                        // }
                        // printf("\n\n");
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
        // for (int u = 0; u < m; ++u) {
        //     for (int v = 0; v < n; ++v) {
        //         printf("%lf ", blockRow[u * n + v]);
        //     }
        //     printf("\n");
        // }
        // printf("\n\n");
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

int blockInvert(int n, int m, int procCols, int p, int k, MPI_Comm com,
                double corEl, double *a, double *b, int *indicesTable,
                double *minInvertedNormList, double *minInvNormIndexTriple,
                int *invertedStatus, double *blockCol, double *block1,
                double *block2, double *block3, double *mlBlock,
                double *lmBlock, double *lmBlock1, double *llBlock,
                double *llBlock1, double *llBlock2)
{
    int i, j, q;
    int fullBlockRows = n / m; // vertical m * m blocks
    int remRows = n % m; // last vertical rows number
    int procBlockNum = numOfBlockColsInProc(n, m, p, k);
    int ownerProc;
    int localBlock;
    int remFullBlocks;
    int operationBlocks; // кол во блоков у процесса для выбора блока с мин нормой
    int operationStartBlock;
    int operationEndBlock;
    int operationAddBlocks;
    int procHorStartBlock; // с какоголокального столбца процесс начинает изм исходную матр
    double curMinInvertedNorm; // for searching min norm on i-th loop
    double minInvertedNorm; // min i-th inverted norm in proc
    //int curMinIndex; // for searching index of min inv norm
    int minIndex; // min i-th index
    int siCheck;
    int isInverted;
    int globInverted;
    double globInvertedNorm = 0.0;
    MPI_Status stat;
    int error = 0;
    int globColInd = local_to_global(1, p, k, procBlockNum - 1);
    int tmpInvStat;
    // начало прямого хода без остатоных строк
    for (i = 0; i < fullBlockRows; ++i) {
        remFullBlocks = fullBlockRows - i;
        ownerProc = i % p;
        procHorStartBlock = i / p;
        if (k == ownerProc) { // владелец столбца копирует ее в буфер для рассылки
            for (j = i; j < fullBlockRows; ++j) {
                getBlock(procCols, m, j, procHorStartBlock, m, m, a, block1);
                putBlock(m, m, j, 0, m, m, blockCol, block1);
            }

            if (remRows > 0) {
                getBlock(procCols, m, j, procHorStartBlock, m, remRows, a, lmBlock);
                putBlock(m, m, j, 0, m, remRows, blockCol, lmBlock);
            }
        } // конец копирования

        // раздача столбца всем процессам
        MPI_Bcast(blockCol + i * m * m,
                    remFullBlocks * m * m + remRows * m,
                    MPI_DOUBLE, ownerProc, com);
        procHorStartBlock += (ownerProc >= k ? 1 : 0);
        // выбор блока с мин нормой обр матрицы
        isInverted = 0;
        minInvertedNorm = -1.0;
        minIndex = -1;
        operationBlocks = remFullBlocks / p;
        operationAddBlocks = remFullBlocks % p;
        operationStartBlock = i + k * operationBlocks + min(k, operationAddBlocks);
        operationBlocks += (operationAddBlocks > k ? 1 : 0);
        operationEndBlock = operationStartBlock + operationBlocks;
        j = operationStartBlock;
        if (/*operationStartBlock < fullBlockRows &&*/ operationBlocks > 0) {
            getBlock(m, m, j, 0, m, m, blockCol, block1);
            initSimpleUnitMatrix(m, block2);
            siCheck = simpleInvert(m, block1, block2, corEl,
                                        indicesTable);
            if (siCheck != 0) {
                ++j;
            }

            while (siCheck != 0 && j != operationEndBlock) {
                getBlock(m, m, j, 0, m, m, blockCol, block1);
                initSimpleUnitMatrix(m, block2);
                siCheck = simpleInvert(m, block1, block2, corEl,
                                            indicesTable);
                if (siCheck != 0) {
                    ++j;
                }
            }

            if (siCheck == 0) {
                isInverted = 1;
                minIndex = j;
                minInvertedNorm = simpleFindNorm(m, block1);
                ++j;
                for (; j < operationEndBlock; ++j) {
                    getBlock(m, m, j, 0, m, m, blockCol, block1);
                    initSimpleUnitMatrix(m, block2);
                    siCheck = simpleInvert(m, block1, block2, corEl,
                                            indicesTable);
                    if (siCheck == 0) {
                        curMinInvertedNorm = simpleFindNorm(m, block1);
                        if (curMinInvertedNorm < minInvertedNorm) {
                            minInvertedNorm = curMinInvertedNorm;
                            minIndex = j;
                        }
                    }
                }
            }
        }

        minInvNormIndexTriple[0] = isInverted;
        minInvNormIndexTriple[1] = minInvertedNorm;
        minInvNormIndexTriple[2] = minIndex;
        if (k == 0) { // принять тройки - обратимость, мин норма, индекс
            minInvertedNormList[0] = isInverted;
            minInvertedNormList[1] = minInvertedNorm;
            minInvertedNormList[2] = minIndex;
            for (j = 1; j < p; ++j) {
                MPI_Recv(minInvertedNormList + j * 3, 3, MPI_DOUBLE, j,
                            0, com, &stat);
            }

            globInverted = 0;
            j = 0;
            while (globInverted < 1 && j < 3 * p) {
                globInverted = (int)(minInvertedNormList[j]);
                if (globInverted == 1) {
                    globInvertedNorm = minInvertedNormList[j + 1];
                    minIndex = (int)(minInvertedNormList[j + 2]);
                }

                j += 3;
            }

            if (globInverted == 0) {
                ++error;
            }

            for (; j < 3 * p; j+= 3) {
                if ((int)(minInvertedNormList[j]) == 1) {
                    minInvertedNorm = minInvertedNormList[j + 1];
                    if (minInvertedNorm < globInvertedNorm) {
                        globInvertedNorm = minInvertedNorm;
                        minIndex = (int)(minInvertedNormList[j + 2]);
                    }
                }
            }

            minInvNormIndexTriple[0] = error;
            minInvNormIndexTriple[1] = minIndex;
        } else {
            MPI_Send(minInvNormIndexTriple, 3, MPI_DOUBLE, 0, 0, com);
        }

        MPI_Bcast(minInvNormIndexTriple, 2, MPI_DOUBLE, 0, com);
        error = (int)(minInvNormIndexTriple[0]);
        if (error > 0) {
            if (k == 0) {
                printf("Matrix cannot be inverted.\n\n");
            }

            return 1;
        }

        minIndex = (int)(minInvNormIndexTriple[1]);
        // перестановка i и minIndex строк
        if (i != minIndex) {
            if (procBlockNum > 0) {
                for (j = procHorStartBlock; j < procBlockNum - 1; ++j) {
                    getBlock(procCols, m, i, j, m, m, a, block1);
                    getBlock(procCols, m, minIndex, j, m, m, a, block2);
                    putBlock(procCols, m, minIndex, j, m, m, a, block1);
                    putBlock(procCols, m, i, j, m, m, a, block2);
                }

                for (j = 0; j < procBlockNum - 1; ++j) {
                    getBlock(procCols, m, i, j, m, m, b, block1);
                    getBlock(procCols, m, minIndex, j, m, m, b, block2);
                    putBlock(procCols, m, minIndex, j, m, m, b, block1);
                    putBlock(procCols, m, i, j, m, m, b, block2);
                }

                //globColInd = local_to_global(1, p, k, j);
                if (remRows > 0 && globColInd == fullBlockRows) {
                    getBlock(procCols, m, i, j, remRows, m, a, mlBlock);
                    getBlock(procCols, m, minIndex, j, remRows, m, a, lmBlock);
                    putBlock(procCols, m, minIndex, j, remRows, m, a, mlBlock);
                    putBlock(procCols, m, i, j, remRows, m, a, lmBlock);
                    getBlock(procCols, m, i, j, remRows, m, b, mlBlock);
                    getBlock(procCols, m, minIndex, j, remRows, m, b, lmBlock);
                    putBlock(procCols, m, minIndex, j, remRows, m, b, mlBlock);
                    putBlock(procCols, m, i, j, remRows, m, b, lmBlock);
                } else {
                    getBlock(procCols, m, i, j, m, m, a, block1);
                    getBlock(procCols, m, minIndex, j, m, m, a, block2);
                    putBlock(procCols, m, minIndex, j, m, m, a, block1);
                    putBlock(procCols, m, i, j, m, m, a, block2);
                    getBlock(procCols, m, i, j, m, m, b, block1);
                    getBlock(procCols, m, minIndex, j, m, m, b, block2);
                    putBlock(procCols, m, minIndex, j, m, m, b, block1);
                    putBlock(procCols, m, i, j, m, m, b, block2);
                }

                getBlock(m, m, i, 0, m, m, blockCol, block1);
                getBlock(m, m, minIndex, 0, m, m, blockCol, block2);
                putBlock(m, m, minIndex, 0, m, m, blockCol, block1);
                putBlock(m, m, i, 0, m, m, blockCol, block2);
                for (j = 0; j < procBlockNum; ++j) {
                    tmpInvStat = invertedStatus[i * procBlockNum + j];
                    invertedStatus[i * procBlockNum + j] = invertedStatus[minIndex * procBlockNum + j];
                    invertedStatus[minIndex * procBlockNum + j] = tmpInvStat;
                }
            }
        } // i and minIndex rows are swaped

        // mult new i-th row using minNorm matrix (minIndex block in blockCol)
        if (procBlockNum > 0) {
            getBlock(m, m, i, 0, m, m, blockCol, block1);
            initSimpleUnitMatrix(m, block2);
            simpleInvert(m, block1, block2, corEl, indicesTable);
            //printf("%d %d %d %d\n", i, k, procHorStartBlock ,procBlockNum);
            for (j = procHorStartBlock; j < procBlockNum - 1; ++j) {
                getBlock(procCols, m, i, j, m, m, a, block2);
                multMatr(block1, block2, block3, m, m, m, m);
                putBlock(procCols, m, i, j, m, m, a, block3);
            }

            for (j = 0; j < procBlockNum - 1; ++j) {
                if (invertedStatus[i * procBlockNum + j] == 1) {
                    getBlock(procCols, m, i, j, m, m, b, block2);
                    multMatr(block1, block2, block3, m, m, m, m);
                    putBlock(procCols, m, i, j, m, m, b, block3);
                }
            }

            if (remRows > 0 && globColInd == fullBlockRows) {
                getBlock(procCols, m, i, j, remRows, m, a, mlBlock);
                multMatr(block1, mlBlock, lmBlock, m, m, remRows, remRows);
                putBlock(procCols, m, i, j, remRows, m, a, lmBlock);
                //printf("%d %d\n", i, k);
                if (invertedStatus[i * procBlockNum + j] == 1) {
                    getBlock(procCols, m, i, j, remRows, m, b, mlBlock);
                    multMatr(block1, mlBlock, lmBlock, m, m, remRows, remRows);
                    putBlock(procCols, m, i, j, remRows, m, b, lmBlock);
                }
            } else {
                getBlock(procCols, m, i, j, m, m, a, block2);
                multMatr(block1, block2, block3, m, m, m, m);
                putBlock(procCols, m, i, j, m, m, a, block3);
                if (invertedStatus[i * procBlockNum + j] == 1) {
                    getBlock(procCols, m, i, j, m, m, b, block2);
                    multMatr(block1, block2, block3, m, m, m, m);
                    putBlock(procCols, m, i, j, m, m, b, block3); 
                }
            }
        } // mult of i-th row end

        // sub from lower rows
        if (procBlockNum > 0) {
            for (j = i + 1; j < fullBlockRows; ++j) {
                getBlock(m, m, j, 0, m, m, blockCol, block1);
                for (q = procHorStartBlock; q < procBlockNum - 1; ++q) {
                    getBlock(procCols, m, i, q, m, m, a, block2);
                    getBlock(procCols, m, j, q, m, m, a, block3);
                    multSub(block1, block2, block3, m, m, m);
                    putBlock(procCols, m, j, q, m, m, a, block3);
                }

                for (q = 0; q < procBlockNum - 1; ++q) {
                    if (invertedStatus[i * procBlockNum + q] == 1) {
                        getBlock(procCols, m, i, q, m, m, b, block2);
                        getBlock(procCols, m, j, q, m, m, b, block3);
                        multSub(block1, block2, block3, m, m, m);
                        putBlock(procCols, m, j, q, m, m, b, block3);
                        invertedStatus[j * procBlockNum + q] = 1;
                    }
                }

                if (remRows > 0 && globColInd == fullBlockRows) {
                    getBlock(procCols, m, i, q, remRows, m, a, mlBlock);
                    getBlock(procCols, m, j, q, remRows, m, a, lmBlock);
                    multSub(block1, mlBlock, lmBlock, m, m, remRows);
                    putBlock(procCols, m, j, q, remRows, m, a, lmBlock);
                    if (invertedStatus[i * procBlockNum + q] == 1) {
                        getBlock(procCols, m, i, q, remRows, m, b, mlBlock);
                        getBlock(procCols, m, j, q, remRows, m, b, lmBlock);
                        multSub(block1, mlBlock, lmBlock, m, m, remRows);
                        putBlock(procCols, m, j, q, remRows, m, b, lmBlock);
                        invertedStatus[j * procBlockNum + q] = 1;
                    }
                } else {
                    getBlock(procCols, m, i, q, m, m, a, block2);
                    getBlock(procCols, m, j, q, m, m, a, block3);
                    multSub(block1, block2, block3, m, m, m);
                    putBlock(procCols, m, j, q, m, m, a, block3);
                    if (invertedStatus[i * procBlockNum + q] == 1) {
                        getBlock(procCols, m, i, q, m, m, b, block2);
                        getBlock(procCols, m, j, q, m, m, b, block3);
                        multSub(block1, block2, block3, m, m, m);
                        putBlock(procCols, m, j, q, m, m, b, block3);
                        invertedStatus[j * procBlockNum + q] = 1;
                    }
                }
            }

            if (remRows > 0) {
                getBlock(m, m, fullBlockRows, 0, m, remRows, blockCol, lmBlock);
                for (j = procHorStartBlock; j < procBlockNum - 1; ++j) {
                    getBlock(procCols, m, i, j, m, m, a, block1);
                    getBlock(procCols, m, fullBlockRows, j, m, remRows,
                                a, mlBlock);
                    multSub(lmBlock, block1, mlBlock, remRows, m, m);
                    putBlock(procCols, m, fullBlockRows, j, m, remRows,
                                a, mlBlock);
                }

                for (j = 0; j < procBlockNum - 1; ++j) {
                    if (invertedStatus[i * procBlockNum + j] == 1) {
                        getBlock(procCols, m, i, j, m, m, b, block1);
                        getBlock(procCols, m, fullBlockRows, j, m, remRows,
                                    b, mlBlock);
                        multSub(lmBlock, block1, mlBlock, remRows, m, m);
                        putBlock(procCols, m, fullBlockRows, j, m, remRows,
                                    b, mlBlock);
                        invertedStatus[fullBlockRows * procBlockNum + j] = 1;
                    }
                }

                if (globColInd == fullBlockRows) {
                    getBlock(procCols, m, i, j, remRows, m, a, mlBlock);
                    getBlock(procCols, m, fullBlockRows, j, remRows,
                                remRows, a, llBlock);
                    multSub(lmBlock, mlBlock, llBlock, remRows, m, remRows);
                    putBlock(procCols, m, fullBlockRows, j,
                                remRows, remRows, a, llBlock);
                    if (invertedStatus[i * procBlockNum + j] == 1) {
                        getBlock(procCols, m, i, j, remRows, m, b, mlBlock);
                        getBlock(procCols, m, fullBlockRows, j, remRows,
                                    remRows, b, llBlock);
                        multSub(lmBlock, mlBlock, llBlock, remRows, m, remRows);
                        putBlock(procCols, m, fullBlockRows, j, remRows,
                                    remRows, b, llBlock);
                    }
                } else {
                    getBlock(procCols, m, i, j, m, m, a, block1);
                    getBlock(procCols, m, fullBlockRows, j, m, remRows,
                                a, mlBlock);
                    multSub(lmBlock, block1, mlBlock, remRows, m, m);
                    putBlock(procCols, m, fullBlockRows, j, m, remRows,
                                a, mlBlock);
                    if (invertedStatus[i * procBlockNum + j] == 1) {
                        getBlock(procCols, m, i, j, m, m, b, block1);
                        getBlock(procCols, m, fullBlockRows, j, m, remRows,
                                    b, mlBlock);
                        multSub(lmBlock, block1, mlBlock, remRows, m, m);
                        putBlock(procCols, m, fullBlockRows, j, m, remRows,
                                    b, mlBlock);
                    }
                }
            }
        } // sub end
    } // конец прямого хода без остаточных строк

    if(remRows > 0) { // working with remRows
        ownerProc = fullBlockRows % p;
        if (k == ownerProc) { // owner of last col inverts block
            getBlock(procCols, m, fullBlockRows, procBlockNum - 1,
                        remRows, remRows, a, llBlock);
            initSimpleUnitMatrix(remRows, llBlock1);
            siCheck = simpleInvert(remRows, llBlock, llBlock1, corEl,
                                    indicesTable);
            if (siCheck != 0) {
                ++error;
            } else {
                for (i = 0; i < fullBlockRows; ++i) {
                    getBlock(procCols, m, i, procBlockNum - 1, remRows,
                                m, a, block1);
                    putBlock(m, m, i, 0, remRows, m, blockCol, block1);
                }

                putBlock(m, m, fullBlockRows, 0, remRows, remRows,
                            blockCol, llBlock);
            }
        }

        // inverted block and last col are received by other procs
        MPI_Bcast(&error, 1, MPI_INT, ownerProc, com);
        if (error > 0) {
            if (k == 0) {
                printf("Matrix cannot be inverted.\n\n");
            }

            return 1;
        } else {
            MPI_Bcast(blockCol, n * m, MPI_DOUBLE, ownerProc, com);
        }

        if (procBlockNum > 0) {
            getBlock(m, m, fullBlockRows, 0, remRows, remRows, blockCol,
                        llBlock);
            for (i = 0; i < procBlockNum - 1; ++i) { // mult last B row
                getBlock(procCols, m, fullBlockRows, i, m, remRows, b,
                            lmBlock);
                multMatr(llBlock, lmBlock, lmBlock1, remRows, remRows, m, remRows);
                putBlock(procCols, m, fullBlockRows, i, m, remRows, b,
                            lmBlock1);
            }

            if (globColInd == fullBlockRows) {
                getBlock(procCols, m, fullBlockRows, i, remRows, remRows,
                            b, llBlock1);
                multMatr(llBlock, llBlock1, llBlock2, remRows, remRows,
                            remRows, remRows);
                putBlock(procCols, m, fullBlockRows, i, remRows, remRows,
                            b, llBlock2);
            } else {
                getBlock(procCols, m, fullBlockRows, i, m, remRows, b,
                            lmBlock);
                multMatr(llBlock, lmBlock, lmBlock1, remRows, remRows, m, remRows);
                putBlock(procCols, m, fullBlockRows, i, m, remRows, b,
                            lmBlock1);
            }
        } // конец прямого хода

        // обработка посл столбца при обратном ходе
        if (procBlockNum > 0) {
            for (i = fullBlockRows - 1; i > -1; --i) {
                getBlock(m, m, i, 0, remRows, m, blockCol, mlBlock);
                for (j = 0; j < procBlockNum - 1; ++j) {
                    getBlock(procCols, m, fullBlockRows, j, m, remRows,
                                b, lmBlock);
                    getBlock(procCols, m, i, j, m, m, b, block1);
                    multSub(mlBlock, lmBlock, block1, m, remRows, m);
                    putBlock(procCols, m, i, j, m, m, b, block1);
                }

                if (globColInd == fullBlockRows) {
                    getBlock(procCols, m, fullBlockRows, j, remRows, remRows,
                                b, llBlock);
                    getBlock(procCols, m, i, j, remRows, m, b, lmBlock);
                    multSub(mlBlock, llBlock, lmBlock, m, remRows, remRows);
                    putBlock(procCols, m, i, j, remRows, m, b, lmBlock);
                } else {
                    getBlock(procCols, m, fullBlockRows, j, m, remRows,
                                b, lmBlock);
                    getBlock(procCols, m, i, j, m, m, b, block1);
                    multSub(mlBlock, lmBlock, block1, m, remRows, m);
                    putBlock(procCols, m, i, j, m, m, b, block1);
                }
            }
        }
    }

    //начало обратного хода
    for (i = fullBlockRows - 1; i > 0; --i) {
        ownerProc = i % p;
        localBlock = i / p;
        if (k == ownerProc) {
            for (j = 0; j < i; ++j) {
                getBlock(procCols, m, j, localBlock, m, m, a, block1);
                //printf("%d %lf %lf\n", k, block1[0], a[j * procCols + i]);
                putBlock(m, m, j, 0, m, m, blockCol, block1);
            }
        }

        MPI_Bcast(blockCol, i * m * m, MPI_DOUBLE, ownerProc, com);
        if (procBlockNum > 0) {
            for (j = i - 1; j > -1; --j) {
                getBlock(m, m, j, 0, m, m, blockCol, block1);
                for (q = 0; q < procBlockNum - 1; ++q) {
                    getBlock(procCols, m, i, q, m, m, b, block2);
                    getBlock(procCols, m, j, q, m, m, b, block3);
                    multSub(block1, block2, block3, m, m, m);
                    putBlock(procCols, m, j, q, m, m, b, block3);
                }

                if (globColInd == fullBlockRows) {
                    getBlock(procCols, m, i, q, remRows, m, b, mlBlock);
                    getBlock(procCols, m, j, q, remRows, m, b, lmBlock);
                    multSub(block1, mlBlock, lmBlock, m, m, remRows);
                    putBlock(procCols, m, j, q, remRows, m, b, lmBlock);
                } else {
                    getBlock(procCols, m, i, q, m, m, b, block2);
                    getBlock(procCols, m, j, q, m, m, b, block3);
                    multSub(block1, block2, block3, m, m, m);
                    putBlock(procCols, m, j, q, m, m, b, block3);
                }
            }
        }
    }

    return 0;
}

void subUnitMatr(int n, int m, int p, int k, int procCols,
                        double *a)
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

void freeAllMemory(double *matrix, double *invertedMatrix, double *blockStringBuf, double *blockRow,
                   double *block1, double *block2, double *block3, double *mlBlock, double *lmBlock,
                   double *lmBlock1, double *llBlock, double *llBlock1, double *llBlock2, int *indicesTable,
                   double *minInvertedNormList, double *minInvNormIndexTriple, int *invertedStatus/*,
                   int procCols, int remSize*/)
{
    if (matrix != nullptr) {
        delete [] matrix;
    }

    if (invertedMatrix != nullptr) {
        delete [] invertedMatrix;
    }

    if (invertedStatus != nullptr) {
        delete [] invertedStatus;
    }

    if (blockRow != nullptr) {
        delete [] blockRow;
    }

    if (mlBlock != nullptr) {
        delete [] mlBlock;
    }

    if (lmBlock != nullptr) {
        delete [] lmBlock;
    }

    if (lmBlock1 != nullptr) {
        delete [] lmBlock1;
    }

    if (llBlock != nullptr) {
        delete [] llBlock;
    }

    if (llBlock1 != nullptr) {
        delete [] llBlock1;
    }

    if (llBlock2 != nullptr) {
        delete [] llBlock2;
    }

    if (blockStringBuf != nullptr) {
        delete [] blockStringBuf;
    }

    if (block1 != nullptr) {
        delete [] block1;
    }

    if (block2 != nullptr) {
        delete [] block2;
    }

    if (block3 != nullptr) {
        delete [] block3;
    }

    if (indicesTable != nullptr) {
        delete [] indicesTable;
    }

    if (minInvertedNormList != nullptr) {
        delete [] minInvertedNormList;
    }

    if (minInvNormIndexTriple != nullptr) {
        delete [] minInvNormIndexTriple;
    }
}

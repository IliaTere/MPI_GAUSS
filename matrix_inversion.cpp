#include "matrix_inversion.h"
#include "utils.h"
#include "matrix_operations.h"
#include <cmath>
#include <cstdio>

void initInvertedStatus(int p, int k, int locBlocks, int totalBlocks, int *a)
{
    int i, j_loc, j_glob;
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

    for (i = 0; i < n; ++i) {       
        j = 0;
        while (indicesTable[j] != i) {
            ++j;
        }

        iInd = j;
        replaceInd = iInd;
        maxColEl = fabs(a[iInd * n + i]);
        for (j = 0; j < n; ++j) {
            if (indicesTable[j] > i) {
                curColEl = fabs(a[j * n + i]);
                if (curColEl > maxColEl) {
                    replaceInd = j;
                    maxColEl = curColEl;
                }
            }
        }

        if (fabs(maxColEl) < (EPS * corEl)) {
            return 1;
        }

        if (replaceInd != iInd) {
            indicesTable[iInd] = indicesTable[replaceInd];
            indicesTable[replaceInd] = i;
        }

        div = a[replaceInd * n + i];
        for (j = i + 1; j < n; ++j) {
            a[replaceInd * n + j] /= div; 
        }

        for (j = 0; j < n; ++j) {
            b[replaceInd * n + j] /= div;
        }

        for (j = 0; j < n; ++j) {
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
    }

    for (i = n - 1; i > 0; --i) {
        tmpInd = 0;
        while (indicesTable[tmpInd] != i) {
            ++tmpInd;
        }

        for (j = 0; j < n; ++j) {
            if (indicesTable[j] < i) {
                multElem = a[j * n + i];
                for (k = 0; k < n; ++k) {
                    b[j * n + k] -= (b[tmpInd * n + k] * multElem);
                }
            }
        }
    }

    for (i = 0; i < n; ++i) {
        tmpInd = indicesTable[i];
        for (j = 0; j < n; ++j) {
            a[tmpInd * n + j] = b[i * n + j];
        }
    }

    return 0;
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
    int fullBlockRows = n / m;
    int remRows = n % m;
    int procBlockNum = numOfBlockColsInProc(n, m, p, k);
    int ownerProc, localBlock, remFullBlocks, operationBlocks, operationStartBlock, operationEndBlock, operationAddBlocks, procHorStartBlock;
    double curMinInvertedNorm, minInvertedNorm;
    int minIndex, siCheck, isInverted, globInverted;
    double globInvertedNorm = 0.0;
    MPI_Status stat;
    int error = 0;
    int globColInd = local_to_global(1, p, k, procBlockNum - 1);
    int tmpInvStat;
    for (i = 0; i < fullBlockRows; ++i) {
        remFullBlocks = fullBlockRows - i;
        ownerProc = i % p;
        procHorStartBlock = i / p;
        if (k == ownerProc) {
            for (j = i; j < fullBlockRows; ++j) {
                getBlock(procCols, m, j, procHorStartBlock, m, m, a, block1);
                putBlock(m, m, j, 0, m, m, blockCol, block1);
            }

            if (remRows > 0) {
                getBlock(procCols, m, j, procHorStartBlock, m, remRows, a, lmBlock);
                putBlock(m, m, j, 0, m, remRows, blockCol, lmBlock);
            }
        }

        MPI_Bcast(blockCol + i * m * m,
                    remFullBlocks * m * m + remRows * m,
                    MPI_DOUBLE, ownerProc, com);
        procHorStartBlock += (ownerProc >= k ? 1 : 0);
        isInverted = 0;
        minInvertedNorm = -1.0;
        minIndex = -1;
        operationBlocks = remFullBlocks / p;
        operationAddBlocks = remFullBlocks % p;
        operationStartBlock = i + k * operationBlocks + min(k, operationAddBlocks);
        operationBlocks += (operationAddBlocks > k ? 1 : 0);
        operationEndBlock = operationStartBlock + operationBlocks;
        j = operationStartBlock;
        if ( operationBlocks > 0) {
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
        if (k == 0) {   
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
        }

        if (procBlockNum > 0) {
            getBlock(m, m, i, 0, m, m, blockCol, block1);
            initSimpleUnitMatrix(m, block2);
            simpleInvert(m, block1, block2, corEl, indicesTable);
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
        }   

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
        }
    }

    if(remRows > 0) {
        ownerProc = fullBlockRows % p;
        if (k == ownerProc) {
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
            for (i = 0; i < procBlockNum - 1; ++i) {
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
        }

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

    for (i = fullBlockRows - 1; i > 0; --i) {
        ownerProc = i % p;
        localBlock = i / p;
        if (k == ownerProc) {
            for (j = 0; j < i; ++j) {
                getBlock(procCols, m, j, localBlock, m, m, a, block1);
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
#include <stdio.h>
#include <sstream>
#include <cstdlib>
#include <ctime>
#include <mpi.h>
#include "utils.h"
#include "matrix_io.h"
#include "matrix_operations.h"
#include "matrix_inversion.h"
#include "memory.h"

int main (int argc, char *argv[])
{
    int totalProc, curProc; // p, k
    int matrixSize; // n
    int blockSize; // m
    int outSize; // r
    int initFormula; // s
    char *inputFileName = nullptr; // filename
    double *matrix = nullptr;
    double *invertedMatrix = nullptr;
    double *blockStringBuf = nullptr;
    double *blockRow = nullptr;
    double *block1 = nullptr;
    double *block2 = nullptr;
    double *block3 = nullptr;
    double *mlBlock = nullptr;
    double *lmBlock = nullptr;
    double *lmBlock1 = nullptr;
    double *llBlock = nullptr;
    double *llBlock1 = nullptr;
    double *llBlock2 = nullptr;
    int *indicesTable = nullptr;
    double *minInvertedNormList = nullptr; // 3 * p, triples - isInverted + invNorm + index for p - 1
    double *minInvNormIndexTriple = nullptr;
    int *invertedStatus = nullptr;
    int procColNum;
    int locBlocks;
    int remSize;
    int totalBlocks;
    int error = 0;
    double corEl;
    double solutionTime = 0.0;
    double errorTime = 0.0;
    double r1, r2;
    MPI_Comm com = MPI_COMM_WORLD;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(com, &totalProc);
    MPI_Comm_rank(com, &curProc);
    switch (argc) {
        case 5:
            std::istringstream(argv[1]) >> matrixSize;
            std::istringstream(argv[2]) >> blockSize;
            std::istringstream(argv[3]) >> outSize;
            std::istringstream(argv[4]) >> initFormula;
            break;
        case 6:
            std::istringstream(argv[1]) >> matrixSize;
            std::istringstream(argv[2]) >> blockSize;
            std::istringstream(argv[3]) >> outSize;
            std::istringstream(argv[4]) >> initFormula;
            inputFileName = argv[5];
            break;
        default:
            if (curProc == 0) {
                printf("Wrong amount of arguments.\n");
            }

            return -1;
    }

    locBlocks = numOfBlockColsInProc(matrixSize, blockSize, totalProc, curProc);
    procColNum = numOfColsInProc(matrixSize, blockSize, totalProc, curProc);
    totalBlocks = matrixSize / blockSize;
    totalBlocks += (matrixSize % blockSize > 0 ? 1 : 0);
    remSize = matrixSize % blockSize;
    if (procColNum > 0) {
        matrix = new double[matrixSize * procColNum];
        invertedMatrix = new double[matrixSize * procColNum];
        invertedStatus = new int[totalBlocks * locBlocks];
        blockRow = new double[blockSize * procColNum];
    }

    if (remSize > 0) {
        mlBlock = new double[blockSize * remSize];
        lmBlock = new double[remSize * blockSize];
        lmBlock1 = new double[remSize * blockSize];
        llBlock = new double[remSize * remSize];
        llBlock1 = new double[remSize * remSize];
        llBlock2 = new double[remSize * remSize];
    }

    blockStringBuf = new double[blockSize * matrixSize];
    block1 = new double[blockSize * blockSize];
    block2 = new double[blockSize * blockSize];
    block3 = new double[blockSize * blockSize];
    indicesTable = new int[blockSize];
    minInvertedNormList = new double[3 * totalProc];
    minInvNormIndexTriple = new double[3];
    if (initFormula == 1 || initFormula == 2 ||
        initFormula == 3 || initFormula == 4) {
        initMatrix(matrix, matrixSize, blockSize, procColNum,
                    totalProc, curProc, initFormula, partialInit);
    } else if (initFormula == 0) {
        error += readMatrix(matrix, matrixSize, blockSize, totalProc,
                            curProc, procColNum, inputFileName,
                            blockStringBuf, block1, com);
    }

    //printf("Proc %d, error %d\n", curProc, error);
    if (error != 0) {
        freeAllMemory(matrix, invertedMatrix, blockStringBuf, blockRow, block1,
                      block2, block3, mlBlock, lmBlock, lmBlock1, llBlock,
                      llBlock1, llBlock2, indicesTable, minInvertedNormList, minInvNormIndexTriple,
                      invertedStatus);
        if (curProc == 0) {
            printf("Reading error.\n");
        }

        MPI_Finalize();
        return -1;
    }

    printMatrix(matrix, matrixSize, blockSize, totalProc, curProc,
                procColNum, block1, outSize, com);
    corEl = findCorEl(matrixSize, blockSize, procColNum, curProc, totalProc,
                        matrix, com, initFormula);
    procInitUnitMatrix(matrixSize, blockSize, totalProc, curProc,
                        procColNum, invertedMatrix);
    initInvertedStatus(totalProc, curProc, locBlocks, totalBlocks,
                        invertedStatus);
    MPI_Barrier(com);
    solutionTime = MPI_Wtime();
    error = blockInvert(matrixSize, blockSize, procColNum, totalProc, curProc,
            com, corEl, matrix, invertedMatrix, indicesTable,
            minInvertedNormList, minInvNormIndexTriple, invertedStatus,
            blockStringBuf, block1, block2, block3, mlBlock, lmBlock,
            lmBlock1, llBlock, llBlock1, llBlock2);
    MPI_Barrier(com);
    solutionTime = MPI_Wtime() - solutionTime;
    if (error > 0) {
        freeAllMemory(matrix, invertedMatrix, blockStringBuf, blockRow, block1,
                      block2, block3, mlBlock, lmBlock, lmBlock1, llBlock,
                      llBlock1, llBlock2, indicesTable, minInvertedNormList, minInvNormIndexTriple,
                      invertedStatus);
        if (curProc == 0) {
            //printf("Something went wrong.\n");
            printf("%s : Task = 12 Res1 = -1 Res2 = -1 T1 = %.2f T2 = 0 S = %d N = %d M = %d P = %d\n",
                    argv[0], solutionTime, initFormula, matrixSize, blockSize, totalProc);
        }

        MPI_Finalize();
        return 1;
    }

    printMatrix(invertedMatrix, matrixSize, blockSize, totalProc, curProc,
                procColNum, block1, outSize, com);
    if (initFormula == 1 || initFormula == 2 ||
        initFormula == 3 || initFormula == 4) {
        initMatrix(matrix, matrixSize, blockSize, procColNum,
                    totalProc, curProc, initFormula, partialInit);
    } else if (initFormula == 0) {
        readMatrix(matrix, matrixSize, blockSize, totalProc,
                    curProc, procColNum, inputFileName,
                    blockStringBuf, block1, com);
    }

    MPI_Barrier(com);
    errorTime = MPI_Wtime();
    r1 = findRes(matrix, invertedMatrix, blockStringBuf, blockRow,
                    block1, mlBlock, llBlock, matrixSize, procColNum,
                    blockSize, totalProc, curProc, com);
    if (initFormula == 1 || initFormula == 2 ||
        initFormula == 3 || initFormula == 4) {
        initMatrix(matrix, matrixSize, blockSize, procColNum,
                    totalProc, curProc, initFormula, partialInit);
    } else if (initFormula == 0) {
        readMatrix(matrix, matrixSize, blockSize, totalProc,
                    curProc, procColNum, inputFileName,
                    blockStringBuf, block1, com);
    }

    r2 = findRes(invertedMatrix, matrix, blockStringBuf, blockRow,
                    block1, mlBlock, llBlock, matrixSize, procColNum,
                    blockSize, totalProc, curProc, com);
    MPI_Barrier(com);
    errorTime = MPI_Wtime() - errorTime;
    if (curProc == 0) {
        printf("%s : Task = 12 Res1 = %e Res2 = %e T1 = %.2f T2 = %.2f S = %d N = %d M = %d P = %d\n",
                argv[0], r1, r2, solutionTime, errorTime, initFormula, matrixSize, blockSize, totalProc);
    }

    freeAllMemory(matrix, invertedMatrix, blockStringBuf, blockRow, block1,
                  block2, block3, mlBlock, lmBlock, lmBlock1, llBlock,
                  llBlock1, llBlock2, indicesTable, minInvertedNormList, minInvNormIndexTriple,
                  invertedStatus);

    MPI_Finalize();
    return 0;
}

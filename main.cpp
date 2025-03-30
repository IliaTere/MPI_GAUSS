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
    MPI_Comm communicator = MPI_COMM_WORLD;
    MPI_Init(&argc, &argv);
    
    int processCount, processRank;
    MPI_Comm_size(communicator, &processCount);
    MPI_Comm_rank(communicator, &processRank);
    
    int matrixDimension;    // n
    int blockDimension;     // m
    int outputSize;         // r
    int initMethod;         // s
    char *inputFile = nullptr; // filename
    
    bool validArguments = true;
    if (argc == 5) {
        std::istringstream(argv[1]) >> matrixDimension;
        std::istringstream(argv[2]) >> blockDimension;
        std::istringstream(argv[3]) >> outputSize;
        std::istringstream(argv[4]) >> initMethod;
    } else if (argc == 6) {
        std::istringstream(argv[1]) >> matrixDimension;
        std::istringstream(argv[2]) >> blockDimension;
        std::istringstream(argv[3]) >> outputSize;
        std::istringstream(argv[4]) >> initMethod;
        inputFile = argv[5];
    } else {
        validArguments = false;
        if (processRank == 0) {
            printf("Wrong amount of arguments.\n");
        }
    }
    
    if (!validArguments) {
        MPI_Finalize();
        return -1;
    }
    
    int localBlockCount = numOfBlockColsInProc(matrixDimension, blockDimension, processCount, processRank);
    int localColumnCount = numOfColsInProc(matrixDimension, blockDimension, processCount, processRank);
    int totalBlockCount = (matrixDimension + blockDimension - 1) / blockDimension;
    int remainderSize = matrixDimension % blockDimension;
    
    double *matrix = nullptr;
    double *invertedMatrix = nullptr;
    int *invertedStatus = nullptr;
    double *blockRow = nullptr;
    
    if (localColumnCount > 0) {
        matrix = new double[matrixDimension * localColumnCount];
        invertedMatrix = new double[matrixDimension * localColumnCount];
        invertedStatus = new int[totalBlockCount * localBlockCount];
        blockRow = new double[blockDimension * localColumnCount];
    }
    
    double *mlBlock = nullptr;
    double *lmBlock = nullptr;
    double *lmBlock1 = nullptr;
    double *llBlock = nullptr;
    double *llBlock1 = nullptr;
    double *llBlock2 = nullptr;
    
    if (remainderSize > 0) {
        mlBlock = new double[blockDimension * remainderSize];
        lmBlock = new double[remainderSize * blockDimension];
        lmBlock1 = new double[remainderSize * blockDimension];
        llBlock = new double[remainderSize * remainderSize];
        llBlock1 = new double[remainderSize * remainderSize];
        llBlock2 = new double[remainderSize * remainderSize];
    }
    
    double *blockStringBuf = new double[blockDimension * matrixDimension];
    double *block1 = new double[blockDimension * blockDimension];
    double *block2 = new double[blockDimension * blockDimension];
    double *block3 = new double[blockDimension * blockDimension];
    int *indicesTable = new int[blockDimension];
    double *minInvertedNormList = new double[3 * processCount];
    double *minInvNormIndexTriple = new double[3];
    
    int errorCode = 0;
    if (initMethod >= 1 && initMethod <= 4) {
        initMatrix(matrix, matrixDimension, blockDimension, localColumnCount,
                  processCount, processRank, initMethod, partialInit);
    } else if (initMethod == 0) {
        errorCode = readMatrix(matrix, matrixDimension, blockDimension, processCount,
                              processRank, localColumnCount, inputFile,
                              blockStringBuf, block1, communicator);
    }
    
    if (errorCode != 0) {
        freeAllMemory(matrix, invertedMatrix, blockStringBuf, blockRow, block1,
                      block2, block3, mlBlock, lmBlock, lmBlock1, llBlock,
                      llBlock1, llBlock2, indicesTable, minInvertedNormList, 
                      minInvNormIndexTriple, invertedStatus);
        if (processRank == 0) {
            printf("Reading error.\n");
        }
        MPI_Finalize();
        return -1;
    }
    
    printMatrix(matrix, matrixDimension, blockDimension, processCount, processRank,
                localColumnCount, block1, outputSize, communicator);
    
    double correlationElement = findCorEl(matrixDimension, blockDimension, localColumnCount, 
                                          processRank, processCount, matrix, communicator, initMethod);
    procInitUnitMatrix(matrixDimension, blockDimension, processCount, processRank,
                       localColumnCount, invertedMatrix);
    initInvertedStatus(processCount, processRank, localBlockCount, totalBlockCount,
                       invertedStatus);
    
    MPI_Barrier(communicator);
    double startTime = MPI_Wtime();
    
    errorCode = blockInvert(matrixDimension, blockDimension, localColumnCount, processCount, 
                           processRank, communicator, correlationElement, matrix, invertedMatrix, 
                           indicesTable, minInvertedNormList, minInvNormIndexTriple, invertedStatus,
                           blockStringBuf, block1, block2, block3, mlBlock, lmBlock,
                           lmBlock1, llBlock, llBlock1, llBlock2);
    
    MPI_Barrier(communicator);
    double endTime = MPI_Wtime();
    double executionTime = endTime - startTime;
    

    if (errorCode > 0) {
        freeAllMemory(matrix, invertedMatrix, blockStringBuf, blockRow, block1,
                      block2, block3, mlBlock, lmBlock, lmBlock1, llBlock,
                      llBlock1, llBlock2, indicesTable, minInvertedNormList, 
                      minInvNormIndexTriple, invertedStatus);
        if (processRank == 0) {
            printf("%s : Task = 12 Res1 = -1 Res2 = -1 T1 = %.2f T2 = 0 S = %d N = %d M = %d P = %d\n",
                   argv[0], executionTime, initMethod, matrixDimension, blockDimension, processCount);
        }
        MPI_Finalize();
        return 1;
    }
    
    printMatrix(invertedMatrix, matrixDimension, blockDimension, processCount, processRank,
                localColumnCount, block1, outputSize, communicator);
    
    if (initMethod >= 1 && initMethod <= 4) {
        initMatrix(matrix, matrixDimension, blockDimension, localColumnCount,
                  processCount, processRank, initMethod, partialInit);
    } else if (initMethod == 0) {
        readMatrix(matrix, matrixDimension, blockDimension, processCount,
                  processRank, localColumnCount, inputFile,
                  blockStringBuf, block1, communicator);
    }
    
    MPI_Barrier(communicator);
    startTime = MPI_Wtime();
    
    double residual1 = findRes(matrix, invertedMatrix, blockStringBuf, blockRow,
                              block1, mlBlock, llBlock, matrixDimension, localColumnCount,
                              blockDimension, processCount, processRank, communicator);
    
    if (initMethod >= 1 && initMethod <= 4) {
        initMatrix(matrix, matrixDimension, blockDimension, localColumnCount,
                  processCount, processRank, initMethod, partialInit);
    } else if (initMethod == 0) {
        readMatrix(matrix, matrixDimension, blockDimension, processCount,
                  processRank, localColumnCount, inputFile,
                  blockStringBuf, block1, communicator);
    }
    
    double residual2 = findRes(invertedMatrix, matrix, blockStringBuf, blockRow,
                              block1, mlBlock, llBlock, matrixDimension, localColumnCount,
                              blockDimension, processCount, processRank, communicator);
    
    MPI_Barrier(communicator);
    endTime = MPI_Wtime();
    double verificationTime = endTime - startTime;
    
    if (processRank == 0) {
        printf("%s : Task = 12 Res1 = %e Res2 = %e T1 = %.2f T2 = %.2f S = %d N = %d M = %d P = %d\n",
               argv[0], residual1, residual2, executionTime, verificationTime, 
               initMethod, matrixDimension, blockDimension, processCount);
    }
    
    freeAllMemory(matrix, invertedMatrix, blockStringBuf, blockRow, block1,
                  block2, block3, mlBlock, lmBlock, lmBlock1, llBlock,
                  llBlock1, llBlock2, indicesTable, minInvertedNormList, 
                  minInvNormIndexTriple, invertedStatus);
    
    MPI_Finalize();
    return 0;
}

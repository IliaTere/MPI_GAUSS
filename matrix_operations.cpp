#include "matrix_operations.h"
#include "utils.h"
#include <immintrin.h>
#include <cstring>
#include <cmath>

void getBlock(int n, int m, int indv, int indh, int l,
                int h, double *a, double *block)
{
    double* sourceMatrix = a + (indv * n * m) + (indh * m);
    
    for (int row = 0; row < h; row++) {
        double* sourceRow = sourceMatrix + (row * n);
        double* destRow = block + (row * l);
        
        memcpy(destRow, sourceRow, sizeof(double) * l);
    }
}

void putBlock(int n, int m, int indv, int indh, int l,
                int h, double *a, double *block)
{
    double* destStart = a + (indv * n * m) + (indh * m);
    
    for (int rowIdx = 0; rowIdx < h; rowIdx++) {
        double* srcPtr = block + (rowIdx * l);
        double* dstPtr = destStart + (rowIdx * n);
        
        memcpy(dstPtr, srcPtr, l * sizeof(double));
    }
}

void initSimpleUnitMatrix(int n, double *a)
{
    for (int i = 0; i < n * n; i++) {
        a[i] = 0.0;
    }
    
    for (int diag = 0; diag < n; diag++) {
        a[diag * n + diag] = 1.0;
    }
}

double simpleFindNorm(int n, double *a)
{
    double maxRowSum = 0.0;
    
    for (int row = 0; row < n; row++) {
        double rowSum = 0.0;
        for (int col = 0; col < n; col++) {
            rowSum += fabs(a[row + col * n]);
        }
        
        if (rowSum > maxRowSum) {
            maxRowSum = rowSum;
        }
    }
    
    return maxRowSum;
}

double procFindNorm(int n, int procCol, double *a, MPI_Comm com)
{
    double localMax = 0.0;
    double globalMax = 0.0;
    
    for (int currentRow = 0; currentRow < procCol; currentRow++) {
        double rowSum = std::fabs(a[currentRow]);
        
        for (int colOffset = 1; colOffset < n; colOffset++) {
            int elementIndex = currentRow + colOffset * procCol;
            rowSum += std::fabs(a[elementIndex]);
        }
        
        if (rowSum > localMax) {
            localMax = rowSum;
        }
    }
    
    MPI_Allreduce(
        &localMax,
        &globalMax,
        1,
        MPI_DOUBLE,
        MPI_MAX,
        com
    );
    
    return globalMax;
}

double findCorEl(int n, int m, int procCols, int k, int p, double *a,
                    MPI_Comm com, int s)
{
    double corEl = 0.0;
    int lastColProc = getNumOfProc(m, p, n - 1);
    if (k == lastColProc) {
        corEl = a[n * procCols - 1];
    }

    MPI_Bcast(&corEl, 1, MPI_DOUBLE, lastColProc, com);
    corEl = (s == 4 ? corEl * corEl : procFindNorm(n, procCols, a, com));
    return corEl;
}

// void mult_sse(double *a, double *b, double *res, int m1, int m2, int m3, int m /*double norm, int n*/)
// {
//     // Оптимизированная версия для x86 архитектуры
//     // double upper_threshold = (n < 11000) ? 1e+200 * norm : 1e+16 * norm;
//     // double lower_threshold = (n < 11000) ? 1e-100 * norm : 1e-16 * norm;
//     // int count_b = 0;
    
//     // Проверка значений на пороговые величины и обнуление результата
//     for (int i = 0; i < m; i++)
//     {
//         for (int j = 0; j < m; j++)
//         {
//             res[i * m + j] = 0.0;
//             // double local = fabs(b[i*m+j]);
//             // if (upper_threshold < local || local < lower_threshold)
//             // {
//             //     b[i * m + j] = 0.;
//             //     count_b++;
//             // }
//         }
//     }
    
//     // Если все значения в b были обнулены, возвращаем нулевую матрицу
//     // if(count_b == m*m) {
//     //     return;
//     // }
    
//     int v = m1, h = m3, ah = m2;
//     int i, j, k;
    
//     // Обработка блоками 2x2 (развертывание внешних циклов)
//     for (i = 0; i <= v - 2; i += 2)
//     {
//         for (j = 0; j <= h - 2; j += 2)
//         {
//             // Используем 4 аккумулятора SSE для 2x2 блока результата
//             __m128d sum00 = _mm_setzero_pd();
//             __m128d sum01 = _mm_setzero_pd();
//             __m128d sum10 = _mm_setzero_pd();
//             __m128d sum11 = _mm_setzero_pd();
            
//             // Развертывание внутреннего цикла для более эффективного использования SSE
//             for (k = 0; k <= ah - 4; k += 4)
//             {
//                 // Загрузка строк матрицы a
//                 __m128d a0_vec0 = _mm_set_pd(a[i * m + (k+1)], a[i * m + k]);
//                 __m128d a0_vec1 = _mm_set_pd(a[i * m + (k+3)], a[i * m + (k+2)]);
//                 __m128d a1_vec0 = _mm_set_pd(a[(i+1) * m + (k+1)], a[(i+1) * m + k]);
//                 __m128d a1_vec1 = _mm_set_pd(a[(i+1) * m + (k+3)], a[(i+1) * m + (k+2)]);
                
//                 // Загрузка столбцов матрицы b
//                 __m128d b0_vec0 = _mm_set_pd(b[(k+1) * m + j], b[k * m + j]);
//                 __m128d b0_vec1 = _mm_set_pd(b[(k+3) * m + j], b[(k+2) * m + j]);
//                 __m128d b1_vec0 = _mm_set_pd(b[(k+1) * m + (j+1)], b[k * m + (j+1)]);
//                 __m128d b1_vec1 = _mm_set_pd(b[(k+3) * m + (j+1)], b[(k+2) * m + (j+1)]);
                
//                 // Умножение и накопление для блока 2x2
//                 sum00 = _mm_add_pd(sum00, _mm_mul_pd(a0_vec0, b0_vec0));
//                 sum00 = _mm_add_pd(sum00, _mm_mul_pd(a0_vec1, b0_vec1));
                
//                 sum01 = _mm_add_pd(sum01, _mm_mul_pd(a0_vec0, b1_vec0));
//                 sum01 = _mm_add_pd(sum01, _mm_mul_pd(a0_vec1, b1_vec1));
                
//                 sum10 = _mm_add_pd(sum10, _mm_mul_pd(a1_vec0, b0_vec0));
//                 sum10 = _mm_add_pd(sum10, _mm_mul_pd(a1_vec1, b0_vec1));
                
//                 sum11 = _mm_add_pd(sum11, _mm_mul_pd(a1_vec0, b1_vec0));
//                 sum11 = _mm_add_pd(sum11, _mm_mul_pd(a1_vec1, b1_vec1));
//             }
            
//             // Обработка оставшихся элементов
//             for (; k <= ah - 2; k += 2)
//             {
//                 __m128d a0_vec = _mm_set_pd(a[i * m + (k+1)], a[i * m + k]);
//                 __m128d a1_vec = _mm_set_pd(a[(i+1) * m + (k+1)], a[(i+1) * m + k]);
                
//                 __m128d b0_vec = _mm_set_pd(b[(k+1) * m + j], b[k * m + j]);
//                 __m128d b1_vec = _mm_set_pd(b[(k+1) * m + (j+1)], b[k * m + (j+1)]);
                
//                 sum00 = _mm_add_pd(sum00, _mm_mul_pd(a0_vec, b0_vec));
//                 sum01 = _mm_add_pd(sum01, _mm_mul_pd(a0_vec, b1_vec));
//                 sum10 = _mm_add_pd(sum10, _mm_mul_pd(a1_vec, b0_vec));
//                 sum11 = _mm_add_pd(sum11, _mm_mul_pd(a1_vec, b1_vec));
//             }
            
//             // Горизонтальное сложение для получения скалярных значений
//             double sum00_arr[2], sum01_arr[2], sum10_arr[2], sum11_arr[2];
//             _mm_storeu_pd(sum00_arr, sum00);
//             _mm_storeu_pd(sum01_arr, sum01);
//             _mm_storeu_pd(sum10_arr, sum10);
//             _mm_storeu_pd(sum11_arr, sum11);
            
//             double total00 = sum00_arr[0] + sum00_arr[1];
//             double total01 = sum01_arr[0] + sum01_arr[1];
//             double total10 = sum10_arr[0] + sum10_arr[1];
//             double total11 = sum11_arr[0] + sum11_arr[1];
            
//             // Обработка оставшегося одиночного элемента (если ah нечетное)
//             if (k < ah) {
//                 total00 += a[i * m + k] * b[k * m + j];
//                 total01 += a[i * m + k] * b[k * m + (j+1)];
//                 total10 += a[(i+1) * m + k] * b[k * m + j];
//                 total11 += a[(i+1) * m + k] * b[k * m + (j+1)];
//             }
            
//             // Запись результатов
//             res[i * m + j] += total00;
//             res[i * m + (j+1)] += total01;
//             res[(i+1) * m + j] += total10;
//             res[(i+1) * m + (j+1)] += total11;
//         }
        
//         // Обработка оставшегося столбца (если h нечетное)
//         if (j < h) {
//             for (int i2 = i; i2 < i + 2 && i2 < v; i2++) {
//                 double sum = 0.0;
//                 for (k = 0; k < ah; k++) {
//                     sum += a[i2 * m + k] * b[k * m + j];
//                 }
//                 res[i2 * m + j] += sum;
//             }
//         }
//     }
    
//     // Обработка оставшейся строки (если v нечетное)
//     if (i < v) {
//         for (j = 0; j < h; j++) {
//             double sum = 0.0;
//             for (k = 0; k < ah; k++) {
//                 sum += a[i * m + k] * b[k * m + j];
//             }
//             res[i * m + j] += sum;
//         }
//     }
// }

// void multMatr(double *a, double *b, double *c,
//                 int rowNum1, int colNum1, int colNum2, int m)
// {
//     mult_sse(a, b, c, rowNum1, colNum1, colNum2, m);
// }
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
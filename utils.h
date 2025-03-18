#ifndef UTILS_H
#define UTILS_H

const double EPS = 1e-14;

// Utility functions
int min(int a, int b);
double min(double a, double b);
double max(double a, double b);

// Block and process related functions
int local_to_global(int m, int p, int k, int j_loc);
int global_to_local(int m, int p, int j_glob);
int numOfBlockColsInProc(int n, int m, int p, int k);
int numOfColsInProc(int n, int m, int p, int k);
int getNumOfProc(int m, int p, int j_glob);

#endif // UTILS_H 
/**
* The sequential.h & sequential.cu are composed of host methods which are the sequential version of the methods in kernel.cu.
* More details about the method arguments and how the results are returned to the CPU will be stated in kernel.cu.
*/
#ifndef SEQUENTIAL_H
#define SEQUENTIAL_H

#include<vector>

 using namespace std;

void s_matrixAddition(vector<float> &A, vector<float> &B, vector<float> &C, int N);

void s_matrixSubtraction(vector<float> &A, vector<float> &B, vector<float> &C, int N);

void s_scalarMultiplication(vector<float> &A, float scalar, vector<float> &C, int N);

void s_matrixMultiplication(vector<float> &A, vector<float> &B, vector<float> &C, int N);

void s_matrixTransposition(vector<float> &A, vector<float> &C, int N);

//void s_LU_base(float* A, int N);

void s_LU_right_looking(vector<float> &A, int N);

void s_LU_left_looking(vector<float> &A, int N);

void s_LU_onepass(vector<float> &A, int N);

//void s_LU_crout(float*A, int N);

float s_getMatrixDeterminant(vector<float> &U,int N);

void s_upperTriangleInversion(vector<float> &U,int N);

void s_matrixInversion(vector<float> &C, vector<float> &L, vector<float> &U,int N);

void s_solveUpperTriangleEquations(vector<float> &U,vector<float> &x,vector<float> &y,int N);

void s_solveLowerTriangleEquations(vector<float> &L,vector<float> &y,vector<float> &b,int N);

void s_solveEquations(vector<float> &L,vector<float> &U,vector<float> &x,vector<float> &b,int N);

void split_LU(vector<float> &A, vector<float> &L, vector<float> &U, int N);

#endif
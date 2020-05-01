/**
* The kernel.h & kernel.cu are composed of host methods which contain the corresponding kernel configuration and kernel method calling statements.
* For example, in the matrixAddition method, we'll set the grideSize and blockSize and call the global kernel method for matrix addition operation.
* More details about the method arguments and how the results are returned to the CPU will be stated in kernel.cu.
*/

#ifndef _KERNEL_H_ 
#define _KERNEL_H_

void matrixAddition(float* A, float* B, float* C, int N);

void matrixSubtraction(float* A, float* B, float* C, int N);

void scalarMultiplication(float* A, float scalar, float* C, int N);

void matrixMultiplication(float* A, float* B, float* C, int N);

void matrixTransposition(float* A, float* C, int N);

void LU_base(float* A, int N);

void LU_right_looking(float*A, int N);

void LU_left_looking(float*A, int N);

void LU_onepass(float*A, int N);

void LU_crout(float*A, int N);

float getMatrixDeterminant(float* U,int N);

void upperTriangleInversion(float *U,int N);

void matrixInversion(float* C, float* L, float* U,int N);

void solveUpperTriangleEquations(float* U,float* x,float *y,int N);

void solveLowerTriangleEquations(float* L,float* y,float *b,int N);

void solveEquations(float* L,float* U,float* x,float *b,int N);

#endif
/**
* sequential.cu: the implementations of the sequential version of the methods in kernel.cu (which could be run on CPU and used to be compared with the GPU version methods)
*/

#include "sequential.h"
#include<iostream>
#include<vector>

using namespace std;

/**
* s_matrixAddition
*
* @param  A   the first Matrix (N*N)
*         B   the second Matrix (N*N)
*		  C   the result Matrix (N*N)
*         N   the side of the array
*/
void s_matrixAddition(vector<float>& A, vector<float>& B, vector<float>& C, int N){
	for (int ROW = 0; ROW < N; ROW++) {
		for (int COL = 0; COL < N; COL++){
			C[ROW * N + COL]=A[ROW * N + COL]+B[ROW * N + COL];
		}
	}
}

/**
* s_matrixSubtraction
*
* @param  A   the first Matrix (N*N)
*         B   the second Matrix (N*N)
*		  C   the result Matrix (N*N)
*         N   the side of the array
*/
void s_matrixSubtraction(vector<float>& A, vector<float>& B, vector<float>& C, int N){
	for (int ROW = 0; ROW < N; ROW++) {
		for (int COL = 0; COL < N; COL++){
			C[ROW * N + COL]=A[ROW * N + COL]-B[ROW * N + COL];
		}
	}
}

/**
* s_scalarMultiplication
*
* @param  A   the first Matrix (N*N)
*         scalar   the multiplier
*		  C   the result Matrix (N*N)
*         N   the side of the array
*/
void s_scalarMultiplication(vector<float>& A, float scalar, vector<float>& C, int N){
	for (int ROW = 0; ROW < N; ROW++) {
		for (int COL = 0; COL < N; COL++){
			C[ROW * N + COL]=A[ROW * N + COL]*scalar;
		}
	}
}

/**
* s_matrixMultiplication
*
* @param  A   the first Matrix (N*N)
*         B   the second Matrix (N*N)
*		  C   the result Matrix (N*N)
*         N   the side of the array
*/
void s_matrixMultiplication(vector<float>& A, vector<float>& B, vector<float>& C, int N){
	float sum;
	for (int row = 0; row < N; row++) {
		for (int col = 0; col < N; col++) {
			sum = 0.f;
			for (int n = 0; n < N; n++) {
				sum += A[row * N + n] * B[n * N + col];
			}
			C[row * N + col] = sum;
		}
	}
}

/**
* s_matrixTransposition
*
* @param  A   the given Matrix (N*N)
*		  C   the result Matrix (N*N)
*         N   the side of the array
*/
void s_matrixTransposition(vector<float>& A, vector<float>& C, int N){
	for (int ROW = 0; ROW < N; ROW++) {
		for (int COL = 0; COL < N; COL++){
			C[ROW * N + COL]=A[COL * N + ROW];
		}
	}
}

/**
* s_LU_right_looking
*
* @param  A   the given Matrix (N*N)
*         N   the side of the array
*/
void s_LU_right_looking(vector<float>&A, int N){
	for(int i = 0; i < N; i++){
		for (int j = i+1; j < N; j++) {
		//update L
		A[j * N + i] = A[j * N + i] / A[i * N + i]; 
		for (int k = i + 1; k < N; k++) { // iterates over the remaining columns
			A[j * N + k] -= A[j * N + i] * A[i * N + k]; // Updates U and L 
			}
		}
	}
}

/**
* s_LU_left_looking
*
* @param  A   the given Matrix (N*N)
*         N   the side of the array
*/
void s_LU_left_looking(vector<float>&A, int N){
	
	for(int i = 0; i < N; i++){
		for(int j = 0; j < i; j++){
			for(int k=j+1;k<N;k++)
				A[k * N + i] -= A[k * N + j] * A[j * N + i]; 
		}
		for(int k = i + 1; k < N; k++){
			A[k * N + i] /= A[i * N+ i ];
		}
	}
	
}

/**
* s_LU_onepass
*
* @param  A   the given Matrix (N*N)
*         N   the side of the array
*/
void s_LU_onepass(vector<float>&A, int N){
	for(int i = 0; i < N; i++){
		for (int j = i+1; j < N; j++) {
			for(int k = i; k < N; k++){
				if(i==k){A[j * N + i] = A[j * N + i] / A[i * N + i];}
				else{
				A[j * N + k] -= A[j * N + i] * A[i * N + k] /A[i*N+i];
				}
			}
		}
	}
}

/**
* s_getMatrixDeterminant
*
* @param  U   the upper triangular matrix (N*N)
*         N   the side of the array
* @return     the determinant value
*/
float s_getMatrixDeterminant(vector<float>& U,int N){
	float det=1;
	for(int row=0;row<N;row++){
		det*=U[row*N+row];
	}
	return det;
}

/**
* s_upperTriangleInversion
*
* @param  U   the upper triangular matrix (N*N)
*         N   the side of the array
*/
void s_upperTriangleInversion(vector<float>& U,int N){
	vector<float> prev_U(U);
	for(int dert=0;dert<N;dert++){
		for(int row=0;row+dert<N;row++){
			int col = row+dert;
			if(dert==0){
				U[row*N+col]=1/prev_U[row*N+col];
			}
			else{
				float sum=0;
				for(int k=row+1;k<=col;k++){
					sum+=U[k*N+col]*prev_U[row*N+k];
				}
				U[row*N+col]=-sum/prev_U[row*N+row];
			}
		}
	}
}

/**
* s_matrixInversion
*
* @param  C   the result matrix (N*N)
*         L   the lower triangular matrix (N*N)
*         U   the upper triangular matrix (N*N)
*         N   the side of the array
*/
void s_matrixInversion(vector<float>& C, vector<float>& L, vector<float>& U,int N){
	vector<float> trans_L(N*N);
	s_upperTriangleInversion(U,N);
	s_matrixTransposition(L,trans_L,N);
	s_upperTriangleInversion(trans_L,N);
	s_matrixTransposition(trans_L,L,N);
	s_matrixMultiplication(U,L,C,N);
}

/**
* s_solveUpperTriangleEquations
*
* @param  U   the upper triangular matrix (N*N)
*		  x   the solution vector for "Ux=y"
*         y   the right part of the equation
*         N   the side of the array
*/
void s_solveUpperTriangleEquations(vector<float>& U,vector<float>& x,vector<float>& y,int N){
	for(int row=N=1;row<N>=0;row--){
		int cur_x = y[row];
		for(int i=N-1;i>row;i--){
			y[row]-=x[i]*U[row*N+i];
		}
		x[row]=y[row]/U[row*N+row];
	}
}

/**
* s_solveLowerTriangleEquations
*
* @param  L   the lower triangular matrix (N*N
*		  y   the solution vector for "Ly=b"
*         b   the right part of the equation
*         N   the side of the array
*/
void s_solveLowerTriangleEquations(vector<float>& L,vector<float>& y,vector<float>& b,int N){
	for(int row=0;row<N;row++){
		int cur_x = b[row];
		for(int i=0;i<row;i++){
			b[row]-=y[i]*L[row*N+i];
		}
		y[row]=b[row]/L[row*N+row];
	}
}

/**
* solveEquations
*
* @param  L   the lower triangular matrix (N*N
*         U   the upper triangular matrix (N*N)
*		  x   the solution vector for "Ax=b"
*         b   the right part of the equation
*         N   the side of the array
*/
void s_solveEquations(vector<float>& L,vector<float>& U,vector<float>& x,vector<float>& b,int N){
	vector<float> y(N);
	s_solveLowerTriangleEquations(L,y,b,N);
	s_solveUpperTriangleEquations(U,x,y,N);
}

/**
* split_LU
*
* @param  A   the matrix to be split (which store the values of L&U)
*         L   the lower triangular matrix (N*N)
*		  U   the upper triangular matrix (N*N)
*         N   the side of the array
*/
void split_LU(vector<float>&A, vector<float>&L, vector<float>&U, int N) {
	for (int i = 0; i < N; i++){
		for (int j = 0; j < N; j++){
			if (i < j){
				U[i*N+j] = A[i*N+j];
				L[i*N+j] = 0;
			}
			if (i > j) {
				L[i*N+j] = A[i*N+j];
				U[i*N+j] = 0;
			}
			if (i == j) {
				L[i*N+j] = 1;
				U[i*N+j] = A[i*N+j];
			}
		}
	}
}
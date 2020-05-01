/**
* kernel.cu: the implementations for the methods stated in kernel.h. Each operation has one (or more than one) host method(s) to call the kernel and one 
* (or more than one) global kernel method(s) to run the parallel algorithm on GPU.
*/

#include <math.h>
#include <iostream>
#include "cuda_runtime.h"
#include "kernel.h"
#include "dev_array.h"
#include <stdlib.h>

//reference: https://www.quantstart.com/articles/Matrix-Matrix-Multiplication-on-the-GPU-with-Nvidia-CUDA/

using namespace std;

/**
* Do the atomic multiplication based on the atomicCAS given by CUDA
*
* @param  address   the address where the target variable is at
*         val   the multiplier
* @return      the old value before being updated
*/
__device__ float atomicMul(float* address, float val)
{
    int* address_as_i = (int*)address;
    int old = *address_as_i, assumed;

    do {
        assumed = old;
        old = atomicCAS(address_as_i, assumed,
                        __float_as_int(val *
                               __int_as_float(assumed)));

    // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
    } while (assumed != old);

    return __int_as_float(old);
}

/**
* matrixAdditionKernel
*
* @param  A   the first Matrix (N*N)
*         B   the second Matrix (N*N)
*		  C   the result Matrix (N*N)
*         N   the side of the array
*/
__global__ void matrixAdditionKernel(float* A, float* B, float* C, int N) {

	int ROW = blockIdx.y * blockDim.y + threadIdx.y;
	int COL = blockIdx.x * blockDim.x + threadIdx.x;

	//there may be some redundant threads which won't be assigned any task
	if (ROW < N && COL < N) {
		// each thread computes one element of the block sub-matrix
		C[ROW * N + COL] = A[ROW * N + COL]+B[ROW * N + COL];
		//prfloatf("C[%d]==A[%d]+B[%d],%d",ROW * N + COL,ROW * N + COL,ROW * N + COL,C[ROW * N + COL] );
	}
}

/**
* matrixSubtractionKernel
*
* @param  A   the first Matrix (N*N)
*         B   the second Matrix (N*N)
*		  C   the result Matrix (N*N)
*         N   the side of the array
*/
__global__ void matrixSubtractionKernel(float* A, float* B, float* C, int N) {

	int ROW = blockIdx.y * blockDim.y + threadIdx.y;
	int COL = blockIdx.x * blockDim.x + threadIdx.x;

	//there may be some redundant threads which won't be assigned any task
	if (ROW < N && COL < N) {
		// each thread computes one element of the block sub-matrix
		C[ROW * N + COL] = A[ROW * N + COL] - B[ROW * N + COL];
	}
}

/**
* scalarMultiplicationKernel
*
* @param  A   the first Matrix (N*N)
*         scalar   the multiplier
*		  C   the result Matrix (N*N)
*         N   the side of the array
*/
__global__ void scalarMultiplicationKernel(float* A, float scalar, float* C, int N) {

	int ROW = blockIdx.y * blockDim.y + threadIdx.y;
	int COL = blockIdx.x * blockDim.x + threadIdx.x;

	//there may be some redundant threads which won't be assigned any task
	if (ROW < N && COL < N) {
		// each thread computes one element of the block sub-matrix
		C[ROW * N + COL] = A[ROW * N + COL] * scalar;
	}
}

/**
* matrixMultiplicationKernel
*
* @param  A   the first Matrix (N*N)
*         B   the second Matrix (N*N)
*		  C   the result Matrix (N*N)
*         N   the side of the array
*/
__global__ void matrixMultiplicationKernel(float* A, float* B, float* C, int N) {

	extern __shared__ float local[];

	if(threadIdx.x==0 && threadIdx.y==0){
		for(int i=0;i<N;i++){
			for(int j=0;j<N;j++){
				local[i*N+j]=A[i*N+j];
			}
		}
		for(int i=N;i<N*2;i++){
			for(int j=0;j<N;j++){
				local[i*N+j]=B[(i-N)*N+j];
			}
		}
	}
	__syncthreads();

	int ROW = blockIdx.y * blockDim.y + threadIdx.y;
	int COL = blockIdx.x * blockDim.x + threadIdx.x;

	float tmpSum = 0;

	//reduce: to be updated (limited by the total number of threads that can run concurrently, we didn't implement reduce method here.)

	//there may be some redundant threads which won't be assigned any task
	if (ROW < N && COL < N) {
		// each thread computes one element of the block sub-matrix
		for (int i = 0; i < N; i++) {
			tmpSum += local[ROW * N + i] * local[(i+N) * N + COL];
		}
	}
	C[ROW * N + COL] = tmpSum;
}

/**
* matrixTranspositionKernel
*
* @param  A   the given Matrix (N*N)
*		  C   the result Matrix (N*N)
*         N   the side of the array
*/
__global__ void matrixTranspositionKernel(float* A, float* C, int N) {

	int ROW = blockIdx.y * blockDim.y + threadIdx.y;
	int COL = blockIdx.x * blockDim.x + threadIdx.x;

	//there may be some redundant threads which won't be assigned any task
	if (ROW < N && COL < N) {
		// each thread computes one element of the block sub-matrix
		C[COL * N + ROW] = A[ROW * N + COL];
	}
}

/**
* decompose_multipliers
*
* @param  A   the given Matrix (N*N)
*		  rows_per_thread   the number of threads each thread takes care of
*         i	  the iterator for the outer loop (in the caller for this method)
*         N   the side of the array
*/
__global__ void decompose_multipliers(float *A, int rows_per_thread, int i, int N) {

	extern __shared__ float local[];

	if(threadIdx.x==0){
		local[0]=A[i * N + i];
	}

	__syncthreads();

	float tid = blockIdx.x * blockDim.x + threadIdx.x;

	int jstart = (i + 1) + tid * rows_per_thread;
	int jend = jstart + rows_per_thread;

	for (int j = jstart; j < jend && j < N; j++) {
		A[j * N + i] = A[j * N + i] / local[0]; // Computes the multipliers and updates L in A
		//printf("new L in A[%d][%d] is %d\n", j, i, A[j*N+i]);
		//printf("A[%d][%d] is %d\n",i,i,A[i*N+i]);
	}
}

/**
* decompose_elimination
*
* @param  A   the given Matrix (N*N)
*		  rows_per_thread   the number of threads each thread takes care of
*         i	  the iterator for the outer loop (in the caller for this method)
*         N   the side of the array
*/
__global__ void decompose_elimination(float *A, int rows_per_thread, int i, int N) {

	extern __shared__ float local[];

	if(threadIdx.x==0){
		for(int iteration=0;iteration<N;iteration++){
			local[0*N+iteration]=A[i*N+iteration];
		}
		for(int iteration=0;iteration<N;iteration++){
			local[1*N+iteration]=A[iteration*N+i];
		}
	}

	__syncthreads();

	float tid = blockIdx.x * blockDim.x + threadIdx.x;
	float eid = blockIdx.y * blockDim.y + threadIdx.y;

	int jstart = (i + 1) + tid * rows_per_thread;
	int jend = jstart + rows_per_thread;

	int kstart = (i + 1) + eid * rows_per_thread;
	int kend = kstart + rows_per_thread;

	for (int j = jstart; j < jend && j < N; j++) { // Iterates over the remaining rows
		for (int k = kstart; k < kend && k < N; k++) { // iterates over the remaining columns
			A[j * N + k] -= local[1 * N + j] * local[0 * N +k ]; // Updates U and L 
			//printf("new L and U in A[%d][%d] is %d\n", j, i, A[j*N+i]);
		}
	}

}

/**
* decompose_right_looking
*
* @param  A   the given Matrix (N*N)
*		  rows_per_thread   the number of threads each thread takes care of
*         i	  the iterator for the outer loop (in the caller for this method)
*         N   the side of the array
*/
__global__ void decompose_right_looking(float *A, int rows_per_thread, int i, int N){

	float tid = blockIdx.x * blockDim.x + threadIdx.x;
	float eid = blockIdx.y * blockDim.y + threadIdx.y;

	int jstart = (i + 1) + tid * rows_per_thread;
	int jend = jstart + rows_per_thread;
	
	//int k = (i + 1) + eid;
	//int kend = kstart + rows_per_thread;


	for (int j = jstart; j < jend && j < N; j++) {
		//update L
		A[j * N + i] = A[j * N + i] / A[i * N + i]; 
		for(int k = i+1; k < N; k++){
	    // iterates over the remaining columns
		A[j * N + k] -= A[j * N + i] * A[i * N + k]; // Updates U and L 
	    }
	}

}

/**
* decompose_left_looking
*
* @param  A   the given Matrix (N*N)
*		  rows_per_thread   the number of threads each thread takes care of
*         i	  the iterator for the outer loop (in the caller for this method)
*         N   the side of the array
*/
__global__ void decompose_left_looking(float *A, int rows_per_thread, int i, int N){

	float tid = blockIdx.x * blockDim.x + threadIdx.x;

	//int jstart_0 = 0 + tid * rows_per_thread;
	int jstart = (i+1) + tid * rows_per_thread;
	//int jend_0 = jstart_0 + rows_per_thread;
	int jend = jstart + rows_per_thread;


	for (int j = 0; j < i; j++) {
		//update L
		//A[j * N + i] = A[j * N + i] / A[i * N + i]; 
		for (int k = j + 1; k < N; k++) { // iterates over the remaining columns
			A[k * N + i] -= A[k * N + j] * A[j * N + i]; // Updates U and L 
		}
	}
	//A[i * N + i] = 1/A[i * N + i];
	for(int j=jstart; j < jend && j<N; j++){
		A[j * N + i] = A[j * N + i] / A[i * N + i];
	}
}

/**
* decompose_onepass
*
* @param  A   the given Matrix (N*N)
*		  rows_per_thread   the number of threads each thread takes care of
*         i	  the iterator for the outer loop (in the caller for this method)
*         N   the side of the array
*/
__global__ void decompose_onepass(float *A, int rows_per_thread, int i, int N){

	extern __shared__ float local[];

	if(threadIdx.x==0){
		for(int iteration=0;iteration<N;iteration++){
			local[0*N+iteration]=A[i*N+iteration];
		}
		for(int iteration=0;iteration<N;iteration++){
			local[1*N+iteration]=A[iteration*N+i];
		}
	}

	__syncthreads();

	float tid = blockIdx.x * blockDim.x + threadIdx.x;
	float eid = blockIdx.y * blockDim.y + threadIdx.y;

	int jstart = (i + 1) + tid * rows_per_thread;
	int jend = jstart + rows_per_thread;

	int kstart = i  + eid * rows_per_thread;
	int kend = kstart + rows_per_thread;


	for (int j = jstart; j < jend && j < N; j++) {
		for (int k =i;k < N; k++) {// iterates over the remaining columns
			if(i == k){//update L
				A[j * N + i] = A[j * N + i] / local[0*N+i];
			}
			else{
				A[j * N + k] -= local[1 * N + j] * local[0 * N +k ]/local[0*N+i]; // Updates U and L 
			}
		}
	}

}

/**
* getMatrixDeterminantKernel
*
* @param  U   the upper triangular matrix (N*N)
*		  determ   the determinant to be calculated (initialized as 1)
*         N   the side of the array
*/
__global__ void getMatrixDeterminantKernel(float*U, float* determ,int N){
	int ROW = blockIdx.x * blockDim.x + threadIdx.x;
	//printf("cur det is %f\n",*determ);

	if(ROW< N){
		atomicMul(determ,U[ROW*N+ROW]);
		//printf("cur det is %f, times %f\n",*determ,U[ROW*N+ROW]);
	}
}

__global__ void pre_crout(float* A, int N) {

	int COL = blockIdx.x * blockDim.x + threadIdx.x;

	if(COL==0) A[0*N+0] = 1/A[0*N+0];
	__syncthreads();
	//there may be some redundant threads which won't be assigned any task
	if (COL < N && COL > 1) {
		// each thread computes one element of the block sub-matrix
		A[0*N + COL] = A[0*N + COL] * A[0*N+0];
	}
}

/**
* decompose_crout (deprecated)
*
* @param  A   the given Matrix (N*N)
*		  rows_per_thread   the number of threads each thread takes care of
*         i	  the iterator for the outer loop (in the caller for this method)
*         N   the side of the array
*/
__global__ void decompose_crout(float *A, int rows_per_thread, int i, int N){

	int tid = blockIdx.x * blockDim.x + threadIdx.x;

	int jstart = (i + 1) + tid * rows_per_thread;
	int jend = jstart + rows_per_thread;
    int jstart_0 = 0 + tid * rows_per_thread;
    int jend_0 = jstart_0 + rows_per_thread;

	for (int j = jstart_0; j < jend_0 && j < i; j++) {
		for (int k = i; k < N; k++) {// iterates over the remaining columns
				A[k * N + i] -= A[k * N + j] * A[j * N + i];
			}
	}
	
	for(int k = 0; k < i; k++){
		for(int j = jstart;j < jend && j < N; j++){
			A[i * N + j] -= A[i * N + k] * A[k * N + j]; 
		}
	}
	for(int k = 0;k < i; k++){
			A[i * N + k] /= A[i * N + i];
		}

}


/**
* upperTriangleInversionKernel
*
* @param  U   the upper triangular matrix (N*N)
*		  prev_U   the original version of matrix U (N*N)
*         N   the side of the array
*/
__global__ void upperTriangleInversionKernel (float* U, float* prev_U,int N){
	extern __shared__ float local[];

	if(threadIdx.x==0 && threadIdx.y==0){
		for(int i=0;i<N;i++){
			for(int j=0;j<N;j++){
				local[i*N+j]=U[i*N+j];
			}
		}
		for(int i=N;i<N*2;i++){
			for(int j=0;j<N;j++){
				local[i*N+j]=prev_U[(i-N)*N+j];
			}
		}
	}
	__syncthreads();
	for(int dert=0;dert<N;dert++){
		int row = blockIdx.x * blockDim.x + threadIdx.x;
		int col = row+dert;
		if(dert==0){
			U[row*N+col]=1/local[(row+N)*N+col];
			local[row*N+col]=U[row*N+col];
		}
		else{
			if(row+dert<N){
				float sum=0;
				for(int k=row+1;k<=col;k++){
					sum+=local[k*N+col]*local[(row+N)*N+k];
				}
				float update_val;
				update_val=-sum/local[(row+N)*N+row];
				U[row*N+col]=update_val;
				local[row*N+col]=update_val;
			}
		}
		__syncthreads();
	}
}

/**
* solveUpperTriangleEquationsKernel
*
* @param  U   the upper triangular matrix (N*N
*		  x   the solution vector for "Ux=y"
*         y   the right part of the equation
*         N   the side of the array
*/
__global__ void solveUpperTriangleEquationsKernel(float* U,float* x,float *y,int N){
	extern __shared__ float local_x[];
	local_x[threadIdx.x]=x[threadIdx.x];
	__syncthreads();
	for(int row=N-1;row>=0;row--){
		if(threadIdx.x>row){
			atomicAdd(&y[row],-local_x[threadIdx.x]*U[row*N+threadIdx.x]);
			//printf("current_x is %f\n",y[row]);
		}
		__syncthreads();
		if(threadIdx.x==N-1){
			float update_val=y[row]/U[row*N+row];
			x[row]=update_val;
			local_x[row]=update_val;
			//printf("x[%d]is %f\n",row,x[row]);
		}
		__syncthreads();
	}
}

/**
* solveLowerTriangleEquationsKernel
*
* @param  L   the lower triangular matrix (N*N
*		  y   the solution vector for "Ly=b"
*         b   the right part of the equation
*         N   the side of the array
*/
__global__ void solveLowerTriangleEquationsKernel(float* L,float* y,float *b,int N){
	extern __shared__ float local_y[];
	local_y[threadIdx.x]=y[threadIdx.x];
	__syncthreads();
	for(int row=0;row<N;row++){
		if(threadIdx.x<row){
			atomicAdd(&b[row],-local_y[threadIdx.x]*L[row*N+threadIdx.x]);
			//printf("current_y is %f\n",b[row]);
		}
		__syncthreads();
		if(threadIdx.x==0){
			float update_val=b[row]/L[row*N+row];
			y[row]=update_val;
			local_y[row]=update_val;
			//printf("y[%d]is %f\n",row,y[row]);
		}
		__syncthreads();
	}
}

/**
* matrixAddition
*
* @param  A   the first Matrix (N*N)
*         B   the second Matrix (N*N)
*		  C   the result Matrix (N*N)
*         N   the side of the array
*/
void matrixAddition(float* A, float* B, float* C, int N) {
	// declare the number of blocks per grid and the number of threads per block
	dim3 threadsPerBlock(N, N);
	dim3 blocksPerGrid(1, 1);

	if (N * N > 32 * 32) {
		threadsPerBlock.x = 32;
		threadsPerBlock.y = 32;
		blocksPerGrid.x = ceil(double(N) / double(threadsPerBlock.x));
		blocksPerGrid.y = ceil(double(N) / double(threadsPerBlock.y));
	}

	matrixAdditionKernel << <blocksPerGrid, threadsPerBlock >> > (A, B, C, N);
}

/**
* matrixSubtraction
*
* @param  A   the first Matrix (N*N)
*         B   the second Matrix (N*N)
*		  C   the result Matrix (N*N)
*         N   the side of the array
*/
void matrixSubtraction(float* A, float* B, float* C, int N) {
	// declare the number of blocks per grid and the number of threads per block
	dim3 threadsPerBlock(N, N);
	dim3 blocksPerGrid(1, 1);

	if (N * N > 32 * 32) {
		threadsPerBlock.x = 32;
		threadsPerBlock.y = 32;
		blocksPerGrid.x = ceil(double(N) / double(threadsPerBlock.x));
		blocksPerGrid.y = ceil(double(N) / double(threadsPerBlock.y));
	}

	matrixSubtractionKernel << <blocksPerGrid, threadsPerBlock >> > (A, B, C, N);
}

/**
* scalarMultiplication
*
* @param  A   the first Matrix (N*N)
*         scalar   the multiplier
*		  C   the result Matrix (N*N)
*         N   the side of the array
*/
void scalarMultiplication(float* A, float scalar, float* C, int N) {
	// declare the number of blocks per grid and the number of threads per block
	dim3 threadsPerBlock(N, N);
	dim3 blocksPerGrid(1, 1);

	if (N * N > 32 * 32) {
		threadsPerBlock.x = 32;
		threadsPerBlock.y = 32;
		blocksPerGrid.x = ceil(double(N) / double(threadsPerBlock.x));
		blocksPerGrid.y = ceil(double(N) / double(threadsPerBlock.y));
	}

	scalarMultiplicationKernel << <blocksPerGrid, threadsPerBlock >> > (A, scalar, C, N);
}

/**
* matrixMultiplication
*
* @param  A   the first Matrix (N*N)
*         B   the second Matrix (N*N)
*		  C   the result Matrix (N*N)
*         N   the side of the array
*/
void matrixMultiplication(float* A, float* B, float* C, int N) {

	// declare the number of blocks per grid and the number of threads per block
	dim3 threadsPerBlock(N, N);
	dim3 blocksPerGrid(1, 1);

	if (N * N > 32*32) {
		threadsPerBlock.x = 32;
		threadsPerBlock.y = 32;
		blocksPerGrid.x = ceil(double(N) / double(threadsPerBlock.x));
		blocksPerGrid.y = ceil(double(N) / double(threadsPerBlock.y));
	}

	matrixMultiplicationKernel << <blocksPerGrid, threadsPerBlock,N*N*2*sizeof(float)>> > (A, B, C, N);
}

/**
* matrixTransposition
*
* @param  A   the given Matrix (N*N)
*		  C   the result Matrix (N*N)
*         N   the side of the array
*/
void matrixTransposition(float* A, float* C, int N) {
	// declare the number of blocks per grid and the number of threads per block
	// use 1 to 512 threads per block
	dim3 threadsPerBlock(N, N);
	dim3 blocksPerGrid(1, 1);

	if (N * N > 32 * 32) {
		threadsPerBlock.x = 32;
		threadsPerBlock.y = 32;
		blocksPerGrid.x = ceil(double(N) / double(threadsPerBlock.x));
		blocksPerGrid.y = ceil(double(N) / double(threadsPerBlock.y));
	}

	matrixTranspositionKernel << <blocksPerGrid, threadsPerBlock >> > (A, C, N);
}

/**
* LU_base
*
* @param  A   the given Matrix (N*N)
*         N   the side of the array
*/
void LU_base(float* A, int N) {
	dim3 threadsPerBlock(N);
    dim3 blocksPerGrid(1);

	dim3 threadsPerBlockU(N,N);
	dim3 blocksPerGridU(1,1);

	if (N > 32 * 32){
		threadsPerBlock.x=32*32;

		blocksPerGrid.x=ceil(double(N) / double(threadsPerBlock.x));
	}

	if (N * N > 32 * 32) {
		threadsPerBlockU.x = 32;
		threadsPerBlockU.y = 32;
		blocksPerGridU.x = ceil(double(N) / double(threadsPerBlockU.x));
		blocksPerGridU.y = ceil(double(N) / double(threadsPerBlockU.y));
	}


	float ops_per_thread = ceil(double(N) / (double)(threadsPerBlock.x*blocksPerGrid.x));

	
	for (int i = 0; i < N; i++) { // Iterates over the columns to remove
		decompose_multipliers << <blocksPerGrid, threadsPerBlock,sizeof(float)>> > (A, ops_per_thread, i, N);
		decompose_elimination << <blocksPerGridU, threadsPerBlockU,2*N*sizeof(float)>> > (A, ops_per_thread, i, N);
	}
}

/**
* LU_right_looking
*
* @param  A   the given Matrix (N*N)
*         N   the side of the array
*/
void LU_right_looking(float*A, int N){
	dim3 threadsPerBlock(N);
    dim3 blocksPerGrid(1);

	if (N > 32 * 32){
		threadsPerBlock.x=32*32;
		blocksPerGrid.x=ceil(double(N) / double(threadsPerBlock.x));
	}

	float ops_per_thread = ceil(double(N) / (double)(threadsPerBlock.x*blocksPerGrid.x));

	
	for (int i = 0; i < N; i++) { // Iterates over the columns to remove
		decompose_right_looking << <blocksPerGrid, threadsPerBlock >> > (A, ops_per_thread, i, N);
	}
}

/**
* LU_left_looking
*
* @param  A   the given Matrix (N*N)
*         N   the side of the array
*/
void LU_left_looking(float*A, int N){
	dim3 threadsPerBlock(N);
    dim3 blocksPerGrid(1);

	if (N > 32 * 32){
		threadsPerBlock.x=32*32;
		blocksPerGrid.x=ceil(double(N) / double(threadsPerBlock.x));
	}

	float ops_per_thread = ceil(double(N) / (double)(threadsPerBlock.x*blocksPerGrid.x));

	
	for (int i = 0; i < N; i++) { // Iterates over the columns to remove
		decompose_left_looking << <blocksPerGrid, threadsPerBlock >> > (A, ops_per_thread, i, N);
	}
}

/**
* LU_onepass
*
* @param  A   the given Matrix (N*N)
*         N   the side of the array
*/
void LU_onepass(float*A, int N){
	dim3 threadsPerBlock(N);
    dim3 blocksPerGrid(1);

	if (N > 32 * 32){
		threadsPerBlock.x=32*32;
		blocksPerGrid.x=ceil(double(N) / double(threadsPerBlock.x));
	}

	float ops_per_thread = ceil(double(N) / (double)(threadsPerBlock.x*blocksPerGrid.x));

	
	for (int i = 0; i < N; i++) { // Iterates over the columns to remove
		decompose_onepass << <blocksPerGrid, threadsPerBlock,2*N*sizeof(float)>> > (A, ops_per_thread, i, N);
	}
}

/**
* LU_crout (deprecated)
*
* @param  A   the given Matrix (N*N)
*         N   the side of the array
*/
void LU_crout(float*A, int N){
	dim3 threadsPerBlock(N);
    dim3 blocksPerGrid(1);

	if (N > 32 * 32){
		threadsPerBlock.x=32*32;
		blocksPerGrid.x=ceil(double(N) / double(threadsPerBlock.x));
	}

	float ops_per_thread = ceil(double(N) / (double)(threadsPerBlock.x*blocksPerGrid.x));

	//pre_crout <<<blocksPerGrid, threadsPerBlock >> > (A, N);
	for (int i = 0; i < N; i++) { // Iterates over the columns to remove
		decompose_crout << <blocksPerGrid, threadsPerBlock >> > (A, ops_per_thread, i, N);
	}
}

/**
* getMatrixDeterminant
*
* @param  U   the upper triangular matrix (N*N)
*         N   the side of the array
* @return     the determinant value
*/
float getMatrixDeterminant(float* U,int N) {
	dim3 threadsPerBlock(N);
    dim3 blocksPerGrid(1);

	float* ans=(float*)malloc(sizeof(float));
	*ans=1;
	float* d_ans;

	cudaMalloc((void**)&d_ans, sizeof(float));
	cudaMemcpy(d_ans, ans, sizeof(float), cudaMemcpyHostToDevice);  
	
	getMatrixDeterminantKernel<<<blocksPerGrid, threadsPerBlock>>>(U,d_ans,N);

	cudaDeviceSynchronize();

	cudaMemcpy(ans, d_ans, sizeof(float), cudaMemcpyDeviceToHost);
	
	return *ans;
}

/**
* upperTriangleInversion
*
* @param  U   the upper triangular matrix (N*N)
*         N   the side of the array
*/
void upperTriangleInversion(float *U,int N){
	dim3 threadsPerBlock(N);
    dim3 blocksPerGrid(1);
	dev_array<float> prev_U(N*N);
	cudaMemcpy(prev_U.getData(), U, N*N*sizeof(float), cudaMemcpyDeviceToDevice);
	upperTriangleInversionKernel<<<blocksPerGrid, threadsPerBlock,2*N*N*sizeof(float)>>>(U,prev_U.getData(),N);
}

/**
* matrixInversion
*
* @param  C   the result matrix (N*N)
*         L   the lower triangular matrix (N*N)
*         U   the upper triangular matrix (N*N)
*         N   the side of the array
*/
void matrixInversion(float* C,float* L,float *U,int N){

	dev_array<float> d_trans_L(N*N);

	upperTriangleInversion(U,N);
	matrixTransposition(L,d_trans_L.getData(),N);
	upperTriangleInversion(d_trans_L.getData(),N);
	matrixTransposition(d_trans_L.getData(),L,N);
	matrixMultiplication(U,L,C,N);
	
}

/**
* solveUpperTriangleEquations
*
* @param  U   the upper triangular matrix (N*N)
*		  x   the solution vector for "Ux=y"
*         y   the right part of the equation
*         N   the side of the array
*/
void solveUpperTriangleEquations(float* U,float* x,float *y,int N){
	dim3 threadsPerBlock(N);
    dim3 blocksPerGrid(1);
	solveUpperTriangleEquationsKernel<<<blocksPerGrid, threadsPerBlock,N*sizeof(float)>>>(U,x,y,N);
}

/**
* solveLowerTriangleEquations
*
* @param  L   the lower triangular matrix (N*N
*		  y   the solution vector for "Ly=b"
*         b   the right part of the equation
*         N   the side of the array
*/
void solveLowerTriangleEquations(float* L,float* y,float *b,int N){
	dim3 threadsPerBlock(N);
    dim3 blocksPerGrid(1);
	solveLowerTriangleEquationsKernel<<<blocksPerGrid, threadsPerBlock,N*sizeof(float)>>>(L,y,b,N);
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
void solveEquations(float* L,float* U,float* x,float *b,int N){
	dev_array<float> d_y(N);
	solveLowerTriangleEquations(L,d_y.getData(),b,N);
	solveUpperTriangleEquations(U,x,d_y.getData(),N);
}


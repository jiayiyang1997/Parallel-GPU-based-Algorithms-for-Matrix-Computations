/**
* main.cu: run the interface for function calling and results display
*/

#include <iostream>
#include <vector>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>
#include "dev_array.h"
#include "sequential.h"
#include "kernel.h"
#include <math.h>
#include<string>

#define MAX_SIZE 1024*1024

using namespace std;

/**
* matrix_read
*
* @param  A   the vector to store the matrix input
*         N   the side of the array
*/
void matrix_read(string filename,vector<float>& A, int N) {
	FILE* fp;
	float row, col;

	fp = fopen(filename.c_str(), "r");//open output file
	//open failed
	if (fp == NULL) {
		printf("Couldn't open the target file.\n");
		return;
	}

	for (row = 0; row < N; row++) {
		for (col = 0; col < N; col++)
			if (fscanf(fp, "%f,", &A[row * N + col]) == EOF) break;//read data
		if (feof(fp)) break;//if the file is over
	}

	fclose(fp);//close file
}

/**
* col_vector_read
*
* @param  A   the vector to store the column vector input
*         N   the side of the array
*/
void col_vector_read(string filename,vector<float>& A, int N) {
	FILE* fp;
	float row, col;

	fp = fopen(filename.c_str(), "r");//open output file
	//open failed
	if (fp == NULL) {
		printf("Couldn't open the target file.\n");
		return;
	}

	for (row = 0; row < N; row++) {
		for (col = 0; col < 1; col++)
			if (fscanf(fp, "%f", &A[row]) == EOF) break;//read data
		if (feof(fp)) break;//if the file is over
	}

	fclose(fp);//close file
}

/**
* output_result_matrix
*
* @param  C   the matrix to output
*         N   the side of the array
*/
void output_result_matrix(vector<float>& C, int N) {
	for (float ROW = 0; ROW < N; ROW++) {
		for (float COL = 0; COL < N; COL++) {
			if (COL == N - 1) {
				cout << C[ROW * N + COL];
			}
			else {
				cout << C[ROW * N + COL] << ", ";
			}
		}
		cout << endl;
	}
}

/**
* output_solution
*
* @param  C   the solution vector to output
*         N   the side of the array
*/
void output_solution(vector<float>& C, int N) {
	for (int i = 0; i < N; i++) {
		cout << C[i] << endl;
	}
}


//main method: run the user interface
int main(){
	int choice = -1;
	while (choice != 12) {
		cout << "*******Please choose the parallel operations on matrices*******" << endl;
		cout << "1. Matrix Addition" << endl;
		cout << "2. Matrix Subtraction" << endl;
		cout << "3. Matrix Multiplication" << endl;
		cout << "4. Scalar Multiplication" << endl;
		cout << "5. Matrix Transposition" << endl;
		cout << "6. Matrix LU Factorization (Base/Right Looking + Two pass)" << endl;
		//cout << "7. Matrix LU Factorization (Right Looking)" << endl;
		cout << "7. Matrix LU Factorization (Right Looking + One pass)" << endl;
		cout << "8. Matrix LU Factorization (Left Looking)" << endl;
		//cout << "10. Matrix LU Factorization (Two Pass)" << endl;
		cout << "9. Matrix Inversion" << endl;
		cout << "10. Matrix Determinant Calculation" << endl;
		cout << "11. Solver for Systems of equations" << endl;
		cout << "12. Exit" << endl;
		cout << "***************************Menu end****************************" << endl;
		cin >> choice;
		while (choice<1 || choice>12) {
			cout << "Please enter valid number." << endl;
			cin >> choice;
		}
		if(choice==12){
			break;
		}
		//Addition/Subtraction/Matrix Multiplication
		if (choice >= 1 && choice <= 3) {
			int N=-1;
			cout << "Please enter the dimension N of the matrices"<<endl;
			cin >> N;
			int SIZE = N * N;

			// Allocate memory on the host
			vector<float> h_A(SIZE);
			vector<float> h_B(SIZE);
			vector<float> h_C(SIZE);

			// Allocate memory on the device
			dev_array<float> d_A(SIZE);
			dev_array<float> d_B(SIZE);
			dev_array<float> d_C(SIZE);

			string file_A, file_B;
			//read A
			cout << "Please enter the name of fileA:" << endl;
			cin >> file_A;
			matrix_read(file_A, h_A, N);
			//read B
			cout << "Please enter the name of fileB:" << endl;
			cin >> file_B;
			matrix_read(file_B, h_B, N);

			//update values for d_A & d_B
			d_A.set(&h_A[0], SIZE);
			d_B.set(&h_B[0], SIZE);

			switch (choice) {
				//1. Matrix Addition
				case 1: {
					matrixAddition(d_A.getData(), d_B.getData(), d_C.getData(), N);
					break;
				}
				//2. Matrix Subtraction
				case 2: {
					matrixSubtraction(d_A.getData(), d_B.getData(), d_C.getData(), N);
					break;
				}
				//3. Scalar Multiplication
				case 3: {
					matrixMultiplication(d_A.getData(), d_B.getData(), d_C.getData(), N);
					break;
				}
			}
			d_C.get(&h_C[0], SIZE);

			//output the result C
			cout << "The result matrix is:" << endl;
			output_result_matrix(h_C, N);
		}
		//Scalar Multiplication
		else if (choice == 4) {
			int N = -1;
			cout << "Please enter the dimension N of the matrix A" << endl;
			cin >> N;
			int SIZE = N * N;

			// Allocate memory on the host
			vector<float> h_A(SIZE);
			vector<float> h_C(SIZE);

			// Allocate memory on the device
			dev_array<float> d_A(SIZE);
			dev_array<float> d_C(SIZE);

			string file_A;
			//read A
			cout << "Please enter the name of fileA:" << endl;
			cin >> file_A;
			matrix_read(file_A, h_A, N);

			//update values for d_A
			d_A.set(&h_A[0], SIZE);

			float scalar;
			cout << "Please enter the scalar:" << endl;
			cin >> scalar;

			//call scalarMultiplication method
			scalarMultiplication(d_A.getData(), scalar, d_C.getData(), N);
			d_C.get(&h_C[0], SIZE);

			//output the result C
			cout << "The result matrix is:" << endl;
			output_result_matrix(h_C, N);
		}
		else if (choice == 5) {
			int N = -1;
			cout << "Please enter the dimension N of the matrix A" << endl;
			cin >> N;
			float SIZE = N * N;

			// Allocate memory on the host
			vector<float> h_A(SIZE);
			vector<float> h_C(SIZE);
			vector<float> h_L(SIZE);
			vector<float> h_U(SIZE);

			// Allocate memory on the device
			dev_array<float> d_A(SIZE);
			dev_array<float> d_C(SIZE);

			string file_A;
			//read A
			cout << "Please enter the name of fileA:" << endl;
			cin >> file_A;
			matrix_read(file_A, h_A, N);

			//update values for d_A
			d_A.set(&h_A[0], SIZE);

			//call kernel
			matrixTransposition(d_A.getData(), d_C.getData(), N);
			d_C.get(&h_C[0], SIZE);
			//output the result C
			cout << "The result matrix is:" << endl;
			output_result_matrix(h_C, N);
		}
		else{
			int N = -1;
			cout << "Please enter the dimension N of the matrix A" << endl;
			cin >> N;
			float SIZE = N * N;

			// Allocate memory on the host
			vector<float> h_A(SIZE);
			vector<float> h_C(SIZE);
			vector<float> h_L(SIZE);
			vector<float> h_U(SIZE);

			// Allocate memory on the device
			dev_array<float> d_A(SIZE);
			dev_array<float> d_C(SIZE);

			string file_A;
			//read A
			cout << "Please enter the name of fileA:" << endl;
			cin >> file_A;
			matrix_read(file_A, h_A, N);

			//update values for d_A
			d_A.set(&h_A[0], SIZE);

			//LU: choice 6-11
			switch(choice){
				//6. Matrix LU Factorization (Base/Right Looking + Two pass)
				case 6:{
					LU_base(d_A.getData(),N);
					break;
			    }
				//7. Matrix LU Factorization (Right Looking + One pass)
				case 7 :{
					LU_onepass(d_A.getData(),N);
					break;
				}
				//8. Matrix LU Factorization (Left Looking)
				case 8:{
					LU_left_looking(d_A.getData(),N);
					break;
				}
				//For operations 9/10/11, use the LU_base algorithm to do the factorizaton
				default:{
					LU_base(d_A.getData(),N);
				}
				//to be added: case 10: two pass
			}
			d_A.get(&h_C[0], SIZE);
			split_LU(h_C,h_L,h_U,N);
			//If the function only conatins LU Factorization
			if(choice>=6 && choice<=8){
				//output the result C
				cout<<"The result of LU Factoriztion (overwrite the values in A):"<<endl;
				output_result_matrix(h_C, N);
				//output the result L & U
				cout << "The result L is:" << endl;
				output_result_matrix(h_L, N);
				cout << "The result U is:" << endl;
				output_result_matrix(h_U, N);
			}
			//9. Matrix Inversion 
			else if(choice==9){
				cout << "The result L is:" << endl;
				output_result_matrix(h_L, N);
				cout << "The result U is:" << endl;
				output_result_matrix(h_U, N);
				dev_array<float> d_U(SIZE),d_L(SIZE);
				d_L.set(&h_L[0], SIZE);
				d_U.set(&h_U[0], SIZE);
				matrixInversion(d_C.getData(),d_L.getData(),d_U.getData(),N);
				d_C.get(&h_C[0], SIZE);
				cout<<"The inversion of the matrix is:"<<endl;
				output_result_matrix(h_C, N);
			}
			//10. Matrix Determinant Calculation
			else if(choice==10){
				dev_array<float> d_U(SIZE);
				d_U.set(&h_U[0], SIZE);
				float det = getMatrixDeterminant(d_U.getData(),N);
				cout<<"The determinant of the given matrix is:"<<endl;
				cout<<det<<endl;
			}
			//11. Solver for Systems of equations
			else{
				dev_array<float> d_U(SIZE),d_L(SIZE);
				d_L.set(&h_L[0], SIZE);
				d_U.set(&h_U[0], SIZE);

				// Allocate memory on the device
				dev_array<float> d_b(N);
				dev_array<float> d_x(N);

				// Allocate memory on the host
				vector<float> h_b(N);
				vector<float> h_x(N);

				string file_b;

				//read b
				cout << "Please enter the name of fileb:" << endl;
				cin >> file_b;
				col_vector_read(file_b, h_b, N);
				output_solution(h_b,N);

				//update values for d_b
				d_b.set(&h_b[0], N);
				solveEquations(d_L.getData(),d_U.getData(),d_x.getData(),d_b.getData(),N);
				d_x.get(&h_x[0], N);

				cout<<"The solution to the system of equations is:"<<endl;
				output_solution(h_x,N);
			}
		}
	}
	return 0;
}
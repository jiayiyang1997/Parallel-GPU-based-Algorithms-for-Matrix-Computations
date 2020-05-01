# Parallel GPU based Algorithms for Matrix Computations

This is a term project for EE382C Multicore Computing (2020 Spring). The project title is "Parallel GPU based Algorithms for Matrix Computations". It covers a series of matrix computations from the basic ones (addition/subtraction/scalar multiplication/matrix multiplication) to the complex ones (LU Factorization and three derivative operations: matrix inversion/determinant calculation/solver for systems of equations).<br />
  
You can also find our presentation slides here: [Demo slides](https://docs.google.com/presentation/d/1gr_-XSUF-LBTZuASVELBzPx9WCqXcYmIe5Iq_sruASQ/edit?usp=sharing) <br />
  
and the project report here :[Project report](https://docs.google.com/presentation/d/1gr_-XSUF-LBTZuASVELBzPx9WCqXcYmIe5Iq_sruASQ/edit?usp=sharing).

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

* A local machine or server with CUDA-capable GPU
* OS which supports gcc compiler and toolchain
* NVIDIA CUDA Toolkit (available at http://developer.nvidia.com/cuda-downloads)

### Installing

To run this code on your local machine, please follow these steps.

1. Download the repository to your local machine

```
git clone https://github.com/multicore-sp20/Parallel-GPU-based-Algorithms-for-Matrix-Computations-1.git
```

2. Open the project in an IDE (a supported version of Microsoft Visual Studio or Xcode), after compilation and linking, run main.cu to enter the user interface. <br />
![](https://i.ibb.co/nwS5cbD/interface.png=800x800)

## Running the tests

We've designed a simple test input (a 4×4 integer matrix) to test all of our 11 operations. (Given a small input, we can see the correctness of the results more directly. In addition, since there are some requirements for the input matrices for the LU Factorization algorithms (invertible square matrices whose leading principle minors are all non-zero), besides the valid integer matrix that we chose, we also generate some random float matrices with different sizes (32/64/128/256/512) which are easier to satisfy these conditions.

The integer matrix input:

* A_4.txt
```
1,2,1,-2
2,5,3,-2
-2,-2,3,5
1,3,2,3
```

* B_4.txt
```
1,2,1,-2
2,5,3,-2
-2,-2,3,5
1,3,2,3
```

* b.txt (the right part of the equation)
```
2
8
4
9
```

The correct results of operations based on these inputs:

The result L is:
```
1, 0, 0, 0
2, 1, 0, 0
-2, 2, 1, 0
1, 1, 0, 1
```

The result U is:
```
1, 2, 1, -2
0, 1, 1, 2
0, 0, 3, -3
0, 0, 0, 3
```

The solution vector for "Ax=b" is:
```
1
1
1
1
```

The float input matrices are named as M"Size".txt where "Size" is from 32 to 512. The corresponding b vectors are named as V"Size".txt where "Size" is also from 32 to 512.

## Development Environment

* Microsoft Visual Studio 2010
* CUDA 9.0

## Authors

* **Yating Wu** - [Github Profile](https://github.com/lingchensanwen)
* **Jiayi Yang** - [Github Profile](https://github.com/jiayiyang1997)

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details


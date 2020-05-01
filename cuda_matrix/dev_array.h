/**
* The dev_array class is a class of arrays which are implemented on GPU (device memory)
* It has useful methods such as (1) the constructor (to allocate device memory for the array),(2) set method (copy data from host to device)
* (3) get method (copy data from host to device) and (4) the destructor(free the memory allocated automatically)
* Since in our project, we don't know the size of the matrix that the program is going to process on, we need to allocate the memory for it dynamically. 
* The dev_array class is a good choice for us to store the matrix values and do the operations based on its instances.
* The class design comes from this reference:
* https://www.quantstart.com/articles/dev_array_A_Useful_Array_Class_for_CUDA/
*/

#ifndef DEV_ARRAY_H
#define DEV_ARRAY_H

#include <stdexcept>
#include <algorithm>
#include <cuda_runtime.h>


template <class T>
class dev_array
{
	// public functions
public:
	explicit dev_array()
		: start_(0),
		end_(0)
	{}

	// constructor
	explicit dev_array(size_t size)
	{
		allocate(size);
	}

	// destructor
	~dev_array()
	{
		free();
	}

	// resize the vector
	void resize(size_t size)
	{
		free();
		allocate(size);
	}

	// get the size of the array
	size_t getSize() const
	{
		return end_ - start_;
	}

	// get data
	const T* getData() const
	{
		return start_;
	}

	T* getData()
	{
		return start_;
	}

	// set values (copy data from host to device)
	void set(const T* src, size_t size)
	{
		size_t min = std::min(size, getSize());
		cudaError_t result = cudaMemcpy(start_, src, min * sizeof(T), cudaMemcpyHostToDevice);
		if (result != cudaSuccess)
		{
			throw std::runtime_error("failed to copy to device memory");
		}
	}

	// get values (copy data from device to host)
	void get(T* dest, size_t size)
	{
		size_t min = std::min(size, getSize());
		cudaError_t result = cudaMemcpy(dest, start_, min * sizeof(T), cudaMemcpyDeviceToHost);
		if (result != cudaSuccess)
		{
			throw std::runtime_error("failed to copy to host memory");
		}
	}


	// private functions
private:
	// allocate memory on the device
	void allocate(size_t size)
	{
		cudaError_t result = cudaMalloc((void**)& start_, size * sizeof(T));
		if (result != cudaSuccess)
		{
			start_ = end_ = 0;
			throw std::runtime_error("failed to allocate device memory");
		}
		end_ = start_ + size;
	}

	// free memory on the device
	void free()
	{
		if (start_ != 0)
		{
			cudaFree(start_);
			start_ = end_ = 0;
		}
	}

	T* start_;
	T* end_;
};
#endif
#include "common_kernels.hpp"

#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
// Utilities and system includes
#include <helper_cuda.h>  // helper function CUDA error checking and initialization
#include <helper_functions.h>  // helper for shared functions common to CUDA Samples

__device__ double3 operator+(const double3 &a, const double3 &b) {

	return make_double3(a.x + b.x, a.y + b.y, a.z + b.z);

}
__device__ double3 operator-(const double3 &a, const double3 &b) {

	return make_double3(a.x - b.x, a.y - b.y, a.z - b.z);

}


__device__ double dot_product(const double3 &a, const double3 &b) {

	return (a.x * b.x + a.y * b.y + a.z * b.z);

}


__device__ double myatomicAdd(double* address, double val)
{
	unsigned long long int* address_as_ull =
		(unsigned long long int*)address;
	unsigned long long int old = *address_as_ull, assumed;

	do {
		assumed = old;
		old = atomicCAS(address_as_ull, assumed,
			__double_as_longlong(val +
				__longlong_as_double(assumed)));

		// Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
	} while (assumed != old);

	return __longlong_as_double(old);
}

__global__ void add_test(int n, double* delta_t, double3* area) {

	int index = threadIdx.x;
	int stride = blockDim.x;

	for (int i = index; i < n; i += stride) {
		double3 tmp;
		tmp = area[i];

		delta_t[i] = tmp.x + tmp.y + tmp.z;

	}
	return;
}



__global__ void clone_a_to_b(int n_cells, double4* a, double4* b) {

	//loop through cells

	int index = blockIdx.x * blockDim.x + threadIdx.x;;
	int stride = blockDim.x * gridDim.x;

	for (int i = index; i < n_cells; i += stride) {

		if (i < n_cells) {
			double4 tmp1 = a[i];
			double4 tmp2;

			tmp2.x = tmp1.x;
			tmp2.y = tmp1.y;
			tmp2.z = tmp1.z;
			tmp2.w = tmp1.w;

			b[i] = tmp2;

		}
	}
}

__global__ void fill_zero(int n_cells, double* a) {

	//loop through cells

	int index = blockIdx.x * blockDim.x + threadIdx.x;;
	int stride = blockDim.x * gridDim.x;

	for (int i = index; i < n_cells; i += stride) {
		if (i < n_cells) {
			a[i] = 0;

		}
	}
}

__global__ void fill_double(int n_cells, double* a, double val) {

	//loop through cells

	int index = blockIdx.x * blockDim.x + threadIdx.x;;
	int stride = blockDim.x * gridDim.x;

	for (int i = index; i < n_cells; i += stride) {
		if (i < n_cells) {
			a[i] = val;

		}
	}
}




__global__ void square(int n_cells, double* a) {

	//loop through cells

	int index = blockIdx.x * blockDim.x + threadIdx.x;;
	int stride = blockDim.x * gridDim.x;

	for (int i = index; i < n_cells; i += stride) {
		if (i < n_cells) {
			a[i] = a[i] * a[i];

		}
	}
}


__global__ void add(int n_cells, double* a, double *b) {

	//loop through cells

	int index = blockIdx.x * blockDim.x + threadIdx.x;;
	int stride = blockDim.x * gridDim.x;

	for (int i = index; i < n_cells; i += stride) {
		if (i < n_cells) {
			a[i] = a[i] + b[i];

		}
	}
}
#include <stdio.h>
#include <cuda.h>
#include <iostream>
#include <stdlib.h>
#include <vector>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include "Definitions.h"

namespace WhydahGally
{
	namespace Maths
	{
		//Kernels for computing the dot product between 2 matrices (one using global memory and one using the shared one).
		template<typename T> __global__ void MultipMatrices(T* d_inFirst, T* d_inSecond, T* d_out, int rowFirst, int colFirst, int colSecond);
		template<typename T> __global__ void MultipMatricesSH(T* d_inFirst, T* d_inSecond, T* d_out, int rowFirst, int colFirst, int colSecond);

		//Kernels for computing a transpose matrix (one using global memory and one using the shared one).
		template<typename T> __global__ void Transp(T* d_inMatrix, T* d_out, int rowMatrix, int colMatrix);
		template<typename T> __global__ void TranspSH(T *d_inMatrix, T *d_out, int rowMatrix, int colMatrix);

		//Kernels for computing the outer product between 2 matrices (one using global memory and one using the shared one).
		template<typename T> __global__ void OutPr(T* d_inFirst, T* d_inSecond, T* d_out, int rowFirst, int rowSecond);
		template<typename T> __global__ void OutPrSH(T* d_inFirst, T* d_inSecond, T* d_out, int rowFirst, int rowSecond);

		//Kernels for computing the difference between 2 matrices (one using global memory and one using the shared one).
		template<typename T> __global__ void VecDiff(T* d_inFirst, T* d_inSecond, T* d_out, int rows);
		template<typename T> __global__ void VecDiffSH(T* d_inFirst, T* d_inSecond, T* d_out, int rows);

		//CUDA Functions that invoke the kernels. 
		template<typename T> void matricesDotProductGPU(T* h_first, int rowFirst, int colFirst, T* h_second, int rowSecond, int colSecond, T* h_result);
		template<typename T> void matricesDotProductGPUSH(T* h_first, int rowFirst, int colFirst, T* h_second, int rowSecond, int colSecond, T* h_result);

		template<typename T> void transposeGPU(T* h_matrix, int rowMatrix, int colMatrix, T* h_result);
		template<typename T> void transposeGPUSH(T* h_matrix, int rowMatrix, int colMatrix, T* h_result);

		template<typename T> void outerProdGPU(T* h_first, int rowFirst, T* h_second, int rowSecond, T* h_result);
		template<typename T> void outerProdGPUSH(T* h_first, int rowFirst, T* h_second, int rowSecond, T* h_result);

		template<typename T> void vectorsDiffGPU(T* h_first, T* h_second, int rows, T* h_result);
		template<typename T> void vectorsDiffGPUSH(T* h_first, T* h_second, int rows, T* h_result);
	}
}











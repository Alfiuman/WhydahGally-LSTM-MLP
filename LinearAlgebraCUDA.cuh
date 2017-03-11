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
		__global__ void MultipMatrices(float* d_inFirst, float* d_inSecond, float* d_out, int rowFirst, int colFirst, int colSecond);
		__global__ void MultipMatricesSH(float* d_inFirst, float* d_inSecond, float* d_out, int rowFirst, int colFirst, int colSecond);

		__global__ void Transp(float* d_inMatrix, float* d_out, int rowMatrix, int colMatrix);
		__global__ void TranspSH(float *d_inMatrix, float *d_out, int rowMatrix, int colMatrix);

		__global__ void OutPr(float* d_inFirst, float* d_inSecond, float* d_out, int rowFirst, int rowSecond);
		__global__ void OutPrSH(float* d_inFirst, float* d_inSecond, float* d_out, int rowFirst, int rowSecond);

		__global__ void VecDiff(float* d_inFirst, float* d_inSecond, float* d_out, int rows);
		__global__ void VecDiffSH(float* d_inFirst, float* d_inSecond, float* d_out, int rows);

		void matricesDotProductGPU(float* h_first, const int& rowFirst, const int& colFirst, float* h_second, const int& rowSecond, const int& colSecond, float* h_result);
		void matricesDotProductGPUSH(float* h_first, const int& rowFirst, const int& colFirst, float* h_second, const int& rowSecond, const int& colSecond, float* h_result);

		void transposeGPU(float* h_matrix, const int& rowMatrix, const int& colMatrix, float* h_result);
		void transposeGPUSH(float* h_matrix, const int& rowMatrix, const int& colMatrix, float* h_result);

		void outerProdGPU(float* h_first, const int& rowFirst, float* h_second, const int& rowSecond, float* h_result);
		void outerProdGPUSH(float* h_first, const int& rowFirst, float* h_second, const int& rowSecond, float* h_result);

		void vectorsDiffGPU(float* h_first, float* h_second, const int& rows, float* h_result);
		void vectorsDiffGPUSH(float* h_first, float* h_second, const int& rows, float* h_result);
	}
}











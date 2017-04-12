#include "linearAlgebraCUDA.cuh"

namespace WhydahGally
{
	namespace Maths
	{
		//Kernels.
		template<typename T> __global__ void MultipMatrices(T* d_inFirst, T* d_inSecond, T* d_out, int rowFirst, int colFirst, int colSecond)
		{
			int row = blockIdx.y * blockDim.y + threadIdx.y;
			int col = blockIdx.x * blockDim.x + threadIdx.x;

			T sum = 0.0f;

			if (row >= rowFirst || col >= colSecond)
			{
				return;
			}

			for (int i = 0; i < colFirst; i++)
			{
				sum += d_inFirst[row * colFirst + i] * d_inSecond[i * colSecond + col];
			}

			d_out[row * colSecond + col] = sum;
		}

		template<typename T> __global__ void MultipMatricesSH(T* d_inFirst, T* d_inSecond, T* d_out, int rowFirst, int colFirst, int colSecond)
		{
			int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
			int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;

			__shared__ T shFirst[BLOCK_SIZE * BLOCK_SIZE];
			__shared__ T shSecond[BLOCK_SIZE * BLOCK_SIZE];

			T sum = 0.0f;

			for (int i = 0; i < (BLOCK_SIZE + colFirst - 1) / BLOCK_SIZE; i++)
			{
				if (i * BLOCK_SIZE + threadIdx.x < colFirst && row < rowFirst)
				{
					shFirst[threadIdx.y * BLOCK_SIZE + threadIdx.x] = d_inFirst[row * colFirst + i * BLOCK_SIZE + threadIdx.x];
				}
				else
				{
					shFirst[threadIdx.y * BLOCK_SIZE + threadIdx.x] = 0.0f;
				}

				if (i * BLOCK_SIZE + threadIdx.y < colFirst && col < colSecond)
				{
					shSecond[threadIdx.y * BLOCK_SIZE + threadIdx.x] = d_inSecond[(i * BLOCK_SIZE + threadIdx.y) * colSecond + col];
				}
				else
				{
					shSecond[threadIdx.y * BLOCK_SIZE + threadIdx.x] = 0.0f;
				}

				__syncthreads();

				for (int j = 0; j < BLOCK_SIZE; j++)
				{
					sum += shFirst[threadIdx.y * BLOCK_SIZE + j] * shSecond[j * BLOCK_SIZE + threadIdx.x];
				}

				__syncthreads();
			}

			if (row < rowFirst && col < colSecond)
			{
				d_out[((blockIdx.y * blockDim.y + threadIdx.y) * colSecond) + (blockIdx.x * blockDim.x) + threadIdx.x] = sum;
			}
		}

		template<typename T> __global__ void Transp(T* d_inMatrix, T* d_out, int rowMatrix, int colMatrix)
		{
			int row = blockIdx.y * blockDim.y + threadIdx.y;
			int col = blockIdx.x * blockDim.x + threadIdx.x;

			if (row >= rowMatrix || col >= colMatrix)
			{
				return;
			}

			d_out[col * rowMatrix + row] = d_inMatrix[row * colMatrix + col];
		}

		template<typename T> __global__ void TranspSH(T *d_inMatrix, T *d_out, int rowMatrix, int colMatrix)
		{
			int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
			int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;

			__shared__ T shTile[BLOCK_SIZE * BLOCK_SIZE];

			if (blockIdx.y * blockDim.y + threadIdx.y >= rowMatrix || blockIdx.x * blockDim.x + threadIdx.x >= colMatrix)
			{
				return;
			}

			shTile[threadIdx.y * BLOCK_SIZE + threadIdx.x] = d_inMatrix[row * colMatrix + col];

			__syncthreads();

			d_out[col * rowMatrix + row] = shTile[threadIdx.y * BLOCK_SIZE + threadIdx.x];
		}

		template<typename T> __global__ void OutPr(T* d_inFirst, T* d_inSecond, T* d_out, int rowFirst, int rowSecond)
		{
			int row = blockIdx.y * blockDim.y + threadIdx.y;
			int col = blockIdx.x * blockDim.x + threadIdx.x;

			if (row >= rowFirst || col >= rowSecond)
			{
				return;
			}

			d_out[row * rowSecond + col] = d_inFirst[row] * d_inSecond[col];
		}

		template<typename T> __global__ void OutPrSH(T* d_inFirst, T* d_inSecond, T* d_out, int rowFirst, int rowSecond)
		{
			int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
			int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;

			__shared__ T shFirst[BLOCK_SIZE * BLOCK_SIZE];
			__shared__ T shSecond[BLOCK_SIZE * BLOCK_SIZE];

			T prod = 0.0f;

			if (blockIdx.y * blockDim.y + threadIdx.y >= rowFirst || blockIdx.x * blockDim.x + threadIdx.x >= rowSecond)
			{
				return;
			}

			shFirst[row] = d_inFirst[blockIdx.y * blockDim.y + threadIdx.y];
			shSecond[col] = d_inSecond[blockIdx.x * blockDim.x + threadIdx.x];

			__syncthreads();

			prod = shFirst[row] * shSecond[col];

			d_out[(blockIdx.y * blockDim.y + threadIdx.y) * rowSecond + (blockIdx.x * blockDim.x + threadIdx.x)] = prod;
		}

		template<typename T> __global__ void VecDiff(T* d_inFirst, T* d_inSecond, T* d_out, int rows)
		{
			int row = blockIdx.x * blockDim.x + threadIdx.x;

			if (row >= rows)
			{
				return;
			}

			d_out[row] = d_inFirst[row] - d_inSecond[row];
		}

		template<typename T> __global__ void VecDiffSH(T* d_inFirst, T* d_inSecond, T* d_out, int rows)
		{
			int row = blockIdx.x * blockDim.x + threadIdx.x;

			__shared__ T shFirst[BLOCK_SIZE * BLOCK_SIZE];
			__shared__ T shSecond[BLOCK_SIZE * BLOCK_SIZE];

			T diff = 0.0f;

			if (row >= rows)
			{
				return;
			}

			shFirst[blockIdx.x * BLOCK_SIZE + threadIdx.x] = d_inFirst[row];
			shSecond[blockIdx.x * BLOCK_SIZE + threadIdx.x] = d_inSecond[row];

			__syncthreads();

			diff = shFirst[blockIdx.x * BLOCK_SIZE + threadIdx.x] - shSecond[blockIdx.x * BLOCK_SIZE + threadIdx.x];

			d_out[row] = diff;
		}

		//Functions.
		template<typename T> void matricesDotProductGPU(T* h_first, const int& rowFirst, const int& colFirst, T* h_second, const int& rowSecond, const int& colSecond, T* h_result)
		{
			T* d_inFirst;
			T* d_inSecond;
			T* d_out;

			const int BYFIRST = (rowFirst * colFirst) * sizeof(T);
			const int BYSECOND = (rowSecond * colSecond) * sizeof(T);
			const int BYRESULT = (rowFirst * colSecond) * sizeof(T);

			cudaMalloc((void**)&d_inFirst, BYFIRST);
			cudaMalloc((void**)&d_inSecond, BYSECOND);
			cudaMalloc((void**)&d_out, BYRESULT);

			cudaMemcpy(d_inFirst, h_first, BYFIRST, cudaMemcpyHostToDevice);
			cudaMemcpy(d_inSecond, h_second, BYSECOND, cudaMemcpyHostToDevice);

			dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
			dim3 dimGrid((colSecond + dimBlock.x - 1) / dimBlock.x, (rowFirst + dimBlock.y - 1) / dimBlock.y);

			MultipMatrices <<<dimGrid, dimBlock>>>(d_inFirst, d_inSecond, d_out, rowFirst, colFirst, colSecond);

			cudaMemcpy(h_result, d_out, BYRESULT, cudaMemcpyDeviceToHost);

			cudaFree(d_inFirst);
			cudaFree(d_inSecond);
			cudaFree(d_out);
		}

		template<typename T> void matricesDotProductGPUSH(T* h_first, const int& rowFirst, const int& colFirst, T* h_second, const int& rowSecond, const int& colSecond, T* h_result)
		{
			T* d_inFirst;
			T* d_inSecond;
			T* d_out;

			const int BYFIRST = (rowFirst * colFirst) * sizeof(T);
			const int BYSECOND = (rowSecond * colSecond) * sizeof(T);
			const int BYRESULT = (rowFirst * colSecond) * sizeof(T);

			cudaMalloc((void**)&d_inFirst, BYFIRST);
			cudaMalloc((void**)&d_inSecond, BYSECOND);
			cudaMalloc((void**)&d_out, BYRESULT);

			cudaMemcpy(d_inFirst, h_first, BYFIRST, cudaMemcpyHostToDevice);
			cudaMemcpy(d_inSecond, h_second, BYSECOND, cudaMemcpyHostToDevice);

			dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
			dim3 dimGrid((colSecond + dimBlock.x - 1) / dimBlock.x, (rowFirst + dimBlock.y - 1) / dimBlock.y);

			MultipMatricesSH <<<dimGrid, dimBlock>>>(d_inFirst, d_inSecond, d_out, rowFirst, colFirst, colSecond);

			cudaMemcpy(h_result, d_out, BYRESULT, cudaMemcpyDeviceToHost);

			cudaFree(d_inFirst);
			cudaFree(d_inSecond);
			cudaFree(d_out);
		}

		template<typename T> void transposeGPU(T* h_matrix, const int& rowMatrix, const int& colMatrix, T* h_result)
		{
			T* d_inMatrix;
			T* d_out;

			const int BYTES = (rowMatrix * colMatrix) * sizeof(T);

			cudaMalloc((void**)&d_inMatrix, BYTES);
			cudaMalloc((void**)&d_out, BYTES);

			cudaMemcpy(d_inMatrix, h_matrix, BYTES, cudaMemcpyHostToDevice);

			dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
			dim3 dimGrid((colMatrix + dimBlock.x - 1) / dimBlock.x, (rowMatrix + dimBlock.y - 1) / dimBlock.y);

			Transp <<<dimGrid, dimBlock>>>(d_inMatrix, d_out, rowMatrix, colMatrix);

			cudaMemcpy(h_result, d_out, BYTES, cudaMemcpyDeviceToHost);

			cudaFree(d_inMatrix);
			cudaFree(d_out);
		}

		template<typename T> void transposeGPUSH(T* h_matrix, const int& rowMatrix, const int& colMatrix, T* h_result)
		{
			T* d_inMatrix;
			T* d_out;

			const int BYTES = (rowMatrix * colMatrix) * sizeof(T);

			cudaMalloc((void**)&d_inMatrix, BYTES);
			cudaMalloc((void**)&d_out, BYTES);

			cudaMemcpy(d_inMatrix, h_matrix, BYTES, cudaMemcpyHostToDevice);

			dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
			dim3 dimGrid((colMatrix + dimBlock.x - 1) / dimBlock.x, (rowMatrix + dimBlock.y - 1) / dimBlock.y);

			TranspSH <<<dimGrid, dimBlock>>>(d_inMatrix, d_out, rowMatrix, colMatrix);

			cudaMemcpy(h_result, d_out, BYTES, cudaMemcpyDeviceToHost);

			cudaFree(d_inMatrix);
			cudaFree(d_out);
		}

		template<typename T> void outerProdGPU(T* h_first, const int& rowFirst, T* h_second, const int& rowSecond, T* h_result)
		{
			T* d_inFirst;
			T* d_inSecond;
			T* d_out;

			const int BYFIRST = rowFirst * sizeof(T);
			const int BYSECOND = rowSecond * sizeof(T);
			const int BYRESULT = (rowFirst * rowSecond) * sizeof(T);

			cudaMalloc((void**)&d_inFirst, BYFIRST);
			cudaMalloc((void**)&d_inSecond, BYSECOND);
			cudaMalloc((void**)&d_out, BYRESULT);

			cudaMemcpy(d_inFirst, h_first, BYFIRST, cudaMemcpyHostToDevice);
			cudaMemcpy(d_inSecond, h_second, BYSECOND, cudaMemcpyHostToDevice);

			dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
			dim3 dimGrid((rowSecond + dimBlock.x - 1) / dimBlock.x, (rowFirst + dimBlock.y - 1) / dimBlock.y);

			OutPr <<<dimGrid, dimBlock>>>(d_inFirst, d_inSecond, d_out, rowFirst, rowSecond);

			cudaMemcpy(h_result, d_out, BYRESULT, cudaMemcpyDeviceToHost);

			cudaFree(d_inFirst);
			cudaFree(d_inSecond);
			cudaFree(d_out);
		}

		template<typename T> void outerProdGPUSH(T* h_first, const int& rowFirst, T* h_second, const int& rowSecond, T* h_result)
		{
			T* d_inFirst;
			T* d_inSecond;
			T* d_out;

			const int BYFIRST = rowFirst * sizeof(T);
			const int BYSECOND = rowSecond * sizeof(T);
			const int BYRESULT = (rowFirst * rowSecond) * sizeof(T);

			cudaMalloc((void**)&d_inFirst, BYFIRST);
			cudaMalloc((void**)&d_inSecond, BYSECOND);
			cudaMalloc((void**)&d_out, BYRESULT);

			cudaMemcpy(d_inFirst, h_first, BYFIRST, cudaMemcpyHostToDevice);
			cudaMemcpy(d_inSecond, h_second, BYSECOND, cudaMemcpyHostToDevice);

			dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
			dim3 dimGrid((rowSecond + dimBlock.x - 1) / dimBlock.x, (rowFirst + dimBlock.y - 1) / dimBlock.y);

			OutPrSH <<<dimGrid, dimBlock>>>(d_inFirst, d_inSecond, d_out, rowFirst, rowSecond);

			cudaMemcpy(h_result, d_out, BYRESULT, cudaMemcpyDeviceToHost);

			cudaFree(d_inFirst);
			cudaFree(d_inSecond);
			cudaFree(d_out);
		}

		template<typename T> void vectorsDiffGPU(T* h_first, T* h_second, const int& rows, T* h_result)
		{
			T* d_inFirst;
			T* d_inSecond;
			T* d_out;

			const int BYFIRST = rows * sizeof(T);
			const int BYSECOND = rows * sizeof(T);
			const int BYRESULT = rows * sizeof(T);

			cudaMalloc((void**)&d_inFirst, BYFIRST);
			cudaMalloc((void**)&d_inSecond, BYSECOND);
			cudaMalloc((void**)&d_out, BYRESULT);

			cudaMemcpy(d_inFirst, h_first, BYFIRST, cudaMemcpyHostToDevice);
			cudaMemcpy(d_inSecond, h_second, BYSECOND, cudaMemcpyHostToDevice);

			dim3 dimBlock(BLOCK_SIZE * BLOCK_SIZE);
			dim3 dimGrid((rows + dimBlock.x - 1) / dimBlock.x);

			VecDiff <<<dimGrid, dimBlock>>>(d_inFirst, d_inSecond, d_out, rows);

			cudaMemcpy(h_result, d_out, BYRESULT, cudaMemcpyDeviceToHost);

			cudaFree(d_inFirst);
			cudaFree(d_inSecond);
			cudaFree(d_out);
		}

		template<typename T> void vectorsDiffGPUSH(T* h_first, T* h_second, const int& rows, T* h_result)
		{
			T* d_inFirst;
			T* d_inSecond;
			T* d_out;

			const int BYFIRST = rows * sizeof(T);
			const int BYSECOND = rows * sizeof(T);
			const int BYRESULT = rows * sizeof(T);

			cudaMalloc((void**)&d_inFirst, BYFIRST);
			cudaMalloc((void**)&d_inSecond, BYSECOND);
			cudaMalloc((void**)&d_out, BYRESULT);

			cudaMemcpy(d_inFirst, h_first, BYFIRST, cudaMemcpyHostToDevice);
			cudaMemcpy(d_inSecond, h_second, BYSECOND, cudaMemcpyHostToDevice);

			dim3 dimBlock(BLOCK_SIZE * BLOCK_SIZE);
			dim3 dimGrid((rows + dimBlock.x - 1) / dimBlock.x);

			VecDiffSH <<<dimGrid, dimBlock>>>(d_inFirst, d_inSecond, d_out, rows);

			cudaMemcpy(h_result, d_out, BYRESULT, cudaMemcpyDeviceToHost);

			cudaFree(d_inFirst);
			cudaFree(d_inSecond);
			cudaFree(d_out);
		}

		//Forced template instantiations.
		//float
		template __global__ void MultipMatrices(float* d_inFirst, float* d_inSecond, float* d_out, int rowFirst, int colFirst, int colSecond);
		template __global__ void MultipMatricesSH(float* d_inFirst, float* d_inSecond, float* d_out, int rowFirst, int colFirst, int colSecond);

		template __global__ void Transp(float* d_inMatrix, float* d_out, int rowMatrix, int colMatrix);
		template __global__ void TranspSH(float *d_inMatrix, float *d_out, int rowMatrix, int colMatrix);

		template __global__ void OutPr(float* d_inFirst, float* d_inSecond, float* d_out, int rowFirst, int rowSecond);
		template __global__ void OutPrSH(float* d_inFirst, float* d_inSecond, float* d_out, int rowFirst, int rowSecond);

		template __global__ void VecDiff(float* d_inFirst, float* d_inSecond, float* d_out, int rows);
		template __global__ void VecDiffSH(float* d_inFirst, float* d_inSecond, float* d_out, int rows);

		template void matricesDotProductGPU(float* h_first, const int& rowFirst, const int& colFirst, float* h_second, const int& rowSecond, const int& colSecond, float* h_result);
		template void matricesDotProductGPUSH(float* h_first, const int& rowFirst, const int& colFirst, float* h_second, const int& rowSecond, const int& colSecond, float* h_result);

		template void transposeGPU(float* h_matrix, const int& rowMatrix, const int& colMatrix, float* h_result);
		template void transposeGPUSH(float* h_matrix, const int& rowMatrix, const int& colMatrix, float* h_result);

		template void outerProdGPU(float* h_first, const int& rowFirst, float* h_second, const int& rowSecond, float* h_result);
		template void outerProdGPUSH(float* h_first, const int& rowFirst, float* h_second, const int& rowSecond, float* h_result);

		template void vectorsDiffGPU(float* h_first, float* h_second, const int& rows, float* h_result);
		template void vectorsDiffGPUSH(float* h_first, float* h_second, const int& rows, float* h_result);

		//double
		template __global__ void MultipMatrices(double* d_inFirst, double* d_inSecond, double* d_out, int rowFirst, int colFirst, int colSecond);
		template __global__ void MultipMatricesSH(double* d_inFirst, double* d_inSecond, double* d_out, int rowFirst, int colFirst, int colSecond);

		template __global__ void Transp(double* d_inMatrix, double* d_out, int rowMatrix, int colMatrix);
		template __global__ void TranspSH(double *d_inMatrix, double *d_out, int rowMatrix, int colMatrix);

		template __global__ void OutPr(double* d_inFirst, double* d_inSecond, double* d_out, int rowFirst, int rowSecond);
		template __global__ void OutPrSH(double* d_inFirst, double* d_inSecond, double* d_out, int rowFirst, int rowSecond);

		template __global__ void VecDiff(double* d_inFirst, double* d_inSecond, double* d_out, int rows);
		template __global__ void VecDiffSH(double* d_inFirst, double* d_inSecond, double* d_out, int rows);

		template void matricesDotProductGPU(double* h_first, const int& rowFirst, const int& colFirst, double* h_second, const int& rowSecond, const int& colSecond, double* h_result);
		template void matricesDotProductGPUSH(double* h_first, const int& rowFirst, const int& colFirst, double* h_second, const int& rowSecond, const int& colSecond, double* h_result);

		template void transposeGPU(double* h_matrix, const int& rowMatrix, const int& colMatrix, double* h_result);
		template void transposeGPUSH(double* h_matrix, const int& rowMatrix, const int& colMatrix, double* h_result);

		template void outerProdGPU(double* h_first, const int& rowFirst, double* h_second, const int& rowSecond, double* h_result);
		template void outerProdGPUSH(double* h_first, const int& rowFirst, double* h_second, const int& rowSecond, double* h_result);

		template void vectorsDiffGPU(double* h_first, double* h_second, const int& rows, double* h_result);
		template void vectorsDiffGPUSH(double* h_first, double* h_second, const int& rows, double* h_result);

		//int
		template __global__ void MultipMatrices(int* d_inFirst, int* d_inSecond, int* d_out, int rowFirst, int colFirst, int colSecond);
		template __global__ void MultipMatricesSH(int* d_inFirst, int* d_inSecond, int* d_out, int rowFirst, int colFirst, int colSecond);

		template __global__ void Transp(int* d_inMatrix, int* d_out, int rowMatrix, int colMatrix);
		template __global__ void TranspSH(int *d_inMatrix, int *d_out, int rowMatrix, int colMatrix);

		template __global__ void OutPr(int* d_inFirst, int* d_inSecond, int* d_out, int rowFirst, int rowSecond);
		template __global__ void OutPrSH(int* d_inFirst, int* d_inSecond, int* d_out, int rowFirst, int rowSecond);

		template __global__ void VecDiff(int* d_inFirst, int* d_inSecond, int* d_out, int rows);
		template __global__ void VecDiffSH(int* d_inFirst, int* d_inSecond, int* d_out, int rows);

		template void matricesDotProductGPU(int* h_first, const int& rowFirst, const int& colFirst, int* h_second, const int& rowSecond, const int& colSecond, int* h_result);
		template void matricesDotProductGPUSH(int* h_first, const int& rowFirst, const int& colFirst, int* h_second, const int& rowSecond, const int& colSecond, int* h_result);

		template void transposeGPU(int* h_matrix, const int& rowMatrix, const int& colMatrix, int* h_result);
		template void transposeGPUSH(int* h_matrix, const int& rowMatrix, const int& colMatrix, int* h_result);

		template void outerProdGPU(int* h_first, const int& rowFirst, int* h_second, const int& rowSecond, int* h_result);
		template void outerProdGPUSH(int* h_first, const int& rowFirst, int* h_second, const int& rowSecond, int* h_result);

		template void vectorsDiffGPU(int* h_first, int* h_second, const int& rows, int* h_result);
		template void vectorsDiffGPUSH(int* h_first, int* h_second, const int& rows, int* h_result);

		//unsigned int
		template __global__ void MultipMatrices(unsigned int* d_inFirst, unsigned int* d_inSecond, unsigned int* d_out, int rowFirst, int colFirst, int colSecond);
		template __global__ void MultipMatricesSH(unsigned int* d_inFirst, unsigned int* d_inSecond, unsigned int* d_out, int rowFirst, int colFirst, int colSecond);

		template __global__ void Transp(unsigned int* d_inMatrix, unsigned int* d_out, int rowMatrix, int colMatrix);
		template __global__ void TranspSH(unsigned int *d_inMatrix, unsigned int *d_out, int rowMatrix, int colMatrix);

		template __global__ void OutPr(unsigned int* d_inFirst, unsigned int* d_inSecond, unsigned int* d_out, int rowFirst, int rowSecond);
		template __global__ void OutPrSH(unsigned int* d_inFirst, unsigned int* d_inSecond, unsigned int* d_out, int rowFirst, int rowSecond);

		template __global__ void VecDiff(unsigned int* d_inFirst, unsigned int* d_inSecond, unsigned int* d_out, int rows);
		template __global__ void VecDiffSH(unsigned int* d_inFirst, unsigned int* d_inSecond, unsigned int* d_out, int rows);

		template void matricesDotProductGPU(unsigned int* h_first, const int& rowFirst, const int& colFirst, unsigned int* h_second, const int& rowSecond, const int& colSecond, unsigned int* h_result);
		template void matricesDotProductGPUSH(unsigned int* h_first, const int& rowFirst, const int& colFirst, unsigned int* h_second, const int& rowSecond, const int& colSecond, unsigned int* h_result);

		template void transposeGPU(unsigned int* h_matrix, const int& rowMatrix, const int& colMatrix, unsigned int* h_result);
		template void transposeGPUSH(unsigned int* h_matrix, const int& rowMatrix, const int& colMatrix, unsigned int* h_result);

		template void outerProdGPU(unsigned int* h_first, const int& rowFirst, unsigned int* h_second, const int& rowSecond, unsigned int* h_result);
		template void outerProdGPUSH(unsigned int* h_first, const int& rowFirst, unsigned int* h_second, const int& rowSecond, unsigned int* h_result);

		template void vectorsDiffGPU(unsigned int* h_first, unsigned int* h_second, const int& rows, unsigned int* h_result);
		template void vectorsDiffGPUSH(unsigned int* h_first, unsigned int* h_second, const int& rows, unsigned int* h_result);

		//long
		template __global__ void MultipMatrices(long* d_inFirst, long* d_inSecond, long* d_out, int rowFirst, int colFirst, int colSecond);
		template __global__ void MultipMatricesSH(long* d_inFirst, long* d_inSecond, long* d_out, int rowFirst, int colFirst, int colSecond);

		template __global__ void Transp(long* d_inMatrix, long* d_out, int rowMatrix, int colMatrix);
		template __global__ void TranspSH(long *d_inMatrix, long *d_out, int rowMatrix, int colMatrix);

		template __global__ void OutPr(long* d_inFirst, long* d_inSecond, long* d_out, int rowFirst, int rowSecond);
		template __global__ void OutPrSH(long* d_inFirst, long* d_inSecond, long* d_out, int rowFirst, int rowSecond);

		template __global__ void VecDiff(long* d_inFirst, long* d_inSecond, long* d_out, int rows);
		template __global__ void VecDiffSH(long* d_inFirst, long* d_inSecond, long* d_out, int rows);

		template void matricesDotProductGPU(long* h_first, const int& rowFirst, const int& colFirst, long* h_second, const int& rowSecond, const int& colSecond, long* h_result);
		template void matricesDotProductGPUSH(long* h_first, const int& rowFirst, const int& colFirst, long* h_second, const int& rowSecond, const int& colSecond, long* h_result);

		template void transposeGPU(long* h_matrix, const int& rowMatrix, const int& colMatrix, long* h_result);
		template void transposeGPUSH(long* h_matrix, const int& rowMatrix, const int& colMatrix, long* h_result);

		template void outerProdGPU(long* h_first, const int& rowFirst, long* h_second, const int& rowSecond, long* h_result);
		template void outerProdGPUSH(long* h_first, const int& rowFirst, long* h_second, const int& rowSecond, long* h_result);

		template void vectorsDiffGPU(long* h_first, long* h_second, const int& rows, long* h_result);
		template void vectorsDiffGPUSH(long* h_first, long* h_second, const int& rows, long* h_result);

		//unsigned long
		template __global__ void MultipMatrices(unsigned long* d_inFirst, unsigned long* d_inSecond, unsigned long* d_out, int rowFirst, int colFirst, int colSecond);
		template __global__ void MultipMatricesSH(unsigned long* d_inFirst, unsigned long* d_inSecond, unsigned long* d_out, int rowFirst, int colFirst, int colSecond);

		template __global__ void Transp(unsigned long* d_inMatrix, unsigned long* d_out, int rowMatrix, int colMatrix);
		template __global__ void TranspSH(unsigned long *d_inMatrix, unsigned long *d_out, int rowMatrix, int colMatrix);

		template __global__ void OutPr(unsigned long* d_inFirst, unsigned long* d_inSecond, unsigned long* d_out, int rowFirst, int rowSecond);
		template __global__ void OutPrSH(unsigned long* d_inFirst, unsigned long* d_inSecond, unsigned long* d_out, int rowFirst, int rowSecond);

		template __global__ void VecDiff(unsigned long* d_inFirst, unsigned long* d_inSecond, unsigned long* d_out, int rows);
		template __global__ void VecDiffSH(unsigned long* d_inFirst, unsigned long* d_inSecond, unsigned long* d_out, int rows);

		template void matricesDotProductGPU(unsigned long* h_first, const int& rowFirst, const int& colFirst, unsigned long* h_second, const int& rowSecond, const int& colSecond, unsigned long* h_result);
		template void matricesDotProductGPUSH(unsigned long* h_first, const int& rowFirst, const int& colFirst, unsigned long* h_second, const int& rowSecond, const int& colSecond, unsigned long* h_result);

		template void transposeGPU(unsigned long* h_matrix, const int& rowMatrix, const int& colMatrix, unsigned long* h_result);
		template void transposeGPUSH(unsigned long* h_matrix, const int& rowMatrix, const int& colMatrix, unsigned long* h_result);

		template void outerProdGPU(unsigned long* h_first, const int& rowFirst, unsigned long* h_second, const int& rowSecond, unsigned long* h_result);
		template void outerProdGPUSH(unsigned long* h_first, const int& rowFirst, unsigned long* h_second, const int& rowSecond, unsigned long* h_result);

		template void vectorsDiffGPU(unsigned long* h_first, unsigned long* h_second, const int& rows, unsigned long* h_result);
		template void vectorsDiffGPUSH(unsigned long* h_first, unsigned long* h_second, const int& rows, unsigned long* h_result);
	}
}






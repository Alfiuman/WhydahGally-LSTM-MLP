#include "linearAlgebraCUDA.cuh"

namespace WhydahGally
{
	namespace Maths
	{
		//Kernels.
		__global__ void MultipMatrices(float* d_inFirst, float* d_inSecond, float* d_out, int rowFirst, int colFirst, int colSecond)
		{
			int row = blockIdx.y * blockDim.y + threadIdx.y;
			int col = blockIdx.x * blockDim.x + threadIdx.x;

			float sum = 0.0f;

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

		__global__ void MultipMatricesSH(float* d_inFirst, float* d_inSecond, float* d_out, int rowFirst, int colFirst, int colSecond)
		{
			int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
			int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;

			__shared__ float shFirst[BLOCK_SIZE * BLOCK_SIZE];
			__shared__ float shSecond[BLOCK_SIZE * BLOCK_SIZE];

			float sum = 0.0f;

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

		__global__ void Transp(float* d_inMatrix, float* d_out, int rowMatrix, int colMatrix)
		{
			int row = blockIdx.y * blockDim.y + threadIdx.y;
			int col = blockIdx.x * blockDim.x + threadIdx.x;

			if (row >= rowMatrix || col >= colMatrix)
			{
				return;
			}

			d_out[col * rowMatrix + row] = d_inMatrix[row * colMatrix + col];
		}

		__global__ void TranspSH(float *d_inMatrix, float *d_out, int rowMatrix, int colMatrix)
		{
			int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
			int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;

			__shared__ float shTile[BLOCK_SIZE * BLOCK_SIZE];

			if (blockIdx.y * blockDim.y + threadIdx.y >= rowMatrix || blockIdx.x * blockDim.x + threadIdx.x >= colMatrix)
			{
				return;
			}

			shTile[threadIdx.y * BLOCK_SIZE + threadIdx.x] = d_inMatrix[row * colMatrix + col];

			d_out[col * rowMatrix + row] = shTile[threadIdx.y * BLOCK_SIZE + threadIdx.x];
		}

		__global__ void OutPr(float* d_inFirst, float* d_inSecond, float* d_out, int rowFirst, int rowSecond)
		{
			int row = blockIdx.y * blockDim.y + threadIdx.y;
			int col = blockIdx.x * blockDim.x + threadIdx.x;

			if (row >= rowFirst || col >= rowSecond)
			{
				return;
			}

			d_out[row * rowSecond + col] = d_inFirst[row] * d_inSecond[col];
		}

		__global__ void OutPrSH(float* d_inFirst, float* d_inSecond, float* d_out, int rowFirst, int rowSecond)
		{
			int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
			int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;

			__shared__ float shFirst[BLOCK_SIZE * BLOCK_SIZE];
			__shared__ float shSecond[BLOCK_SIZE * BLOCK_SIZE];

			float prod = 0.0f;

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

		__global__ void VecDiff(float* d_inFirst, float* d_inSecond, float* d_out, int rows)
		{
			int row = blockIdx.x * blockDim.x + threadIdx.x;

			if (row >= rows)
			{
				return;
			}

			d_out[row] = d_inFirst[row] - d_inSecond[row];
		}

		__global__ void VecDiffSH(float* d_inFirst, float* d_inSecond, float* d_out, int rows)
		{
			int row = blockIdx.x * blockDim.x + threadIdx.x;

			__shared__ float shFirst[BLOCK_SIZE * BLOCK_SIZE];
			__shared__ float shSecond[BLOCK_SIZE * BLOCK_SIZE];

			float diff = 0.0f;

			if (blockIdx.x * blockDim.x + threadIdx.x >= rows)
			{
				return;
			}

			shFirst[blockIdx.x * BLOCK_SIZE + threadIdx.x] = d_inFirst[blockIdx.x * blockDim.x + threadIdx.x];
			shSecond[blockIdx.x * BLOCK_SIZE + threadIdx.x] = d_inSecond[blockIdx.x * blockDim.x + threadIdx.x];

			__syncthreads();

			diff = shFirst[blockIdx.x * BLOCK_SIZE + threadIdx.x] - shSecond[blockIdx.x * BLOCK_SIZE + threadIdx.x];

			d_out[row] = diff;
		}

		//Functions.
		void matricesDotProductGPU(float* h_first, const int& rowFirst, const int& colFirst, float* h_second, const int& rowSecond, const int& colSecond, float* h_result)
		{
			float* d_inFirst;
			float* d_inSecond;
			float* d_out;

			const int BYFIRST = (rowFirst * colFirst) * sizeof(float);
			const int BYSECOND = (rowSecond * colSecond) * sizeof(float);
			const int BYRESULT = (rowFirst * colSecond) * sizeof(float);

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

		void matricesDotProductGPUSH(float* h_first, const int& rowFirst, const int& colFirst, float* h_second, const int& rowSecond, const int& colSecond, float* h_result)
		{
			float* d_inFirst;
			float* d_inSecond;
			float* d_out;

			const int BYFIRST = (rowFirst * colFirst) * sizeof(float);
			const int BYSECOND = (rowSecond * colSecond) * sizeof(float);
			const int BYRESULT = (rowFirst * colSecond) * sizeof(float);

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

		void transposeGPU(float* h_matrix, const int& rowMatrix, const int& colMatrix, float* h_result)
		{
			float* d_inMatrix;
			float* d_out;

			const int BYTES = (rowMatrix * colMatrix) * sizeof(float);

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

		void transposeGPUSH(float* h_matrix, const int& rowMatrix, const int& colMatrix, float* h_result)
		{
			float* d_inMatrix;
			float* d_out;

			const int BYTES = (rowMatrix * colMatrix) * sizeof(float);

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

		void outerProdGPU(float* h_first, const int& rowFirst, float* h_second, const int& rowSecond, float* h_result)
		{
			float* d_inFirst;
			float* d_inSecond;
			float* d_out;

			const int BYFIRST = rowFirst * sizeof(float);
			const int BYSECOND = rowSecond * sizeof(float);
			const int BYRESULT = (rowFirst * rowSecond) * sizeof(float);

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

		void outerProdGPUSH(float* h_first, const int& rowFirst, float* h_second, const int& rowSecond, float* h_result)
		{
			float* d_inFirst;
			float* d_inSecond;
			float* d_out;

			const int BYFIRST = rowFirst * sizeof(float);
			const int BYSECOND = rowSecond * sizeof(float);
			const int BYRESULT = (rowFirst * rowSecond) * sizeof(float);

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

		void vectorsDiffGPU(float* h_first, float* h_second, const int& rows, float* h_result)
		{
			float* d_inFirst;
			float* d_inSecond;
			float* d_out;

			const int BYFIRST = rows * sizeof(float);
			const int BYSECOND = rows * sizeof(float);
			const int BYRESULT = rows * sizeof(float);

			cudaMalloc((void**)&d_inFirst, BYFIRST);
			cudaMalloc((void**)&d_inSecond, BYSECOND);
			cudaMalloc((void**)&d_out, BYRESULT);

			cudaMemcpy(d_inFirst, h_first, BYFIRST, cudaMemcpyHostToDevice);
			cudaMemcpy(d_inSecond, h_second, BYSECOND, cudaMemcpyHostToDevice);

			dim3 dimBlock(BLOCK_SIZE);
			dim3 dimGrid((rows + dimBlock.x - 1) / dimBlock.x);

			VecDiff <<<dimGrid, dimBlock>>>(d_inFirst, d_inSecond, d_out, rows);

			cudaMemcpy(h_result, d_out, BYRESULT, cudaMemcpyDeviceToHost);

			cudaFree(d_inFirst);
			cudaFree(d_inSecond);
			cudaFree(d_out);
		}

		void vectorsDiffGPUSH(float* h_first, float* h_second, const int& rows, float* h_result)
		{
			float* d_inFirst;
			float* d_inSecond;
			float* d_out;

			const int BYFIRST = rows * sizeof(float);
			const int BYSECOND = rows * sizeof(float);
			const int BYRESULT = rows * sizeof(float);

			cudaMalloc((void**)&d_inFirst, BYFIRST);
			cudaMalloc((void**)&d_inSecond, BYSECOND);
			cudaMalloc((void**)&d_out, BYRESULT);

			cudaMemcpy(d_inFirst, h_first, BYFIRST, cudaMemcpyHostToDevice);
			cudaMemcpy(d_inSecond, h_second, BYSECOND, cudaMemcpyHostToDevice);

			dim3 dimBlock(BLOCK_SIZE);
			dim3 dimGrid((rows + dimBlock.x - 1) / dimBlock.x);

			VecDiffSH <<<dimGrid, dimBlock>>>(d_inFirst, d_inSecond, d_out, rows);

			cudaMemcpy(h_result, d_out, BYRESULT, cudaMemcpyDeviceToHost);

			cudaFree(d_inFirst);
			cudaFree(d_inSecond);
			cudaFree(d_out);
		}
	}
}






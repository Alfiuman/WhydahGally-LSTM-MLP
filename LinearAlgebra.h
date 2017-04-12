#pragma once

#include <vector>
#include <math.h>

#include "Definitions.h"
#include "Matrix.h"

#if CUDA
#include "LinearAlgebraCUDA.cuh"
#endif

namespace WhydahGally
{
	namespace Maths
	{
		//Tests needed to perform linear algebra operations safely.
		bool isVectMatrixMathCompatible(const std::vector<std::vector<float>>& x);

		bool areMatricesMultiplicable(const std::vector<std::vector<float>>& x, const std::vector<std::vector<float>>& y);
		template<typename T> bool areMatricesMultiplicable(const Matrix<T>& x, const Matrix<T>& y)
		{
			if (x.cols_ == y.rows_)
			{
				return 1;
			}
			else
			{
				return 0;
			}
		}

		bool areVectorsSameSize(const std::vector<float>& x, const std::vector<float>& y);
		template<typename T> bool areMatricesSameSize(const Matrix<T>& x, const Matrix<T>& y)
		{
			//Test that verifies if two matrices have the same structure.
			if ((x.rows_ == y.rows_ && x.cols_ == y.cols_) || (x.rows_ == y.cols_ && x.cols_ == y.rows_))
			{
				return 1;
			}
			else
			{
				return 0;
			}
		}

		template<typename T> bool areMatricesExactSameSize(const Matrix<T>& x, const Matrix<T>& y)
		{
			//Test that verifies if two matrices have the same number of rows and columns.
			if (x.rows_ == y.rows_ && x.cols_ == y.cols_)
			{
				return 1;
			}
			else
			{
				return 0;
			}
		}

		bool areMatrix1VectorSameSize(const std::vector<std::vector<float>>& x, const std::vector<float>& y);

		//Linear algebra operations.
		std::vector<float> matrixVectorProduct(const std::vector<std::vector<float>>& x, std::vector<float>& y);
		std::vector<std::vector<float>> matricesDotProduct(const std::vector<std::vector<float>>& x, const std::vector<std::vector<float>>& y);
		template<typename T> void matricesDotProduct(const Matrix<T>& x, const Matrix<T>& y, Matrix<T>* out, const int& parall, int resize = 1)
		{
			//Dot product between two matrices, with optional CUDA implementation.
			if (resize == 1)
			{
				out->resize(x.rows_, y.cols_);
			}
#if CUDA
			if (parall == 0)
			{
#endif
				for (int z = 0; z < y.cols_; z++)
				{
					for (int w = 0; w < x.rows_; w++)
					{
						float sum = 0.0f;

						for (int e = 0; e < x.cols_; e++)
						{
							sum += x.elements_[w * x.cols_ + e] * y.elements_[e * y.cols_ + z];
						}

						out->elements_[w * out->cols_ + z] = sum;
					}
				}
#if CUDA
			}
			else if (parall == 1)
			{
				matricesDotProductGPU(x.elements_, x.rows_, x.cols_, y.elements_, y.rows_, y.cols_, out->elements_);
			}
			else if (parall == 2 || parall == 11)
			{
				matricesDotProductGPUSH(x.elements_, x.rows_, x.cols_, y.elements_, y.rows_, y.cols_, out->elements_);
			}
#endif
		}
		
		std::vector<float> vectorsDifference(const std::vector<float>& x, const std::vector<float>& y);
		std::vector<float> matrixVectorDifference(const std::vector<std::vector<float>>& x, const std::vector<float>& y);
		template<typename T> void matricesDifference(const Matrix<T>& x, const Matrix<T>& y, Matrix<T>* out, const int& parall, int resize = 1)
		{
			//Difference between two matrices, with optional CUDA implementation.
			if (resize == 1)
			{
				out->resize(x.rows_, x.cols_);
			}
#if CUDA
			if (parall == 0 || parall == 11)
			{
#endif
				for (int i = 0; i < (out->rows_ * out->cols_); i++)
				{
					out->elements_[i] = x.elements_[i] - y.elements_[i];
				}
#if CUDA
			}
			else if (parall == 1)
			{
				vectorsDiffGPU(x.elements_, y.elements_, (x.rows_ * x.cols_), out->elements_);
			}
			else if (parall == 2)
			{
				vectorsDiffGPUSH(x.elements_, y.elements_, (x.rows_ * x.cols_), out->elements_);
			}
#endif
		}
		
		std::vector<std::vector<float>> outerProduct(const std::vector<float>& x, const std::vector<float>& y);
		template<typename T> void outerProduct(const Matrix<T>& x, const Matrix<T>& y, Matrix<T>* out, const int& parall, int resize = 1)
		{
			//Outer product between two matrices, with optional CUDA implementation.
			if (resize == 1)
			{
				out->resize(x.rows_, y.rows_);
			}
#if CUDA
			if (parall == 0 || parall == 11)
			{
#endif
				for (int w = 0; w < out->rows_; w++)
				{
					for (int z = 0; z < out->cols_; z++)
					{
						out->elements_[w * out->cols_ + z] = x.elements_[w] * y.elements_[z];
					}
				}
#if CUDA
			}
			else if (parall == 1)
			{
				outerProdGPU(x.elements_, x.rows_, y.elements_, y.rows_, out->elements_);
			}
			else if (parall == 2)
			{
				outerProdGPUSH(x.elements_, x.rows_, y.elements_, y.rows_, out->elements_);
			}
#endif
		}

		std::vector<std::vector<float>> transposeMatrix(const std::vector<std::vector<float>>& x);
		template<typename T> void transposeMatrix(const Matrix<T>& x, Matrix<T>* out, const int& parall, int resize = 1)
		{
			//Transpose a matrix, with optional CUDA implementation.
			if (resize == 1)
			{
				out->resize(x.cols_, x.rows_);
			}
#if CUDA
			if (parall == 0 || parall == 11)
			{
#endif
				for (int w = 0; w < x.rows_; w++)
				{
					for (int z = 0; z < x.cols_; z++)
					{
						out->elements_[z * out->cols_ + w] = x.elements_[w * x.cols_ + z];
					}
				}
#if CUDA
			}
			else if (parall == 1)
			{
				transposeGPU(x.elements_, x.rows_, x.cols_, out->elements_);
			}
			else if (parall == 2)
			{
				transposeGPUSH(x.elements_, x.rows_, x.cols_, out->elements_);
			}
#endif
		}
	}
}




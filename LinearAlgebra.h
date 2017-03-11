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
		bool isVectMatrixMathCompatible(const std::vector<std::vector<float>>& x);

		bool areMatricesMultiplicable(const std::vector<std::vector<float>>& x, const std::vector<std::vector<float>>& y);
		bool areMatricesMultiplicable(const Matrix& x, const Matrix& y);

		bool areVectorsSameSize(const std::vector<float>& x, const std::vector<float>& y);
		bool areMatricesSameSize(const Matrix& x, const Matrix& y);
		bool areMatricesExactSameSize(const Matrix& x, const Matrix& y);
		bool areMatrix1VectorSameSize(const std::vector<std::vector<float>>& x, const std::vector<float>& y);

		std::vector<float> matrixVectorProduct(const std::vector<std::vector<float>>& x, std::vector<float>& y);
		std::vector<std::vector<float>> matricesDotProduct(const std::vector<std::vector<float>>& x, const std::vector<std::vector<float>>& y);
		void matricesDotProduct(const Matrix& x, const Matrix& y, Matrix* out, const int& parall, int resize = 1);
		
		std::vector<float> vectorsDifference(const std::vector<float>& x, const std::vector<float>& y);
		std::vector<float> matrixVectorDifference(const std::vector<std::vector<float>>& x, const std::vector<float>& y);
		void matricesDifference(const Matrix& x, const Matrix& y, Matrix* out, const int& parall, int resize = 1);
		
		std::vector<std::vector<float>> outerProduct(const std::vector<float>& x, const std::vector<float>& y);
		void outerProduct(const Matrix& x, const Matrix& y, Matrix* out, const int& parall, int resize = 1);

		std::vector<std::vector<float>> transposeMatrix(const std::vector<std::vector<float>>& x);
		void transposeMatrix(const Matrix& x, Matrix* out, const int& parall, int resize = 1);
	}
}




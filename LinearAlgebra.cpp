#include "LinearAlgebra.h"

namespace WhydahGally
{
	namespace Maths
	{
		bool isVectMatrixMathCompatible(const std::vector<std::vector<float>>& x)
		{
			//Test that verifies if a vector is a proper matrix.
			int data = 1;
			bool out = 1;

			for (int i = 1; i < x.size(); ++i)
			{
				if (x[i - 1].size() == x[i].size())
				{
					data *= 1;
				}
				else
				{
					data *= 0;
				}
			}

			if (data = 1)
			{
				out = 1;
			}
			else
			{
				out = 0;
			}

			return out;
		}

		bool areMatricesMultiplicable(const std::vector<std::vector<float>>& x, const std::vector<std::vector<float>>& y)
		{
			if (isVectMatrixMathCompatible(x) && isVectMatrixMathCompatible(y) && x[0].size() == y.size())
			{
				return 1;
			}
			else
			{
				return 0;
			}
		}

		bool areVectorsSameSize(const std::vector<float>& x, const std::vector<float>& y)
		{
			if (x.size() == y.size())
			{
				return 1;
			}
			else
			{
				return 0;
			}
		}

		bool areMatrix1VectorSameSize(const std::vector<std::vector<float>>& x, const std::vector<float>& y)
		{
			if (x.size() == y.size())
			{
				return 1;
			}
			else
			{
				return 0;
			}
		}

		std::vector<float> matrixVectorProduct(const std::vector<std::vector<float>>& x, std::vector<float>& y)
		{
			//Multiplying a 2D vector by a 1D one.
			std::vector<float> data;
			data.resize(x.size());

			for (int w = 0; w < x.size(); w++)
			{
				float sum = 0.0;

				for (int e = 0; e < y.size(); e++)
				{
					sum += x[w][e] * y[e];
				}

				data[w] = sum;
			}

			return data;
		}

		std::vector<std::vector<float>> matricesDotProduct(const std::vector<std::vector<float>> &x, const std::vector<std::vector<float>> &y)
		{
			//Dot product between two 2D vectors.
			std::vector<std::vector<float>> data;
			data.resize(x.size(), std::vector<float>(y[0].size()));

			for (int z = 0; z < y[0].size(); z++)
			{
				for (int w = 0; w < x.size(); w++)
				{
					float sum = 0.0;

					for (int e = 0; e < x[0].size(); e++)
					{
						sum += x[w][e] * y[e][z];
					}

					data[w][z] = sum;
				}
			}

			return data;
		}

		std::vector<float> vectorsDifference(const std::vector<float>& x, const std::vector<float>& y)
		{
			//Difference between two vectors.
			std::vector<float> results;
			results.resize(x.size());

			for (int i = 0; i < x.size(); ++i)
			{
				results.at(i) = x[i] - y[i];
			}

			return results;
		}

		std::vector<float> matrixVectorDifference(const std::vector<std::vector<float>>& x, const std::vector<float>& y)
		{
			//Difference between a fake 2D vector and a 1D one.
			std::vector<float> results;
			results.resize(x.size());

			for (int i = 0; i < x.size(); ++i)
			{
				results.at(i) = x[i][0] - y[i];
			}

			return results;
		}

		std::vector<std::vector<float>> outerProduct(const std::vector<float>& x, const std::vector<float>& y)
		{
			//Outer product between two vectors.
			std::vector<std::vector<float>> results;
			results.resize(x.size(), std::vector<float>(y.size()));

			for (int w = 0; w < x.size(); w++)
			{
				for (int z = 0; z < y.size(); z++)
				{
					results.at(w).at(z) = x[w] * y[z];
				}
			}

			return results;
		}

		std::vector<std::vector<float>> transposeMatrix(const std::vector<std::vector<float>>& x)
		{
			//Transpose a 2D vector.
			std::vector<std::vector<float>> data;
			data.resize(x[0].size(), std::vector<float>(x.size()));

			for (int w = 0; w < x.size(); w++)
			{
				for (int z = 0; z < x[0].size(); z++)
				{
					data.at(z).at(w) = x[w][z];
				}
			}

			return data;
		}
	}
}
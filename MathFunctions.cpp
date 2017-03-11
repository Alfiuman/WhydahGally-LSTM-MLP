#include "MathFunctions.h"

namespace WhydahGally
{
	namespace Maths
	{
		float sigmoid(const float& x)
		{
			return 1 / (1 + exp(-x));
		}

		float derivativeSigmoid(const float& x)
		{
			return x * (1 - x);
		}

		float sign(const float& x)
		{
			if (x >= 0)
			{
				return 1.0f;
			}
			else
			{
				return -1.0f;
			}
		}

		float mean(const std::vector<float>& x)
		{
			float sum = 0.0f;
			for (int i = 0; i < x.size(); i++)
			{
				sum += x[i];
			}
			
			return sum / x.size();
		}

		float mean(const std::vector<std::vector<float>>& x)
		{
			float sum = 0.0f;
			int counter = 0;
			for (int i = 0; i < x.size(); i++)
			{
				for (int j = 0; j < x[i].size(); j++)
				{
					sum += x[i][j];
					counter++;
				}
			}

			return sum / counter;
		}

		float mean(const Matrix& x)
		{
			float sum = 0.0f;

			for (int i = 0; i < (x.rows_ * x.cols_); i++)
			{
				sum += x.elements_[i];
			}

			return sum / (x.rows_ * x.cols_);
		}

		float sum(const std::vector<float>& x)
		{
			float sum = 0.0f;

			for (int i = 0; i < x.size(); i++)
			{
				sum += x[i];
			}

			return sum;
		}

		float sum(const Matrix& x)
		{
			float sum = 0.0f;

			for (int i = 0; i < (x.rows_ * x.cols_); i++)
			{
				sum += x.elements_[i];
			}

			return sum;
		}

		float abs(const float& x)
		{
			if (x >= 0)
			{
				return x;
			}
			else
			{
				return -x;
			}
		}

		std::vector<float> abs(const std::vector<float>& x)
		{
			std::vector<float> results;
			results.resize(x.size());

			for (int i = 0; i < x.size(); i++)
			{
				results.at(i) = abs(x[i]);
			}

			return results;
		}

		std::vector<std::vector<float>> abs(const std::vector<std::vector<float>>& x)
		{
			std::vector<std::vector<float>> results;
			results.resize(x.size());

			for (int i = 0; i < x.size(); i++)
			{
				results[i].resize(x[i].size());
			}

			for (int i = 0; i < x.size(); i++)
			{
				for (int j = 0; j < x[i].size(); j++)
				{
					results.at(i).at(j) = abs(x[i][j]);
				}
			}

			return results;
		}

		float randNormalDistrib(const float& mean, const float& stdDev) //Using the Box-Muller method.
		{
			static bool nBool = 0;
			static float n = 0.0f;

			if (!nBool)
			{
				float x;
				float y;
				float z;

				do
				{
					x = 2.0f * rand() / RAND_MAX - 1.0f;
					y = 2.0f * rand() / RAND_MAX - 1.0f;
					z = pow(x, 2) + pow(y, 2);
				} 
				while (z > 1.0f || z == 0.0f);
				{
					float w = sqrt(-2.0f * log(z) / z);
					float m = x * w;

					n = y * w;

					float result = m * stdDev + mean;
					nBool = 1;

					return result;
				}
			}
			else
			{
				nBool = 0;

				return n * stdDev + mean;
			}
		}
	}
}
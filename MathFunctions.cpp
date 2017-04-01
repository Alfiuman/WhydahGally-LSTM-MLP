#include "MathFunctions.h"

namespace WhydahGally
{
	namespace Maths
	{
		//Mean.
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

		//Sum.
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

		//Absolute value.
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
	}
}
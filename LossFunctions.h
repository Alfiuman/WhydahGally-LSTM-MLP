#pragma once

#include <math.h>

#include "Definitions.h"
#include "Matrix.h"
#include "MathFunctions.h"
#include "LinearAlgebra.h"

namespace WhydahGally
{
	namespace Maths
	{
		//Different kinds of loss functions.
		template<typename T> T lossFunctSimple(const T& x, const T& y)
		{
			return x - y;
		}

		std::vector<float> lossFunctSimple(const std::vector<float>& x, const std::vector<float>& y);
		std::vector<float> lossFunctSimple(const std::vector<std::vector<float>>& x, const std::vector<float>& y);

		template<typename T> T lossFunctLog(const T& x, const T& y)
		{
			return (-((y * log(x)) + ((1 - y) * log(1 - x)))) * sign(x - y);
		}

		std::vector<float> lossFunctLog(const std::vector<float>& x, const std::vector<float>& y);
		std::vector<float> lossFunctLog(const std::vector<std::vector<float>>& x, const std::vector<float>& y);

		template<typename T> T lossFunctLogPow3(const T& x, const T& y)
		{
			return pow((-((y * log(x)) + ((1 - y) * log(1 - x)))) * sign(x - y), 3);
		}

		std::vector<float> lossFunctLogPow3(const std::vector<float>& x, const std::vector<float>& y);
		std::vector<float> lossFunctLogPow3(const std::vector<std::vector<float>>& x, const std::vector<float>& y);

		template<typename T> T lossFunctPow3(const T& x, const T& y)
		{
			return pow(x - y, 3);
		}

		std::vector<float> lossFunctPow3(const std::vector<float>& x, const std::vector<float>& y);
		std::vector<float> lossFunctPow3(const std::vector<std::vector<float>>& x, const std::vector<float>& y);

		template<typename T> T lossFunctPow3PLogPow3(const T& x, const T& y)
		{
			return pow(x - y, 3) + pow((-((y * log(x)) + ((1 - y) * log(1 - x)))) * sign(x - y), 3);
		}

		std::vector<float> lossFunctPow3PLogPow3(const std::vector<float>& x, const std::vector<float>& y);
		std::vector<float> lossFunctPow3PLogPow3(const std::vector<std::vector<float>>& x, const std::vector<float>& y);
	}
}





#pragma once

#include <math.h>
#include <vector>
#include <iostream>

#include "Definitions.h"
#include "Matrix.h"

namespace WhydahGally
{
	namespace Maths
	{
		float sigmoid(const float& x);
		float derivativeSigmoid(const float& x);

		float sign(const float& x);

		float mean(const std::vector<float>& x);
		float mean(const std::vector<std::vector<float>>& x);
		float mean(const Matrix& x);

		float sum(const std::vector<float>& x);
		float sum(const Matrix& x);

		float abs(const float& x);
		std::vector<float> abs(const std::vector<float>& x);
		std::vector<std::vector<float>> abs(const std::vector<std::vector<float>>& x);

		float randNormalDistrib(const float& mean, const float& stdDev);
	}
}


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
		float lossFunctSimple(const float& x, const float& y);
		std::vector<float> lossFunctSimple(const std::vector<float>& x, const std::vector<float>& y);
		std::vector<float> lossFunctSimple(const std::vector<std::vector<float>>& x, const std::vector<float>& y);

		float lossFunctLog(const float& x, const float& y);
		std::vector<float> lossFunctLog(const std::vector<float>& x, const std::vector<float>& y);
		std::vector<float> lossFunctLog(const std::vector<std::vector<float>>& x, const std::vector<float>& y);

		float lossFunctLogPow3(const float& x, const float& y);
		std::vector<float> lossFunctLogPow3(const std::vector<float>& x, const std::vector<float>& y);
		std::vector<float> lossFunctLogPow3(const std::vector<std::vector<float>>& x, const std::vector<float>& y);

		float lossFunctPow3(const float& x, const float& y);
		std::vector<float> lossFunctPow3(const std::vector<float>& x, const std::vector<float>& y);
		std::vector<float> lossFunctPow3(const std::vector<std::vector<float>>& x, const std::vector<float>& y);

		float lossFunctPow3PLogPow3(const float& x, const float& y);
		std::vector<float> lossFunctPow3PLogPow3(const std::vector<float>& x, const std::vector<float>& y);
		std::vector<float> lossFunctPow3PLogPow3(const std::vector<std::vector<float>>& x, const std::vector<float>& y);
	}
}





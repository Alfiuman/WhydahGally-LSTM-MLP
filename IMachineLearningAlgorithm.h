#pragma once

namespace WhydahGally
{
	namespace Base
	{
		class IMachineLearningAlgorithm
		{
		public:
			virtual ~IMachineLearningAlgorithm() {};

			virtual void importWeights() = 0;
			virtual void train() = 0;
			virtual void exportWeights() = 0;
			virtual void test() = 0;
			virtual void classify() = 0;
			virtual void computeStatistics() = 0;
		};
	}
}
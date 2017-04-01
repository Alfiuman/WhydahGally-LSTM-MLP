#pragma once

#include <random>
#include <cmath>

#include "Definitions.h"
#include "Matrix.h"
#include "Importer.h"
#include "LinearAlgebra.h"
#include "LossFunctions.h"
#include "IMachineLearningAlgorithm.h"

namespace WhydahGally
{
	namespace Base
	{
		//Collections of parameters for a Node.
		struct Parameters
		{
			int numMemCell_;
			int concatLen_;
			bool problem_;

			Matrix weightsG_;
			Matrix weightsI_;
			Matrix weightsF_;
			Matrix weightsO_;

			Matrix biasG_;
			Matrix biasI_;
			Matrix biasF_;
			Matrix biasO_;

			Matrix differWeightsG_;
			Matrix differWeightsI_;
			Matrix differWeightsF_;
			Matrix differWeightsO_;

			Matrix differBiasG_;
			Matrix differBiasI_;
			Matrix differBiasF_;
			Matrix differBiasO_;

			Parameters(const int& dimX, const int& numMemCell, const bool& importParam, const float& max, const float& min, const float& seedNo);
			~Parameters();

			void recomputeWeightsBias(const float& alpha);
			void importParameters();
		};

		//State of a Node.
		struct State
		{
			int numMemCell_;

			Matrix g_;
			Matrix i_;
			Matrix f_;
			Matrix o_;
			Matrix s_;
			Matrix h_;
			Matrix bottomDifferH_;
			Matrix bottomDifferS_;
			Matrix bottomDifferX_;

			State(const int& dimX, const int& numMemCell);
			~State();
		};

		//Node of the Artificial Neural Network.
		struct Node
		{
			Parameters* param_;
			State* state_;
			Matrix previousS_;
			Matrix previousH_;
			Matrix x_;
			Matrix xh_;

			Node(Parameters& parameters, State& state);
			~Node();

			void computeBottomData(const Matrix& x, const int& parall);
			void computeBottomData(const Matrix& x, const Matrix& prevS, const Matrix& prevH, const int& parall);
			void computeTopDiffer(const Matrix& topDiffH, const Matrix& topDiffS, const int& parall);
		};

		//Artificial Neural Network for time series analysis.
		class LongShortTermMemory : public IMachineLearningAlgorithm
		{
		private:
			int dimX_;
			int sizeX_;
			int numMemCell_;
			bool changedImporter_;
			float loss_;
			int historyLength_;
			float generalLoss_;
			Parameters* param_;
			Importer* importer_;
			Matrix predictions_;

			std::vector<Node*> nodeList_;
			std::vector<std::vector<float>> listX_;

		private:
			void computeLoss(const Matrix& listY, const int& lossFunct, const int& parall);
			void buildListX(const std::vector<float>& x, const int& parall);

		public:
			LongShortTermMemory(Importer& importer, const int& numMemCell, const bool& importParam, const float& max, const float& min, const int& seedNo);
			~LongShortTermMemory();

			void importWeights() override;
			void train() override;
			void train(const int& times, const int& view, const float& alpha, const bool& print, const int& lossFunct, const int& parall, const bool& exportParam);
			void exportWeights() override;
			void test() override;
			void test(const int& lossFunct, const int& parall);
			void classify() override;
			void classify(const int& parall);
			void computeStatistics() override;

			inline float getLoss() const { return loss_; }
			inline float getGeneralLoss() const { return generalLoss_; }
			inline Matrix getPredictions() const { return predictions_; }
		};
	}
}
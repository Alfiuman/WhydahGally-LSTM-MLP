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

			Matrix<float> weightsG_;
			Matrix<float> weightsI_;
			Matrix<float> weightsF_;
			Matrix<float> weightsO_;

			Matrix<float> biasG_;
			Matrix<float> biasI_;
			Matrix<float> biasF_;
			Matrix<float> biasO_;

			Matrix<float> differWeightsG_;
			Matrix<float> differWeightsI_;
			Matrix<float> differWeightsF_;
			Matrix<float> differWeightsO_;

			Matrix<float> differBiasG_;
			Matrix<float> differBiasI_;
			Matrix<float> differBiasF_;
			Matrix<float> differBiasO_;

			Parameters(const int& dimX, const int& numMemCell, const bool& importParam, const float& max, const float& min, const float& seedNo);
			~Parameters();

			void recomputeWeightsBias(const float& alpha);
			void importParameters();
		};

		//State of a Node.
		struct State
		{
			int numMemCell_;

			Matrix<float> g_;
			Matrix<float> i_;
			Matrix<float> f_;
			Matrix<float> o_;
			Matrix<float> s_;
			Matrix<float> h_;
			Matrix<float> bottomDifferH_;
			Matrix<float> bottomDifferS_;
			Matrix<float> bottomDifferX_;

			State(const int& dimX, const int& numMemCell);
			~State();
		};

		//Node of the Artificial Neural Network.
		struct Node
		{
			Parameters* param_;
			State* state_;
			Matrix<float> previousS_;
			Matrix<float> previousH_;
			Matrix<float> x_;
			Matrix<float> xh_;

			Node(Parameters& parameters, State& state);
			~Node();

			void computeBottomData(const Matrix<float>& x, const int& parall);
			void computeBottomData(const Matrix<float>& x, const Matrix<float>& prevS, const Matrix<float>& prevH, const int& parall);
			void computeTopDiffer(const Matrix<float>& topDiffH, const Matrix<float>& topDiffS, const int& parall);
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
			Matrix<float> predictions_;

			std::vector<Node*> nodeList_;
			std::vector<std::vector<float>> listX_;

		private:
			void computeLoss(const Matrix<float>& listY, const int& lossFunct, const int& parall);
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
			inline Matrix<float> getPredictions() const { return predictions_; }
		};
	}
}
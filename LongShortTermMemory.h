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

			Parameters(int dimX, int numMemCell, bool importParam, float max, float min, float seedNo);
			~Parameters();

			void recomputeWeightsBias(float alpha);
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

			State(int dimX, int numMemCell);
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

			void computeBottomData(const Matrix<float>& x, int parall);
			void computeBottomData(const Matrix<float>& x, const Matrix<float>& prevS, const Matrix<float>& prevH, int parall);
			void computeTopDiffer(const Matrix<float>& topDiffH, const Matrix<float>& topDiffS, int parall);
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
			void computeLoss(const Matrix<float>& listY, int lossFunct, int parall);
			void buildListX(const std::vector<float>& x, int parall);

		public:
			LongShortTermMemory(Importer& importer, int numMemCell, bool importParam, float max, float min, int seedNo);
			~LongShortTermMemory();

			void importWeights() override;
			void train() override;
			void train(int times, int view, float alpha, bool print, int lossFunct, int parall, bool exportParam);
			void exportWeights() override;
			void test() override;
			void test(int lossFunct, int parall);
			void classify() override;
			void classify(int parall);
			void computeStatistics() override;

			inline float getLoss() const { return loss_; }
			inline float getGeneralLoss() const { return generalLoss_; }
			inline Matrix<float> getPredictions() const { return predictions_; }
		};
	}
}
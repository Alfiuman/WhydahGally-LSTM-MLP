#pragma once

#include <random>

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
		struct DistribParamForMLP
		{
			//Parameters for the random distributions used to train the MLP.
			float mu_;
			float sigma_;
			int ranDistr_;
			float epsilon_;
			float muAlpha_;
			float sigmaAlpha_;
			float seedNo_;
		};

		class MultiLayerPerceptron : public IMachineLearningAlgorithm
		{
		protected:
			//Protected in order to be used by the MLPFast.
			float errorV_;
			float error_;
			float bestError_;
			float bestErrorExpl_;
			bool start_;
			int historyLength_;

			std::string importFileName_;
			std::string exportFileName_;

			Importer* importer_;

			std::vector<int> numNeurArr_;
			std::vector<float> lastLayerError_;
			std::vector<float> lastLayerErrorV_;
			std::vector<float> errorsToPlot_;
			std::vector<float> errorsToPlotExpl_;
			std::vector<std::vector<float>> lastLayerTop_;
			std::vector<std::vector<float>> results_;
			std::vector<std::vector<std::vector<float>>> weights_;
			std::vector<std::vector<std::vector<float>>> topWeights_;
			std::vector<std::vector<std::vector<float>>> layers_;
			std::vector<std::vector<std::vector<float>>> layerErrors_;
			std::vector<std::vector<std::vector<float>>> layerDeltas_;
			std::vector<std::vector<std::vector<float>>> layerDeltaUse_;

		private:
			void buildWeights(const float& limMin, const float& limMax, const float& seedNo);
			void calculateLayers();

		protected:
			void computeErrors(const int& counter, const int& lossFunction, const bool& plot, const bool& backpropagation);

		public:
			MultiLayerPerceptron(Importer& importer, const float& limMin, const float& limMax, const float& seedNo, const std::vector<int>& numNeurArr);
			virtual ~MultiLayerPerceptron();

			void importWeights() override;
			void importWeights(const short int& layer, const std::string& fileName = "");
			void exportWeights() override;
			void exportWeights(const short int& layer, const std::string& fileName = "");
			virtual void train() override;
			void train(DistribParamForMLP& distrParam, int ranges[3], int checkPoints[3], const int& lossFunction, const bool& plot, const bool& print);
			void test() override;
			void test(const int& lossFunction);
			void classify() override;
			void computeStatistics() override;

			//Getter functions.
			inline std::vector<std::vector<std::vector<float>>> getWeights() const { return topWeights_; }
			inline std::vector<std::vector<std::vector<float>>> getLayers() const { return layers_; }
			inline float getError() const { return error_; }
			inline float getErrorV() const { return errorV_; }
			inline std::vector<int> getNumNeur() const { return numNeurArr_; }
		};
	}
}
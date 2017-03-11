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
		class MultiLayerPerceptron : public IMachineLearningAlgorithm
		{
		protected:
			int numNeurLayers_;
			float errorV_;
			float error_;
			float bestError_;
			float bestErrorExpl_;
			bool start_;
			int historyLength_;

			std::string importFileName_;
			std::string exportFileName_;

			int* numNeurArr_;
			Importer* importer_;

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
			MultiLayerPerceptron(Importer& importer, const float& limMin, const float& limMax, const float& seedNo, const int& numNeur1, const int& numNeur2 = 0, const int& numNeur3 = 0, const int& numNeur4 = 0, const int& numNeur5 = 0, const int& numNeur6 = 0, const int& numNeur7 = 0, const int& numNeur8 = 0, const int& numNeur9 = 0, const int& numNeur10 = 0, const int& numNeur11 = 0, const int& numNeur12 = 0);
			virtual ~MultiLayerPerceptron();

			void importWeights() override;
			void importWeights(const short int& layer, const std::string& fileName = "");
			void exportWeights() override;
			void exportWeights(const short int& layer, const std::string& fileName = "");
			virtual void train() override;
			void train(const float& mu, const float& sigma, const int& ranDistr, const int& range1, const int& range2, const int& range3, const int& checkPoint1, const int& checkPoint2, const int& checkPoint3, const float& epsilon, const float& muAlpha, const float& sigmaAlpha, const int& lossFunction, const bool& plot, const bool& print, const float& seedNo);
			void test() override;
			void test(const int& lossFunction);
			void classify() override;
			void computeStatistics() override;

			inline std::vector<std::vector<std::vector<float>>> getWeights() const { return topWeights_; }
			inline std::vector<std::vector<std::vector<float>>> getLayers() const { return layers_; }
			inline float getError() const { return error_; }
			inline float getErrorV() const { return errorV_; }
		};
	}
}
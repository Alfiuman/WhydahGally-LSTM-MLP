#pragma once

#include "MultiLayerPerceptron.h"

namespace WhydahGally
{
	namespace Base
	{
		class MLPFast : public MultiLayerPerceptron
		{
		//Faster version of the MLP; it is faster thanks to the use of the matrices instead of the 2D vectors for the computations.
		private:
			std::vector<Matrix<float>> weightsF_;
			std::vector<Matrix<float>> layersF_;
			std::vector<Matrix<float>> layerDeltaUseF_;

		private:
			void calculateLayers(int parall);
			void computeErrors(int counter, int lossFunction, bool plot, bool backpropagation, int parall);
			
		public:
			MLPFast(Importer& importer, float limMin, float limMax, float seedNo, const std::vector<int>& numNeurArr);
			~MLPFast();

			void train() override;
			void train(DistribParamForMLP& distrParam, int ranges[3], int checkPoints[3], int lossFunction, bool plot, bool print, int parall);
		};
	}
}






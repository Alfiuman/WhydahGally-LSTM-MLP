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
			std::vector<Matrix> weightsF_;
			std::vector<Matrix> layersF_;
			std::vector<Matrix> layerDeltaUseF_;

		private:
			void calculateLayers(const int& parall);
			void computeErrors(const int& counter, const int& lossFunction, const bool& plot, const bool& backpropagation, const int& parall);
			
		public:
			MLPFast(Importer& importer, const float& limMin, const float& limMax, const float& seedNo, const int& numNeur1, const int& numNeur2 = 0, const int& numNeur3 = 0, const int& numNeur4 = 0, const int& numNeur5 = 0, const int& numNeur6 = 0, const int& numNeur7 = 0, const int& numNeur8 = 0, const int& numNeur9 = 0, const int& numNeur10 = 0, const int& numNeur11 = 0, const int& numNeur12 = 0);
			~MLPFast();

			void train() override;
			void train(const float& mu, const float& sigma, const int& ranDistr, const int& range1, const int& range2, const int& range3, const int& checkPoint1, const int& checkPoint2, const int& checkPoint3, const float& epsilon, const float& muAlpha, const float& sigmaAlpha, const int& lossFunction, const bool& plot, const bool& print, const float& seedNo, const int& parall);
		};
	}
}






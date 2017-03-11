#include "MLPFast.h"

namespace WhydahGally
{
	namespace Base
	{
		MLPFast::MLPFast(Importer& importer, const float& limMin, const float& limMax, const float& seedNo, const int& numNeur1, const int& numNeur2, const int& numNeur3, const int& numNeur4, const int& numNeur5, const int& numNeur6, const int& numNeur7, const int& numNeur8, const int& numNeur9, const int& numNeur10, const int& numNeur11, const int& numNeur12) : MultiLayerPerceptron(importer, limMin, limMax, seedNo, numNeur1, numNeur2, numNeur3, numNeur4, numNeur5, numNeur6, numNeur7, numNeur8, numNeur9, numNeur10, numNeur11, numNeur12)
		{

		}

		MLPFast::~MLPFast()
		{
			
		}

		void MLPFast::calculateLayers(const int& parall)
		{
			for (short int i = 0; i < numNeurLayers_; i++)
			{
				for (int j = 0; j < layersF_[i + 1].rows_; j++)
				{
					Matrix results(1);
					Maths::matricesDotProduct(layersF_[i], weightsF_[i], &results, parall);

					for (int k = 0; k < layersF_[i + 1].cols_ - 1; k++)
					{
						layersF_[i + 1].elements_[j * layersF_[i + 1].cols_ + k] = Maths::sigmoid(results.elements_[j * results.cols_ + k]);
					}
				}
			}

			Matrix results(1);

			Maths::matricesDotProduct(layersF_[layersF_.size() - 2], weightsF_[weights_.size() - 1], &results, parall);

			for (int i = 0; i < layersF_[layersF_.size() - 1].rows_; i++)
			{
				layersF_[layersF_.size() - 1].elements_[i * layersF_[layersF_.size() - 1].cols_ + 0] = Maths::sigmoid(results.elements_[i * results.cols_ + 0]);

				if (layersF_[layersF_.size() - 1].elements_[i * layersF_[layersF_.size() - 1].cols_ + 0] >= 1.0f)
				{
					layersF_[layersF_.size() - 1].elements_[i * layersF_[layersF_.size() - 1].cols_ + 0] = 0.99999f;
				}
				if (layersF_[layersF_.size() - 1].elements_[i * layersF_[layersF_.size() - 1].cols_ + 0] <= 0.0f)
				{
					layersF_[layersF_.size() - 1].elements_[i * layersF_[layersF_.size() - 1].cols_ + 0] = 0.00001f;
				}
			}
		}

		void MLPFast::computeErrors(const int& counter, const int& lossFunction, const bool& plot, const bool& backpropagation, const int& parall)
		{
			Matrix results(1);
			Maths::matricesDifference(layersF_[layers_.size() - 1], importer_->getYMat(), &results, parall);

			for (int i = 0; i < importer_->getYMat().rows_; i++)
			{
				lastLayerErrorV_[i] = results.elements_[i];
			}
			
			if (backpropagation == 0)
			{
				if (counter == 0 && start_ == 0)
				{
					bestErrorExpl_ = Maths::sum(Maths::abs(lastLayerErrorV_));
				}

				start_ = 1;

				if (lossFunction == 0)
				{
					for (int i = 0; i < importer_->getYMat().rows_; i++)
					{
						lastLayerError_[i] = Maths::lossFunctSimple(layersF_[layers_.size() - 1].elements_[i], importer_->getYMat().elements_[i]);
					}
				}
				else if (lossFunction == 1)
				{
					for (int i = 0; i < importer_->getYMat().rows_; i++)
					{
						lastLayerError_[i] = Maths::lossFunctLog(layersF_[layers_.size() - 1].elements_[i], importer_->getYMat().elements_[i]);
					}
				}
				else if (lossFunction == 2)
				{
					for (int i = 0; i < importer_->getYMat().rows_; i++)
					{
						lastLayerError_[i] = Maths::lossFunctLogPow3(layersF_[layers_.size() - 1].elements_[i], importer_->getYMat().elements_[i]);
					}
				}
				else if (lossFunction == 3)
				{
					for (int i = 0; i < importer_->getYMat().rows_; i++)
					{
						lastLayerError_[i] = Maths::lossFunctPow3(layersF_[layers_.size() - 1].elements_[i], importer_->getYMat().elements_[i]);
					}
				}
				else if (lossFunction == 4)
				{
					for (int i = 0; i < importer_->getYMat().rows_; i++)
					{
						lastLayerError_[i] = Maths::lossFunctPow3PLogPow3(layersF_[layers_.size() - 1].elements_[i], importer_->getYMat().elements_[i]);
					}
				}

				if (Maths::mean(Maths::abs(lastLayerError_)) < bestError_)
				{
					for (int i = 0; i < weightsF_.size(); i++)
					{
						for (int j = 0; j < weightsF_[i].rows_; j++)
						{
							for (int k = 0; k < weightsF_[i].cols_; k++)
							{
								topWeights_[i][j][k] = weightsF_[i].elements_[j * weightsF_[i].cols_ + k];
							}
						}
					}

					bestError_ = Maths::mean(Maths::abs(lastLayerError_));
					bestErrorExpl_ = Maths::mean(Maths::abs(lastLayerErrorV_));
					lastLayerTop_ = layers_[layers_.size() - 1];
				}

				if (plot == 1)
				{
					errorsToPlot_.push_back(bestError_);
					errorsToPlotExpl_.push_back(bestErrorExpl_);
				}
			}
			else
			{
				if (counter == 0 && start_ == 0)
				{
					bestErrorExpl_ = Maths::mean(Maths::abs(lastLayerErrorV_));
				}

				start_ = 1;

				if (lossFunction == 0)
				{
					for (int i = 0; i < layerErrors_[layerErrors_.size() - 1].size(); i++)
					{
						layerErrors_[layerErrors_.size() - 1][i][0] = Maths::lossFunctSimple(layersF_[layers_.size() - 1].elements_[i], importer_->getYMat().elements_[i]);
					}
				}
				else if (lossFunction == 1)
				{
					for (int i = 0; i < layerErrors_[layerErrors_.size() - 1].size(); i++)
					{
						layerErrors_[layerErrors_.size() - 1][i][0] = Maths::lossFunctLog(layersF_[layers_.size() - 1].elements_[i], importer_->getYMat().elements_[i]);
					}
				}
				else if (lossFunction == 2)
				{
					for (int i = 0; i < layerErrors_[layerErrors_.size() - 1].size(); i++)
					{
						layerErrors_[layerErrors_.size() - 1][i][0] = Maths::lossFunctLogPow3(layersF_[layers_.size() - 1].elements_[i], importer_->getYMat().elements_[i]);
					}
				}
				else if (lossFunction == 3)
				{
					for (int i = 0; i < layerErrors_[layerErrors_.size() - 1].size(); i++)
					{
						layerErrors_[layerErrors_.size() - 1][i][0] = Maths::lossFunctPow3(layersF_[layers_.size() - 1].elements_[i], importer_->getYMat().elements_[i]);
					}
				}
				else if (lossFunction == 4)
				{
					for (int i = 0; i < layerErrors_[layerErrors_.size() - 1].size(); i++)
					{
						layerErrors_[layerErrors_.size() - 1][i][0] = Maths::lossFunctPow3PLogPow3(layersF_[layers_.size() - 1].elements_[i], importer_->getYMat().elements_[i]);
					}
				}

				error_ = Maths::mean(Maths::abs(layerErrors_[layerErrors_.size() - 1]));
				errorV_ = Maths::mean(Maths::abs(lastLayerErrorV_));

				if (plot == 1)
				{
					errorsToPlot_.push_back(error_);
					errorsToPlotExpl_.push_back(errorV_);
				}
			}
		}

		void MLPFast::train()
		{
			train(-10.0f, 10.0f, 0, 1000, 1000, 25000, 100, 100, 100, 0.05f, 0.4f, 0.1f, LOSSFUNCTSIMPLE, 0, 1, 0, 0);
		}

		void MLPFast::train(const float& mu, const float& sigma, const int& ranDistr, const int& range1, const int& range2, const int& range3, const int& checkPoint1, const int& checkPoint2, const int& checkPoint3, const float& epsilon, const float& muAlpha, const float& sigmaAlpha, const int& lossFunction, const bool& plot, const bool& print, const float& seedNo, const int& parall)
		{
			for (int i = 0; i < weights_.size(); i++)
			{
				weightsF_.push_back(Matrix(weights_[i].size(), weights_[i][0].size()));
			}

			for (int i = 0; i < layers_.size(); i++)
			{
				layersF_.push_back(Matrix(layers_[i].size(), layers_[i][0].size()));
			}

			for (int i = 0; i < layerDeltaUse_.size(); i++)
			{
				layerDeltaUseF_.push_back(Matrix(layerDeltaUse_[i].size(), layerDeltaUse_[i][0].size()));
			}

			for (int i = 0; i < weightsF_.size(); i++)
			{
				for (int j = 0; j < weightsF_[i].rows_; j++)
				{
					for (int k = 0; k < weightsF_[i].cols_; k++)
					{
						weightsF_[i].elements_[j * weightsF_[i].cols_ + k] = weights_[i][j][k];
					}
				}
			}

			for (int i = 0; i < layersF_.size(); i++)
			{
				for (int j = 0; j < layersF_[i].rows_; j++)
				{
					for (int k = 0; k < layersF_[i].cols_; k++)
					{
						layersF_[i].elements_[j * layersF_[i].cols_ + k] = layers_[i][j][k];
					}
				}
			}

			for (int i = 0; i < layerDeltaUseF_.size(); i++)
			{
				for (int j = 0; j < layerDeltaUseF_[i].rows_; j++)
				{
					for (int k = 0; k < layerDeltaUseF_[i].cols_; k++)
					{
						layerDeltaUseF_[i].elements_[j * layerDeltaUseF_[i].cols_ + k] = layerDeltaUse_[i][j][k];
					}
				}
			}

			std::vector<std::vector<std::vector<float>>> results;

			results.resize(weightsF_.size());

			for (short int i = 0; i < weightsF_.size(); i++)
			{
				results[i].resize(weightsF_[i].rows_);

				for (int j = 0; j < weightsF_[i].rows_; j++)
				{
					results[i][j].resize(weightsF_[i].cols_);
				}
			}

			srand(seedNo);

			bool fase1 = 0;
			bool fase2 = 0;
			bool fase3 = 0;
			bool backpropagation = 0;
			bool start_ = 0;

			if (range1 > -1)
			{
				fase1 = 1;
			}

			if (range2 > -1)
			{
				fase2 = 1;
			}

			if (range3 > -1)
			{
				fase3 = 1;
			}

			for (int r = 0; r <= range1; r++)
			{
				backpropagation = 0;

				calculateLayers(parall);
				computeErrors(r, lossFunction, plot, backpropagation, parall);

				if (r % checkPoint1 == 0 && print == 1)
				{
					PRINT("Error after " << r << " 1st iterations: " << bestErrorExpl_ << + " or FunctErr: " << bestError_ << "\n");
				}

				if (ranDistr == 0)
				{
					for (int i = 0; i < weightsF_.size(); i++)
					{
						for (int j = 0; j < weightsF_[i].rows_; j++)
						{
							for (int k = 0; k < weightsF_[i].cols_; k++)
							{
								weightsF_[i].elements_[j * weightsF_[i].cols_ + k] = ((sigma - mu) * static_cast <float> (rand()) / static_cast <float> (RAND_MAX)) + mu; //sigma = limMax; mu = limMin;
							}
						}
					}
				}
				else if (ranDistr == 1)
				{
					for (int i = 0; i < weightsF_.size(); i++)
					{
						for (int j = 0; j < weightsF_[i].rows_; j++)
						{
							for (int k = 0; k < weightsF_[i].cols_; k++)
							{
								weightsF_[i].elements_[j * weightsF_[i].cols_ + k] = Maths::randNormalDistrib(mu, sigma);
							}
						}
					}
				}
				else
				{
					PRINT("Wrong number for random distribution.\n");
				}
			}

			if (fase1 == 1)
			{
				for (int i = 0; i < weightsF_.size(); i++)
				{
					for (int j = 0; j < weightsF_[i].rows_; j++)
					{
						for (int k = 0; k < weightsF_[i].cols_; k++)
						{
							weightsF_[i].elements_[j * weightsF_[i].cols_ + k] = topWeights_[i][j][k];
						}
					}
				}
			}

			for (int r = 0; r <= range2; r++)
			{
				backpropagation = 0;

				calculateLayers(parall);
				computeErrors(r, lossFunction, plot, backpropagation, parall);

				if (r % checkPoint2 == 0 && print == 1)
				{
					PRINT("Error after " + std::to_string(r) + " 2nd iterations: " + std::to_string(bestErrorExpl_) + " or FunctErr: " + std::to_string(bestError_) << "\n");
				}

				for (int i = 0; i < weightsF_.size(); i++)
				{
					for (int j = 0; j < weightsF_[i].rows_; j++)
					{
						for (int k = 0; k < weightsF_[i].cols_; k++)
						{
							weightsF_[i].elements_[j * weightsF_[i].cols_ + k] = ((epsilon - (1 - (epsilon - 1))) * static_cast <float> (rand()) / static_cast <float> (RAND_MAX)) + (1 - (epsilon - 1)) * topWeights_.at(i).at(j).at(k);
						}
					}
				}
			}

			if (fase2 == 1)
			{
				for (int i = 0; i < weightsF_.size(); i++)
				{
					for (int j = 0; j < weightsF_[i].rows_; j++)
					{
						for (int k = 0; k < weightsF_[i].cols_; k++)
						{
							weightsF_[i].elements_[j * weightsF_[i].cols_ + k] = topWeights_[i][j][k];
						}
					}
				}
			}

			float alpha = 0.0;

			for (int r = 0; r <= range3; r++)
			{
				backpropagation = 1;
				alpha = Maths::randNormalDistrib(muAlpha, sigmaAlpha);

				calculateLayers(parall);
				computeErrors(r, lossFunction, plot, backpropagation, parall);

				if (r % checkPoint3 == 0 && print == 1)
				{
					PRINT("Error after " << r << " 3rd iterations: " << error_ << " or General Error: " << errorV_ << "\n");
				}

				if (r != range3)
				{
					for (int i = 0; i < layerDeltas_[layerDeltas_.size() - 1].size(); i++)
					{
						layerDeltas_[layerDeltas_.size() - 1][i][0] = layerErrors_[layerErrors_.size() - 1][i][0] * Maths::derivativeSigmoid(layersF_[layersF_.size() - 1].elements_[i * layersF_[layersF_.size() - 1].cols_ + 0]);
						layerDeltaUseF_[layerDeltaUseF_.size() - 1].elements_[i * layerDeltaUseF_[layerDeltaUseF_.size() - 1].cols_ + 0] = layerDeltas_[layerDeltas_.size() - 1][i][0];
					}

					for (short int i = layerDeltas_.size() - 2; i > 0; i--)
					{
						Matrix resultTr(1);
						Matrix resultProd(1);
						Maths::transposeMatrix(weightsF_[i], &resultTr, parall);
						Maths::matricesDotProduct(layerDeltaUseF_[i + 1], resultTr, &resultProd, parall);

						for (int j = 0; j < resultProd.rows_; j++)
						{
							for (int k = 0; k < resultProd.cols_; k++)
							{
								layerErrors_[i][j][k] = resultProd.elements_[j * resultProd.cols_ + k];
							}
						}

						for (int j = 0; j < layersF_[i].rows_; j++)
						{
							for (int k = 0; k < layersF_[i].cols_; k++)
							{
								layerDeltas_[i][j][k] = layerErrors_[i][j][k] * Maths::derivativeSigmoid(layersF_[i].elements_[j * layersF_[i].cols_ + k]);
								if (k != layersF_[i].cols_ - 1)
								{
									layerDeltaUseF_[i].elements_[j * layerDeltaUseF_[i].cols_ + k] = layerDeltas_[i][j][k];
								}
							}
						}
					}

					for (short int i = weightsF_.size() - 1; i > -1; i--)
					{
						Matrix resultTr(1);
						Matrix resultProd(1);
						Maths::transposeMatrix(layersF_[i], &resultTr, parall);
						Maths::matricesDotProduct(resultTr, layerDeltaUseF_[i + 1], &resultProd, parall);

						for (int j = 0; j < resultProd.rows_; j++)
						{
							for (int k = 0; k < resultProd.cols_; k++)
							{
								results[i][j][k] = resultProd.elements_[j * resultProd.cols_ + k];
							}
						}
						

						for (int j = 0; j < weightsF_[i].rows_; j++)
						{
							for (int k = 0; k < weightsF_[i].cols_; k++)
							{
								weightsF_[i].elements_[j * weightsF_[i].cols_ + k] -= alpha * results[i][j][k];
							}
						}
					}
				}
			}

			if (fase3 == 1)
			{
				for (int i = 0; i < weightsF_.size(); i++)
				{
					for (int j = 0; j < weightsF_[i].rows_; j++)
					{
						for (int k = 0; k < weightsF_[i].cols_; k++)
						{
							topWeights_[i][j][k] = weightsF_[i].elements_[j * weightsF_[i].cols_ + k];
						}
					}
				}
			}

			for (int i = 0; i < layersF_.size(); i++)
			{
				for (int j = 0; j < layersF_[i].rows_; j++)
				{
					for (int k = 0; k < layersF_[i].cols_; k++)
					{
						layers_[i][j][k] = layersF_[i].elements_[j * layersF_[i].cols_ + k];
					}
				}
			}
		}
	}
}








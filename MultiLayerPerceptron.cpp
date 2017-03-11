#include "MultiLayerPerceptron.h"

namespace WhydahGally
{
	namespace Base
	{
		MultiLayerPerceptron::MultiLayerPerceptron(Importer& importer, const float& limMin, const float& limMax, const float& seedNo, const int& numNeur1, const int& numNeur2, const int& numNeur3, const int& numNeur4, const int& numNeur5, const int& numNeur6, const int& numNeur7, const int& numNeur8, const int& numNeur9, const int& numNeur10, const int& numNeur11, const int& numNeur12)
			: error_(10000000.1f), errorV_(10000000.1f), bestError_(10000000.1f), bestErrorExpl_(10000000.1f), importFileName_(""), start_(0), importer_(&importer), historyLength_(importer.getHistoryLength())
		{
			if (numNeur1 == 0)
			{
				PRINT("A MLP needs at least one layer of neurons.\n");
			}
			else if (numNeur2 == 0)
			{
				numNeurArr_ = new  int[1]{ numNeur1 };
				numNeurLayers_ = 1;
			}
			else if (numNeur3 == 0)
			{
				numNeurArr_ = new  int[2]{ numNeur1, numNeur2 };
				numNeurLayers_ = 2;
			}
			else if (numNeur4 == 0)
			{
				numNeurArr_ = new  int[3]{ numNeur1, numNeur2, numNeur3 };
				numNeurLayers_ = 3;
			}
			else if (numNeur5 == 0)
			{
				numNeurArr_ = new  int[4]{ numNeur1, numNeur2, numNeur3, numNeur4 };
				numNeurLayers_ = 4;
			}
			else if (numNeur6 == 0)
			{
				numNeurArr_ = new  int[5]{ numNeur1, numNeur2, numNeur3, numNeur4, numNeur5 };
				numNeurLayers_ = 5;
			}
			else if (numNeur7 == 0)
			{
				numNeurArr_ = new  int[6]{ numNeur1, numNeur2, numNeur3, numNeur4, numNeur5, numNeur6 };
				numNeurLayers_ = 6;
			}
			else if (numNeur8 == 0)
			{
				numNeurArr_ = new  int[7]{ numNeur1, numNeur2, numNeur3, numNeur4, numNeur5, numNeur6, numNeur7 };
				numNeurLayers_ = 7;
			}
			else if (numNeur9 == 0)
			{
				numNeurArr_ = new  int[8]{ numNeur1, numNeur2, numNeur3, numNeur4, numNeur5, numNeur6, numNeur7, numNeur8 };
				numNeurLayers_ = 8;
			}
			else if (numNeur10 == 0)
			{
				numNeurArr_ = new  int[9]{ numNeur1, numNeur2, numNeur3, numNeur4, numNeur5, numNeur6, numNeur7, numNeur8, numNeur9 };
				numNeurLayers_ = 9;
			}
			else if (numNeur11 == 0)
			{
				numNeurArr_ = new  int[10]{ numNeur1, numNeur2, numNeur3, numNeur4, numNeur5, numNeur6, numNeur7, numNeur8, numNeur9, numNeur10 };
				numNeurLayers_ = 10;
			}
			else if (numNeur12 == 0)
			{
				numNeurArr_ = new  int[11]{ numNeur1, numNeur2, numNeur3, numNeur4, numNeur5, numNeur6, numNeur7, numNeur8, numNeur9, numNeur10, numNeur11 };
				numNeurLayers_ = 11;
			}
			else
			{
				numNeurArr_ = new  int[12]{ numNeur1, numNeur2, numNeur3, numNeur4, numNeur5, numNeur6, numNeur7, numNeur8, numNeur9, numNeur10, numNeur11, numNeur12 };
				numNeurLayers_ = 12;
			}

			buildWeights(limMin, limMax, seedNo);
		}

		MultiLayerPerceptron::~MultiLayerPerceptron()
		{
			delete[] numNeurArr_;
		}

		void MultiLayerPerceptron::buildWeights(const float& limMin, const float& limMax, const float& seedNo)
		{
			weights_.resize(numNeurLayers_ + 1);
			topWeights_.resize(numNeurLayers_ + 1);
			layers_.resize(numNeurLayers_ + 2);

			weights_[0].resize(importer_->getX()[0].size() + 1, std::vector<float>(numNeurArr_[0]));
			topWeights_[0].resize(importer_->getX()[0].size() + 1, std::vector<float>(numNeurArr_[0]));

			if (numNeurLayers_ != 1)
			{
				for (short int i = 0; i < numNeurLayers_ - 1; i++)
				{
					weights_[i + 1].resize(numNeurArr_[i] + 1, std::vector<float>(numNeurArr_[i + 1]));
					topWeights_[i + 1].resize(numNeurArr_[i] + 1, std::vector<float>(numNeurArr_[i + 1]));
				}
			}

			weights_[numNeurLayers_].resize(numNeurArr_[numNeurLayers_ - 1] + 1, std::vector<float>(1));
			topWeights_[numNeurLayers_].resize(numNeurArr_[numNeurLayers_ - 1] + 1, std::vector<float>(1));

			srand(seedNo);

			for (short int i = 0; i < weights_.size(); i++)
			{
				for (int j = 0; j < weights_[i].size(); j++)
				{
					for (int k = 0; k < weights_[i][j].size(); k++)
					{
						weights_.at(i).at(j).at(k) = ((limMax - limMin) * static_cast <float> (rand()) / static_cast <float> (RAND_MAX)) + limMin;
						topWeights_.at(i).at(j).at(k) = weights_[i][j][k];
					}
				}
			}

			layers_[0].resize(importer_->getX().size(), std::vector<float>(importer_->getX()[0].size() + 1));

			for (short int i = 1; i <= numNeurLayers_; i++)
			{
				layers_[i].resize(importer_->getX().size(), std::vector<float>(numNeurArr_[i - 1] + 1));
			}

			layers_[layers_.size() - 1].resize(importer_->getX().size(), std::vector<float>(1));

			for (short int i = 0; i <= numNeurLayers_; i++)
			{
				for (int j = 0; j < importer_->getX().size(); j++)
				{
					layers_.at(i).at(j).at(layers_[i][j].size() - 1) = importer_->getBias(j);
				}
			}

			for (int i = 0; i < importer_->getX().size(); i++)
			{
				for (int j = 0; j < importer_->getX()[0].size(); j++)
				{
					layers_.at(0).at(i).at(j) = importer_->getX()[i][j];
				}
			}

			lastLayerErrorV_.resize(importer_->getY().size());
			lastLayerError_.resize(lastLayerErrorV_.size());
			lastLayerTop_.resize(layers_[layers_.size() - 1].size(), std::vector<float>(1));

			layerDeltas_.resize(layers_.size());
			layerErrors_.resize(layers_.size());
			layerDeltaUse_.resize(layers_.size());

			for (int i = 0; i < layers_.size(); i++)
			{
				layerDeltas_[i].resize(layers_[i].size());
				layerErrors_[i].resize(layers_[i].size());
				layerDeltaUse_[i].resize(layers_[i].size());

				for (int j = 0; j < layers_[i].size(); j++)
				{
					layerDeltas_[i][j].resize(layers_[i][j].size());
					layerErrors_[i][j].resize(layers_[i][j].size());

					if (i == layers_.size() - 1)
					{
						layerDeltaUse_[i][j].resize(layers_[i][j].size());
					}
					else
					{
						layerDeltaUse_[i][j].resize(layers_[i][j].size() - 1);
					}
				}
			}

			results_.resize(importer_->getY().size(), std::vector<float>(3));

			for (int i = 0; i < importer_->getY().size(); i++)
			{
				results_[i][0] = importer_->getY()[i];
			}
		}

		void MultiLayerPerceptron::importWeights()
		{
			try
			{
				for (short int i = 0; i < numNeurLayers_ + 1; i++)
				{
					std::ifstream file;

					file.open("weights" + std::to_string(i) + ".txt");

					if (file)
					{
						for (int j = 0; j < weights_[i].size(); j++)
						{
							for (int k = 0; k < weights_[i][j].size(); k++)
							{
								file >> weights_.at(i).at(j).at(k);
								topWeights_.at(i).at(j).at(k) = weights_.at(i).at(j).at(k);
							}
						}

						file.close();
					}
					else
					{
						PRINT("Weights not found.\n");
					}
				}
			}
			catch (const std::out_of_range& e)
			{
				PRINT("Out of range error in Weights import.\n");
			}
		}

		void MultiLayerPerceptron::importWeights(const short int& layer, const std::string& fileName)
		{
			std::ifstream file;

			importFileName_ = fileName;

			if (importFileName_ == "")
			{
				PRINT("Please enter the File Name: \n");
				getline(std::cin, importFileName_);
			}

			file.open(importFileName_ + ".txt");

			if (file)
			{
				try
				{
					for (int j = 0; j < weights_[layer].size(); j++)
					{
						for (int k = 0; k < weights_[layer][j].size(); k++)
						{
							file >> weights_.at(layer).at(j).at(k);
							topWeights_.at(layer).at(j).at(k) = weights_.at(layer).at(j).at(k);
						}
					}
				}
				catch (const std::out_of_range& e)
				{
					PRINT("Out of range error in Weights import.\n");
				}

				file.close();
			}
			else
			{
				PRINT("Weights not found.\n");
			}
		}

		void MultiLayerPerceptron::exportWeights()
		{
			for (short int i = 0; i < numNeurLayers_ + 1; i++)
			{
				std::ofstream file;

				file.open("weights" + std::to_string(i) + ".txt");

				if (file)
				{
					for (int j = 0; j < topWeights_[i].size(); j++)
					{
						for (int k = 0; k < topWeights_[i][j].size(); k++)
						{
							file << topWeights_[i][j][k];
							file << "\t";
						}

						file << "\n";
					}

					file.close();
				}
				else
				{
					PRINT("Problem in opening the file.\n");
				}
			}
		}

		void MultiLayerPerceptron::exportWeights(const short int& layer, const std::string& fileName)
		{
			if (layer < weights_.size())
			{
				std::ofstream file;

				exportFileName_ = fileName;

				if (exportFileName_ == "")
				{
					PRINT("Please enter the File Name for the " + std::to_string(layer) + " layer of weights.\n");
					getline(std::cin, exportFileName_);
				}

				file.open(exportFileName_ + ".txt");

				if (file)
				{
					for (int j = 0; j < topWeights_[layer].size(); j++)
					{
						for (int k = 0; k < topWeights_[layer][j].size(); k++)
						{
							file << topWeights_[layer][j][k];
							file << "\t";
						}

						file << "\n";
					}

					file.close();
				}
				else
				{
					PRINT("Problem in opening the file.\n");
				}
			}
			else
			{
				PRINT("The MLP used has only " + std::to_string(weights_.size()) + " layers of weights and they start from 0.\n");
			}
		}

		void MultiLayerPerceptron::train()
		{
			train(-10.0f, 10.0f, 0, 1000, 1000, 25000, 100, 100, 100, 0.05f, 0.4f, 0.1f, LOSSFUNCTSIMPLE, 0, 1, 0);
		}

		void MultiLayerPerceptron::train(const float& mu, const float& sigma, const int& ranDistr, const int& range1, const int& range2, const int& range3, const int& checkPoint1, const int& checkPoint2, const int& checkPoint3, const float& epsilon, const float& muAlpha, const float& sigmaAlpha, const int& lossFunction, const bool& plot, const bool& print, const float& seedNo)
		{
			std::vector<std::vector<std::vector<float>>> results;

			results.resize(weights_.size());

			for (short int i = 0; i < weights_.size(); i++)
			{
				results[i].resize(weights_[i].size());

				for (int j = 0; j < weights_[i].size(); j++)
				{
					results[i][j].resize(weights_[i][j].size());
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

				calculateLayers();
				computeErrors(r, lossFunction, plot, backpropagation);

				if (r % checkPoint1 == 0 && print == 1)
				{
					PRINT("Error after " + std::to_string(r) + " 1st iterations: " + std::to_string(bestErrorExpl_) + " or FunctErr: " + std::to_string(bestError_) << "\n");
				}

				if (ranDistr == 0)
				{
					for (int i = 0; i < weights_.size(); i++)
					{
						for (int j = 0; j < weights_[i].size(); j++)
						{
							for (int k = 0; k < weights_[i][j].size(); k++)
							{
								weights_.at(i).at(j).at(k) = ((sigma - mu) * static_cast <float> (rand()) / static_cast <float> (RAND_MAX)) + mu; //sigma = limMax; mu = limMin;
							}
						}
					}
				}
				else if (ranDistr == 1)
				{
					for (int i = 0; i < weights_.size(); i++)
					{
						for (int j = 0; j < weights_[i].size(); j++)
						{
							for (int k = 0; k < weights_[i][j].size(); k++)
							{
								weights_.at(i).at(j).at(k) = Maths::randNormalDistrib(mu, sigma);
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
				weights_ = topWeights_;
			}

			for (int r = 0; r <= range2; r++)
			{
				backpropagation = 0;

				calculateLayers();
				computeErrors(r, lossFunction, plot, backpropagation);

				if (r % checkPoint2 == 0 && print == 1)
				{
					PRINT("Error after " + std::to_string(r) + " 2nd iterations: " + std::to_string(bestErrorExpl_) + " or FunctErr: " + std::to_string(bestError_) << "\n");
				}

				for (int i = 0; i < weights_.size(); i++)
				{
					for (int j = 0; j < weights_[i].size(); j++)
					{
						for (int k = 0; k < weights_[i][j].size(); k++)
						{
							weights_.at(i).at(j).at(k) = ((epsilon - (1 - (epsilon - 1))) * static_cast <float> (rand()) / static_cast <float> (RAND_MAX)) + (1 - (epsilon - 1)) * topWeights_.at(i).at(j).at(k);
						}
					}
				}
			}

			if (fase2 == 1)
			{
				weights_ = topWeights_;
			}

			float alpha = 0.0;

			for (int r = 0; r <= range3; r++)
			{
				backpropagation = 1;
				alpha = Maths::randNormalDistrib(muAlpha, sigmaAlpha);

				calculateLayers();
				computeErrors(r, lossFunction, plot, backpropagation);

				if (r % checkPoint3 == 0 && print == 1)
				{
					PRINT("Error after " << r << " 3rd iterations: " << error_ << " or General Error: " << errorV_ << "\n");
				}

				if (r != range3)
				{
					for (int i = 0; i < layerDeltas_[layerDeltas_.size() - 1].size(); i++)
					{
						layerDeltas_[layerDeltas_.size() - 1][i][0] = layerErrors_[layerErrors_.size() - 1][i][0] * Maths::derivativeSigmoid(layers_[layers_.size() - 1][i][0]);
						layerDeltaUse_[layerDeltaUse_.size() - 1][i][0] = layerDeltas_[layerDeltas_.size() - 1][i][0];
					}

					for (short int i = layerDeltas_.size() - 2; i > 0; i--)
					{
						layerErrors_[i] = Maths::matricesDotProduct(layerDeltaUse_[i + 1], Maths::transposeMatrix(weights_[i]));

						for (int j = 0; j < layers_[i].size(); j++)
						{
							for (int k = 0; k < layers_[i][j].size(); k++)
							{
								layerDeltas_[i][j][k] = layerErrors_[i][j][k] * Maths::derivativeSigmoid(layers_[i][j][k]);
								if (k != layers_[i][j].size() - 1)
								{
									layerDeltaUse_[i][j][k] = layerDeltas_[i][j][k];
								}
							}
						}
					}

					for (short int i = weights_.size() - 1; i > -1; i--)
					{
						results[i] = Maths::matricesDotProduct(Maths::transposeMatrix(layers_[i]), layerDeltaUse_[i + 1]);

						for (int j = 0; j < weights_[i].size(); j++)
						{
							for (int k = 0; k < weights_[i][j].size(); k++)
							{
								weights_[i][j][k] -= alpha * results[i][j][k];
							}
						}
					}
				}
			}

			if (fase3 == 1)
			{
				topWeights_ = weights_;
			}
		}

		void MultiLayerPerceptron::test()
		{
			test(0);
		}

		void MultiLayerPerceptron::test(const int& lossFunction)
		{
			buildWeights(-1, 1, 0);
			importWeights();

			start_ = 0;

			calculateLayers();
			computeErrors(0, lossFunction, 0, 1);

			PRINT("Test Error is: " << error_ << " or Test General Err: " << errorV_ << "\n");
		}

		void MultiLayerPerceptron::classify()
		{
			buildWeights(-1, 1, 0);
			importWeights();

			calculateLayers();
			computeErrors(0, 0, 0, 1);

			float predict = 0.0;

			for (int i = 0; i < layers_[layers_.size() - 1].size(); i++)
			{
				if (layers_[layers_.size() - 1][i][0] > 0.5)
				{
					predict = 1.0;
				}
				else
				{
					predict = 0.0;
				}

				PRINT("  " << predict << "  (" << layers_[layers_.size() - 1][i][0] << ")\n");
			}

			PRINT("\n");
		}

		void MultiLayerPerceptron::calculateLayers()
		{
			for (short int i = 0; i < numNeurLayers_; i++)
			{
				for (int j = 0; j < layers_[i + 1].size(); j++)
				{
					std::vector<std::vector<float>> results;
					results = Maths::matricesDotProduct(layers_[i], weights_[i]);

					for (int k = 0; k < layers_[i + 1][j].size() - 1; k++)
					{
						layers_.at(i + 1).at(j).at(k) = Maths::sigmoid(results[j][k]);
					}
				}
			}

			std::vector<std::vector<float>> results;

			results = Maths::matricesDotProduct(layers_[layers_.size() - 2], weights_[weights_.size() - 1]);

			for (int i = 0; i < layers_[layers_.size() - 1].size(); i++)
			{
				layers_.at(layers_.size() - 1).at(i).at(0) = Maths::sigmoid(results[i][0]);

				if (layers_.at(layers_.size() - 1).at(i).at(0) >= 1.0f)
				{
					layers_.at(layers_.size() - 1).at(i).at(0) = 0.99999f;
				}
				if (layers_.at(layers_.size() - 1).at(i).at(0) <= 0.0f)
				{
					layers_.at(layers_.size() - 1).at(i).at(0) = 0.00001f;
				}
			}
		}

		void MultiLayerPerceptron::computeErrors(const int& counter, const int& lossFunction, const bool& plot, const bool& backpropagation)
		{
			lastLayerErrorV_ = Maths::matrixVectorDifference(layers_[layers_.size() - 1], importer_->getY());

			if (backpropagation == 0)
			{
				if (counter == 0 && start_ == 0)
				{
					bestErrorExpl_ = Maths::sum(Maths::abs(lastLayerErrorV_));
				}

				start_ = 1;

				if (lossFunction == 0)
				{
					lastLayerError_ = Maths::lossFunctSimple(layers_[layers_.size() - 1], importer_->getY());
				}
				else if (lossFunction == 1)
				{
					lastLayerError_ = Maths::lossFunctLog(layers_[layers_.size() - 1], importer_->getY());
				}
				else if (lossFunction == 2)
				{
					lastLayerError_ = Maths::lossFunctLogPow3(layers_[layers_.size() - 1], importer_->getY());
				}
				else if (lossFunction == 3)
				{
					lastLayerError_ = Maths::lossFunctPow3(layers_[layers_.size() - 1], importer_->getY());
				}
				else if (lossFunction == 4)
				{
					lastLayerError_ = Maths::lossFunctPow3PLogPow3(layers_[layers_.size() - 1], importer_->getY());
				}

				if (Maths::mean(Maths::abs(lastLayerError_)) < bestError_)
				{
					topWeights_ = weights_;
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
						layerErrors_[layerErrors_.size() - 1][i][0] = Maths::lossFunctSimple(layers_[layers_.size() - 1], importer_->getY())[i];
					}
				}
				else if (lossFunction == 1)
				{
					for (int i = 0; i < layerErrors_[layerErrors_.size() - 1].size(); i++)
					{
						layerErrors_[layerErrors_.size() - 1][i][0] = Maths::lossFunctLog(layers_[layers_.size() - 1], importer_->getY())[i];
					}
				}
				else if (lossFunction == 2)
				{
					for (int i = 0; i < layerErrors_[layerErrors_.size() - 1].size(); i++)
					{
						layerErrors_[layerErrors_.size() - 1][i][0] = Maths::lossFunctLogPow3(layers_[layers_.size() - 1], importer_->getY())[i];
					}
				}
				else if (lossFunction == 3)
				{
					for (int i = 0; i < layerErrors_[layerErrors_.size() - 1].size(); i++)
					{
						layerErrors_[layerErrors_.size() - 1][i][0] = Maths::lossFunctPow3(layers_[layers_.size() - 1], importer_->getY())[i];
					}
				}
				else if (lossFunction == 4)
				{
					for (int i = 0; i < layerErrors_[layerErrors_.size() - 1].size(); i++)
					{
						layerErrors_[layerErrors_.size() - 1][i][0] = Maths::lossFunctPow3PLogPow3(layers_[layers_.size() - 1], importer_->getY())[i];
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

		void MultiLayerPerceptron::computeStatistics()
		{
			int right0to0 = 0;
			int right1to1 = 0;
			int errors0to1 = 0;
			int errors1to0 = 0;

			float predict = 0.0;
			std::string errOrCorr = "wrong";

			for (int i = 0; i < results_.size(); i++)
			{
				results_[i][2] = layers_[layers_.size() - 1][i][0];

				if (layers_[layers_.size() - 1][i][0] >= 0.5f)
				{
					results_[i][1] = 1;

					if (results_[i][1] == results_[i][0])
					{
						errOrCorr = "CORRECT";
						right1to1++;
					}
					else
					{
						errOrCorr = "wrong";
						errors0to1++;
					}
				}
				else
				{
					results_[i][1] = 0;

					if (results_[i][1] == results_[i][0])
					{
						errOrCorr = "CORRECT";
						right0to0++;
					}
					else
					{
						errOrCorr = "wrong";
						errors1to0++;
					}
				}
				PRINT("It's  " << results_[i][0] << "  while predicted  " << results_[i][1] << "  (" << results_[i][2] << ")" << "\t\t" << errOrCorr << "\n");
			}

			PRINT("Final errorV: " << errorV_ << "\n");
			PRINT("Final error: " << error_ << "\n");
			PRINT("Right1to1: " << right1to1 << "\n");
			PRINT("Right0to0: " << right0to0 << "\n");
			PRINT("Errors0to1: " << errors0to1 << "\n");
			PRINT("Errors1to0: " << errors1to0 << "\n");
			PRINT("\n");
			PRINT("Total correct: " << right0to0 + right1to1 << "\n");
			PRINT("Total errors: " << errors0to1 + errors1to0 << "\n");
			PRINT("\n\n");
		}
	}
}








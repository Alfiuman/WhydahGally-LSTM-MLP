#include "LongShortTermMemory.h"

namespace WhydahGally
{
	namespace Base
	{
		Parameters::Parameters(const int& dimX, const int& numMemCell, const bool& importParam, const float& max, const float& min, const float& seedNo)
			: numMemCell_(numMemCell), concatLen_(dimX + numMemCell), problem_(0), weightsG_(numMemCell_, concatLen_), weightsI_(numMemCell_, concatLen_), weightsF_(numMemCell_, concatLen_), weightsO_(numMemCell_, concatLen_), biasG_(numMemCell_), biasI_(numMemCell_), biasF_(numMemCell_), biasO_(numMemCell_), differWeightsG_(numMemCell_, concatLen_), differWeightsI_(numMemCell_, concatLen_), differWeightsF_(numMemCell_, concatLen_), differWeightsO_(numMemCell_, concatLen_), differBiasG_(numMemCell_), differBiasI_(numMemCell_), differBiasF_(numMemCell_), differBiasO_(numMemCell_)
		{
			if (importParam == 0)
			{
				srand(seedNo);

				//Populating the weights with random numbers.
				for (int i = 0; i < weightsG_.rows_; i++)
				{
					for (int j = 0; j < weightsG_.cols_; j++)
					{
						weightsG_.elements_[i * weightsG_.cols_ + j] = ((max - min) * static_cast <float> (rand()) / static_cast <float> (RAND_MAX)) + min;
					}
				}

				for (int i = 0; i < weightsI_.rows_; i++)
				{
					for (int j = 0; j < weightsI_.cols_; j++)
					{
						weightsI_.elements_[i * weightsI_.cols_ + j] = ((max - min) * static_cast <float> (rand()) / static_cast <float> (RAND_MAX)) + min;
					}
				}

				for (int i = 0; i < weightsF_.rows_; i++)
				{
					for (int j = 0; j < weightsF_.cols_; j++)
					{
						weightsF_.elements_[i * weightsF_.cols_ + j] = ((max - min) * static_cast <float> (rand()) / static_cast <float> (RAND_MAX)) + min;
					}
				}

				for (int i = 0; i < weightsO_.rows_; i++)
				{
					for (int j = 0; j < weightsO_.cols_; j++)
					{
						weightsO_.elements_[i * weightsO_.cols_ + j] = ((max - min) * static_cast <float> (rand()) / static_cast <float> (RAND_MAX)) + min;
					}
				}

				//Populating the bias matrices with random numbers.
				for (int i = 0; i < biasG_.rows_; i++)
				{
					biasG_.elements_[i] = ((max - min) * static_cast <float> (rand()) / static_cast <float> (RAND_MAX)) + min;
				}

				for (int i = 0; i < biasI_.rows_; i++)
				{
					biasI_.elements_[i] = ((max - min) * static_cast <float> (rand()) / static_cast <float> (RAND_MAX)) + min;
				}

				for (int i = 0; i < biasF_.rows_; i++)
				{
					biasF_.elements_[i] = ((max - min) * static_cast <float> (rand()) / static_cast <float> (RAND_MAX)) + min;
				}

				for (int i = 0; i < biasO_.rows_; i++)
				{
					biasO_.elements_[i] = ((max - min) * static_cast <float> (rand()) / static_cast <float> (RAND_MAX)) + min;
				}
			}
			else
			{
				importParameters();
			}
		}

		Parameters::~Parameters()
		{

		}

		void Parameters::importParameters()
		{
			//Importing parameters from a txt file.
			try
			{
				std::ifstream file;

				file.open("weightsG.txt");

				if (file)
				{
					for (int i = 0; i < weightsG_.rows_; i++)
					{
						for (int j = 0; j < weightsG_.cols_; j++)
						{
							file >> weightsG_.elements_[i * weightsG_.cols_ + j];
						}
					}

					file.close();
				}
				else
				{
					PRINT("WeightsG not found.") << "\n";
					problem_ = 1;
				}
			}
			catch (const std::out_of_range& e)
			{
				PRINT("Out of range error in Weights import.") << "\n";
				problem_ = 1;
			}

			try
			{
				std::ifstream file;

				file.open("weightsI.txt");

				if (file)
				{
					for (int i = 0; i < weightsI_.rows_; i++)
					{
						for (int j = 0; j < weightsI_.cols_; j++)
						{
							file >> weightsI_.elements_[i * weightsI_.cols_ + j];
						}
					}

					file.close();
				}
				else
				{
					PRINT("WeightsI not found.") << "\n";
					problem_ = 1;
				}
			}
			catch (const std::out_of_range& e)
			{
				PRINT("Out of range error in Weights import.") << "\n";
				problem_ = 1;
			}

			try
			{
				std::ifstream file;

				file.open("weightsF.txt");

				if (file)
				{
					for (int i = 0; i < weightsF_.rows_; i++)
					{
						for (int j = 0; j < weightsF_.cols_; j++)
						{
							file >> weightsF_.elements_[i * weightsF_.cols_ + j];
						}
					}

					file.close();
				}
				else
				{
					PRINT("WeightsF not found.") << "\n";
					problem_ = 1;
				}
			}
			catch (const std::out_of_range& e)
			{
				PRINT("Out of range error in Weights import.") << "\n";
				problem_ = 1;
			}

			try
			{
				std::ifstream file;

				file.open("weightsO.txt");

				if (file)
				{
					for (int i = 0; i < weightsO_.rows_; i++)
					{
						for (int j = 0; j < weightsO_.cols_; j++)
						{
							file >> weightsO_.elements_[i * weightsO_.cols_ + j];
						}
					}

					file.close();
				}
				else
				{
					PRINT("WeightsO not found.") << "\n";
					problem_ = 1;
				}
			}
			catch (const std::out_of_range& e)
			{
				PRINT("Out of range error in Weights import.") << "\n";
				problem_ = 1;
			}

			try
			{
				std::ifstream file;

				file.open("biasG.txt");

				if (file)
				{
					for (int i = 0; i < biasG_.rows_; i++)
					{
						file >> biasG_.elements_[i];
					}

					file.close();
				}
				else
				{
					PRINT("biasG not found.") << "\n";
					problem_ = 1;
				}
			}
			catch (const std::out_of_range& e)
			{
				PRINT("Out of range error in Bias import.") << "\n";
				problem_ = 1;
			}

			try
			{
				std::ifstream file;

				file.open("biasI.txt");

				if (file)
				{
					for (int i = 0; i < biasI_.rows_; i++)
					{
						file >> biasI_.elements_[i];
					}

					file.close();
				}
				else
				{
					PRINT("biasI not found.") << "\n";
					problem_ = 1;
				}
			}
			catch (const std::out_of_range& e)
			{
				PRINT("Out of range error in Bias import.") << "\n";
				problem_ = 1;
			}

			try
			{
				std::ifstream file;

				file.open("biasF.txt");

				if (file)
				{
					for (int i = 0; i < biasF_.rows_; i++)
					{
						file >> biasF_.elements_[i];
					}

					file.close();
				}
				else
				{
					PRINT("biasF not found.") << "\n";
					problem_ = 1;
				}
			}
			catch (const std::out_of_range& e)
			{
				PRINT("Out of range error in Bias import.") << "\n";
				problem_ = 1;
			}

			try
			{
				std::ifstream file;

				file.open("biasO.txt");

				if (file)
				{
					for (int i = 0; i < biasO_.rows_; i++)
					{
						file >> biasO_.elements_[i];
					}

					file.close();
				}
				else
				{
					PRINT("biasO not found.") << "\n";
					problem_ = 1;
				}
			}
			catch (const std::out_of_range& e)
			{
				PRINT("Out of range error in Bias import.") << "\n";
				problem_ = 1;
			}
		}

		void Parameters::recomputeWeightsBias(const float& alpha)
		{
			//Recomputing the weights given an alpha and the differentials.
			for (int i = 0; i < weightsG_.rows_; i++)
			{
				for (int j = 0; j < weightsG_.cols_; j++)
				{
					weightsG_.elements_[i * weightsG_.cols_ + j] -= alpha * differWeightsG_.elements_[i * differWeightsG_.cols_ + j];
				}
			}
			

			for (int i = 0; i < weightsI_.rows_; i++)
			{
				for (int j = 0; j < weightsI_.cols_; j++)
				{
					weightsI_.elements_[i * weightsI_.cols_ + j] -= alpha * differWeightsI_.elements_[i * differWeightsI_.cols_ + j];
				}
			}

			for (int i = 0; i < weightsF_.rows_; i++)
			{
				for (int j = 0; j < weightsF_.cols_; j++)
				{
					weightsF_.elements_[i * weightsF_.cols_ + j] -= alpha * differWeightsF_.elements_[i * differWeightsF_.cols_ + j];
				}
			}

			for (int i = 0; i < weightsO_.rows_; i++)
			{
				for (int j = 0; j < weightsO_.cols_; j++)
				{
					weightsO_.elements_[i * weightsO_.cols_ + j] -= alpha * differWeightsO_.elements_[i * differWeightsO_.cols_ + j];
				}
			}

			//Recomputing the bias matrices given an alpha and the differentials.
			for (int i = 0; i < biasG_.rows_; i++)
			{
				biasG_.elements_[i] -= alpha * differBiasG_.elements_[i];
			}

			for (int i = 0; i < biasI_.rows_; i++)
			{
				biasI_.elements_[i] -= alpha * differBiasI_.elements_[i];
			}

			for (int i = 0; i < biasF_.rows_; i++)
			{
				biasF_.elements_[i] -= alpha * differBiasF_.elements_[i];
			}

			for (int i = 0; i < biasO_.rows_; i++)
			{
				biasO_.elements_[i] -= alpha * differBiasO_.elements_[i];
			}

			//Zeroing the differentials.
			for (int i = 0; i < differWeightsG_.rows_; i++)
			{
				for (int j = 0; j < differWeightsG_.cols_; j++)
				{
					differWeightsG_.elements_[i * differWeightsG_.cols_ + j] = 0.0f;
				}
			}

			for (int i = 0; i < differWeightsI_.rows_; i++)
			{
				for (int j = 0; j < differWeightsI_.cols_; j++)
				{
					differWeightsI_.elements_[i * differWeightsI_.cols_ + j] = 0.0f;
				}
			}

			for (int i = 0; i < differWeightsF_.rows_; i++)
			{
				for (int j = 0; j < differWeightsF_.cols_; j++)
				{
					differWeightsF_.elements_[i * differWeightsF_.cols_ + j] = 0.0f;
				}
			}

			for (int i = 0; i < differWeightsO_.rows_; i++)
			{
				for (int j = 0; j < differWeightsO_.cols_; j++)
				{
					differWeightsO_.elements_[i * differWeightsO_.cols_ + j] = 0.0f;
				}
			}

			for (int i = 0; i < differBiasG_.rows_; i++)
			{
				differBiasG_.elements_[i] = 0.0f;
			}

			for (int i = 0; i < differBiasI_.rows_; i++)
			{
				differBiasI_.elements_[i] = 0.0f;
			}

			for (int i = 0; i < differBiasF_.rows_; i++)
			{
				differBiasF_.elements_[i] = 0.0f;
			}

			for (int i = 0; i < differBiasO_.rows_; i++)
			{
				differBiasO_.elements_[i] = 0.0f;
			}
		}

		State::State(const int& dimX, const int& numMemCell)
			: numMemCell_(numMemCell), g_(numMemCell_), i_(numMemCell_), f_(numMemCell_), o_(numMemCell_), s_(numMemCell_), h_(numMemCell_), bottomDifferH_(numMemCell_), bottomDifferS_(numMemCell_), bottomDifferX_(dimX)
		{
			
		}

		State::~State()
		{

		}

		Node::Node(Parameters& parameters, State& state)
			: param_(&parameters), state_(&state), previousS_(1), previousH_(1), x_(1), xh_(1)
		{
			
		}

		Node::~Node()
		{
			delete state_;
		}

		void Node::computeBottomData(const Matrix<float>& x, const int& parall)
		{
			Matrix<float> prevS(state_->s_.rows_);
			Matrix<float> prevH(state_->h_.rows_);

			computeBottomData(x, prevS, prevH, parall);
		}

		void Node::computeBottomData(const Matrix<float>& x, const Matrix<float>& prevS, const Matrix<float>& prevH, const int& parall)
		{
			previousS_.copy(prevS);
			previousH_.copy(prevH);
			
			Matrix<float> xh(x.rows_ + previousH_.rows_);

			for (int i = 0; i < xh.rows_; i++)
			{
				if (i < x.rows_)
				{
					xh.elements_[i] = x.elements_[i];
				}
				else
				{
					xh.elements_[i] = previousH_.elements_[i - x.rows_];
				}

			}
			
			//Computing the State.
			Matrix<float> results(param_->weightsG_.rows_);

			Maths::matricesDotProduct(param_->weightsG_, xh, &results, parall);

			for (int i = 0; i < state_->g_.rows_; i++)
			{
				state_->g_.elements_[i] = tanh(results.elements_[i] + param_->biasG_.elements_[i]);
			}
			
			Maths::matricesDotProduct(param_->weightsI_, xh, &results, parall);

			for (int i = 0; i < state_->i_.rows_; i++)
			{
				state_->i_.elements_[i] = Maths::sigmoid(results.elements_[i] + param_->biasI_.elements_[i]);
			}

			Maths::matricesDotProduct(param_->weightsF_, xh, &results, parall);

			for (int i = 0; i < state_->f_.rows_; i++)
			{
				state_->f_.elements_[i] = Maths::sigmoid(results.elements_[i] + param_->biasF_.elements_[i]);
			}
			
			Maths::matricesDotProduct(param_->weightsO_, xh, &results, parall);

			for (int i = 0; i < state_->o_.rows_; i++)
			{
				state_->o_.elements_[i] = Maths::sigmoid(results.elements_[i] + param_->biasO_.elements_[i]);
			}

			for (int i = 0; i < state_->s_.rows_; i++)
			{
				state_->s_.elements_[i] = (state_->g_.elements_[i] * state_->i_.elements_[i]) + (previousS_.elements_[i] * state_->f_.elements_[i]);
			}

			for (int i = 0; i < state_->h_.rows_; i++)
			{
				state_->h_.elements_[i] = tanh(state_->s_.elements_[i]) * state_->o_.elements_[i];
			}
			
			x_.copy(x);
			xh_.copy(xh);
		}

		void Node::computeTopDiffer(const Matrix<float>& topDiffH, const Matrix<float>& topDiffS, const int& parall)
		{
			Matrix<float> ds(state_->o_.rows_);
			Matrix<float> doo(ds.rows_);
			Matrix<float> di(doo.rows_);
			Matrix<float> dg(di.rows_);
			Matrix<float> df(previousS_.rows_);
			
			//Computing the differentials.
			for (int i = 0; i < ds.rows_; i++)
			{
				ds.elements_[i] = state_->o_.elements_[i] * (1.0f - pow(tanh(state_->s_.elements_[i]), 2.0f)) * topDiffH.elements_[i] + topDiffS.elements_[i];
			}

			for (int i = 0; i < doo.rows_; i++)
			{
				doo.elements_[i] = state_->s_.elements_[i] * topDiffH.elements_[i];
			}

			for (int i = 0; i < di.rows_; i++)
			{
				di.elements_[i] = state_->g_.elements_[i] * ds.elements_[i];
			}

			for (int i = 0; i < dg.rows_; i++)
			{
				dg.elements_[i] = state_->i_.elements_[i] * ds.elements_[i];
			}

			for (int i = 0; i < df.rows_; i++)
			{
				df.elements_[i] = previousS_.elements_[i] * ds.elements_[i];
			}

			Matrix<float> inputDi(di.rows_);
			Matrix<float> inputDf(df.rows_);
			Matrix<float> inputDoo(doo.rows_);
			Matrix<float> inputDg(dg.rows_);

			//Computing the inputs.
			for (int i = 0; i < inputDi.rows_; i++)
			{
				inputDi.elements_[i] = (1.0f - state_->i_.elements_[i]) * state_->i_.elements_[i] * di.elements_[i];
			}

			for (int i = 0; i < inputDf.rows_; i++)
			{
				inputDf.elements_[i] = (1.0f - state_->f_.elements_[i]) * state_->f_.elements_[i] * df.elements_[i];
			}

			for (int i = 0; i < inputDoo.rows_; i++)
			{
				inputDoo.elements_[i] = (1.0f - state_->o_.elements_[i]) * state_->o_.elements_[i] * doo.elements_[i];
			}

			for (int i = 0; i < inputDg.rows_; i++)
			{
				inputDg.elements_[i] = (1.0f - pow(state_->g_.elements_[i], 2.0f)) * dg.elements_[i];
			}

			//Computing the differentials of the various parameters.
			Matrix<float> results(inputDi.rows_, xh_.rows_);

			Maths::outerProduct(inputDi, xh_, &results, parall);

			for (int i = 0; i < param_->differWeightsI_.rows_; i++)
			{
				for (int j = 0; j < param_->differWeightsI_.cols_; j++)
				{
					param_->differWeightsI_.elements_[i * param_->differWeightsI_.cols_ + j] += results.elements_[i * results.cols_ + j];
				}
			}

			Maths::outerProduct(inputDf, xh_, &results, parall);

			for (int i = 0; i < param_->differWeightsF_.rows_; i++)
			{
				for (int j = 0; j < param_->differWeightsF_.cols_; j++)
				{
					param_->differWeightsF_.elements_[i * param_->differWeightsF_.cols_ + j] += results.elements_[i * results.cols_ + j];
				}
			}

			Maths::outerProduct(inputDoo, xh_, &results, parall);

			for (int i = 0; i < param_->differWeightsO_.rows_; i++)
			{
				for (int j = 0; j < param_->differWeightsO_.cols_; j++)
				{
					param_->differWeightsO_.elements_[i * param_->differWeightsO_.cols_ + j] += results.elements_[i * results.cols_ + j];
				}
			}

			Maths::outerProduct(inputDg, xh_, &results, parall);

			for (int i = 0; i < param_->differWeightsG_.rows_; i++)
			{
				for (int j = 0; j < param_->differWeightsG_.cols_; j++)
				{
					param_->differWeightsG_.elements_[i * param_->differWeightsG_.cols_ + j] += results.elements_[i * results.cols_ + j];
				}
			}

			for (int i = 0; i < param_->differBiasI_.rows_; i++)
			{
				param_->differBiasI_.elements_[i] += inputDi.elements_[i];
			}

			for (int i = 0; i < param_->differBiasF_.rows_; i++)
			{
				param_->differBiasF_.elements_[i] += inputDf.elements_[i];
			}

			for (int i = 0; i < param_->differBiasO_.rows_; i++)
			{
				param_->differBiasO_.elements_[i] += inputDoo.elements_[i];
			}

			for (int i = 0; i < param_->differBiasG_.rows_; i++)
			{
				param_->differBiasG_.elements_[i] += inputDg.elements_[i];
			}

			//Computing the bottom differential of the State.
			Matrix<float> differXH(xh_.rows_);

			Matrix<float> results1(differXH.rows_);
			Matrix<float> transpose(1);

			Maths::transposeMatrix(param_->weightsI_, &transpose, parall);
			Maths::matricesDotProduct(transpose, inputDi, &results1, parall);

			for (int i = 0; i < differXH.rows_; i++)
			{
				differXH.elements_[i] += results1.elements_[i];
			}

			Maths::transposeMatrix(param_->weightsF_, &transpose, parall);
			Maths::matricesDotProduct(transpose, inputDf, &results1, parall);

			for (int i = 0; i < differXH.rows_; i++)
			{
				differXH.elements_[i] += results1.elements_[i];
			}

			Maths::transposeMatrix(param_->weightsO_, &transpose, parall);
			Maths::matricesDotProduct(transpose, inputDoo, &results1, parall);

			for (int i = 0; i < differXH.rows_; i++)
			{
				differXH.elements_[i] += results1.elements_[i];
			}

			Maths::transposeMatrix(param_->weightsG_, &transpose, parall);
			Maths::matricesDotProduct(transpose, inputDg, &results1, parall);

			for (int i = 0; i < differXH.rows_; i++)
			{
				differXH.elements_[i] += results1.elements_[i];
			}

			for (int i = 0; i < state_->bottomDifferS_.rows_; i++)
			{
				state_->bottomDifferS_.elements_[i] = ds.elements_[i] * state_->f_.elements_[i];
			}

			for (int i = 0; i < state_->bottomDifferX_.rows_; i++)
			{
				state_->bottomDifferX_.elements_[i] = differXH.elements_[i];
			}

			for (int i = 0; i < state_->bottomDifferH_.rows_; i++)
			{
				state_->bottomDifferH_.elements_[i] = differXH.elements_[i + state_->bottomDifferX_.rows_];
			}
		}

		LongShortTermMemory::LongShortTermMemory(Importer& importer, const int& numMemCell, const bool& importParam, const float& max, const float& min, const int& seedNo)
			: importer_(&importer), dimX_(importer.getXX()[0].size()), sizeX_(importer.getXX().size()), historyLength_(importer.getHistoryLength()), numMemCell_(numMemCell), predictions_(1), loss_(0.0f), generalLoss_(0.0f), changedImporter_(0)
		{
			param_ = new Parameters(dimX_, numMemCell_, importParam, max, min, seedNo);
		}

		LongShortTermMemory::~LongShortTermMemory()
		{
			delete param_;

			for (int i = 0; i < nodeList_.size(); i++)
			{
				delete nodeList_[i];
			}
		}

		void LongShortTermMemory::computeLoss(const Matrix<float>& listY, const int& lossFunct, const int& parall)
		{
			if (listY.rows_ == listX_.size())
			{
				int idx = listX_.size() - 1;

				loss_ = 0.0f;
				generalLoss_ = 0.0f;
				int count = 0;

				//Computing the losses given a particular loss function.
				if (lossFunct == LOSSFUNCTSIMPLE)
				{
					loss_ = pow(Maths::lossFunctSimple(nodeList_.at(idx)->state_->h_.elements_[0], listY.elements_[idx]), 2);
					generalLoss_ = Maths::abs(Maths::lossFunctSimple(nodeList_.at(idx)->state_->h_.elements_[0], listY.elements_[idx]));
				}
				else if(lossFunct == LOSSFUNCTLOG)
				{
					loss_ = pow(Maths::lossFunctLog(nodeList_.at(idx)->state_->h_.elements_[0], listY.elements_[idx]), 2);
					generalLoss_ = Maths::abs(Maths::lossFunctSimple(nodeList_.at(idx)->state_->h_.elements_[0], listY.elements_[idx]));
				}
				else if (lossFunct == LOSSFUNCTLOGPOW3)
				{
					loss_ = pow(Maths::lossFunctLogPow3(nodeList_.at(idx)->state_->h_.elements_[0], listY.elements_[idx]), 2);
					generalLoss_ = Maths::abs(Maths::lossFunctSimple(nodeList_.at(idx)->state_->h_.elements_[0], listY.elements_[idx]));
				}
				else if (lossFunct == LOSSFUNCTPOW3)
				{
					loss_ = pow(Maths::lossFunctPow3(nodeList_.at(idx)->state_->h_.elements_[0], listY.elements_[idx]), 2);
					generalLoss_ = Maths::abs(Maths::lossFunctSimple(nodeList_.at(idx)->state_->h_.elements_[0], listY.elements_[idx]));
				}
				else if (lossFunct == LOSSFUNCTPOW3PLOGPOW3)
				{
					loss_ = pow(Maths::lossFunctPow3PLogPow3(nodeList_.at(idx)->state_->h_.elements_[0], listY.elements_[idx]), 2);
					generalLoss_ = Maths::abs(Maths::lossFunctSimple(nodeList_.at(idx)->state_->h_.elements_[0], listY.elements_[idx]));
				}

				count++;

				//Computing the differentials of H given the Ys.
				Matrix<float> diffH(nodeList_.at(idx)->state_->h_.rows_);
				diffH.elements_[0] = 2 * (nodeList_.at(idx)->state_->h_.elements_[0] - listY.elements_[idx]);
				
				Matrix<float> diffS(param_->numMemCell_);

				nodeList_.at(idx)->computeTopDiffer(diffH, diffS, parall);

				idx--;

				//Computing the losses going backward through the Artificial Neural Network.
				while (idx >= 0)
				{
					if (lossFunct == LOSSFUNCTSIMPLE)
					{
						loss_ += pow(Maths::lossFunctSimple(nodeList_.at(idx)->state_->h_.elements_[0], listY.elements_[idx]), 2);
						generalLoss_ += Maths::abs(Maths::lossFunctSimple(nodeList_.at(idx)->state_->h_.elements_[0], listY.elements_[idx]));
					}
					else if (lossFunct == LOSSFUNCTLOG)
					{
						loss_ += pow(Maths::lossFunctLog(nodeList_.at(idx)->state_->h_.elements_[0], listY.elements_[idx]), 2);
						generalLoss_ += Maths::abs(Maths::lossFunctSimple(nodeList_.at(idx)->state_->h_.elements_[0], listY.elements_[idx]));
					}
					else if (lossFunct == LOSSFUNCTLOGPOW3)
					{
						loss_ += pow(Maths::lossFunctLogPow3(nodeList_.at(idx)->state_->h_.elements_[0], listY.elements_[idx]), 2);
						generalLoss_ += Maths::abs(Maths::lossFunctSimple(nodeList_.at(idx)->state_->h_.elements_[0], listY.elements_[idx]));
					}
					else if (lossFunct == LOSSFUNCTPOW3)
					{
						loss_ += pow(Maths::lossFunctPow3(nodeList_.at(idx)->state_->h_.elements_[0], listY.elements_[idx]), 2);
						generalLoss_ += Maths::abs(Maths::lossFunctSimple(nodeList_.at(idx)->state_->h_.elements_[0], listY.elements_[idx]));
					}
					else if (lossFunct == LOSSFUNCTPOW3PLOGPOW3)
					{
						loss_ += pow(Maths::lossFunctPow3PLogPow3(nodeList_.at(idx)->state_->h_.elements_[0], listY.elements_[idx]), 2);
						generalLoss_ += Maths::abs(Maths::lossFunctSimple(nodeList_.at(idx)->state_->h_.elements_[0], listY.elements_[idx]));
					}

					count++;

					for (int i = 0; i < diffH.rows_; i++)
					{
						diffH.elements_[i] = 0.0f;
					}

					diffH.elements_[0] = 2 * (nodeList_.at(idx)->state_->h_.elements_[0] - listY.elements_[idx]);

					for (int i = 0; i < diffH.rows_; i++)
					{
						diffH.elements_[i] += nodeList_.at(idx + 1)->state_->bottomDifferH_.elements_[i];
					}

					for (int i = 0; i < diffS.rows_; i++)
					{
						diffS.elements_[i] = nodeList_.at(idx + 1)->state_->bottomDifferS_.elements_[i];
					}

					nodeList_.at(idx)->computeTopDiffer(diffH, diffS, parall);
					idx--;
				}

				generalLoss_ /= count;
			}
			else
			{
				PRINT("Problem detected!") << "\n";
			}
		}

		void LongShortTermMemory::buildListX(const std::vector<float>& x, const int& parall)
		{
			//Bulding the X list for time series analysis.
			Matrix<float> xx(x.size());

			for (int i = 0; i < xx.rows_; i++)
			{
				xx.elements_[i] = x.at(i);
			}

			listX_.push_back(x);

			if (listX_.size() > nodeList_.size())
			{
				State* state = new State(dimX_, param_->numMemCell_);
				Node* node = new Node(*param_, *state);
				nodeList_.push_back(node);
			}

			int idx = listX_.size() - 1;
			
			if (idx == 0)
			{
				nodeList_.at(idx)->computeBottomData(xx, parall);
			}
			else
			{
				Matrix<float> prevS(1);
				prevS.copy(nodeList_.at(idx - 1)->state_->s_);

				Matrix<float> prevH(1);
				prevH.copy(nodeList_.at(idx - 1)->state_->h_);

				nodeList_.at(idx)->computeBottomData(xx, prevS, prevH, parall);
			}
		}

		void LongShortTermMemory::importWeights()
		{
			param_->importParameters();
		}

		void LongShortTermMemory::train()
		{
			//Default training of the algorithm using default arguments.
			train(10000, 10, 0.11f, 1, 0, 0, 0);
		}

		void LongShortTermMemory::train(const int& times, const int& view, const float& alpha, const bool& print, const int& lossFunct, const int& parall, const bool& exportParam)
		{
			//Training the Artificial Neural Network.
			if (param_->problem_ == 0)
			{
				Matrix<float> y(historyLength_);

				//Building the X 2D vector.
				std::vector<std::vector<float>> x(historyLength_, std::vector<float>(dimX_));

				for (int h = 0; h < sizeX_ - historyLength_ + 1; h++)
				{
					if (print == 1)
					{
						PRINT(h + 1 << "\n");
					}

					for (int i = 0; i < x.size(); i++)
					{
						for (int j = 0; j < x.at(i).size(); j++)
						{
							x.at(i).at(j) = importer_->getXX().at(i + h).at(j);
						}
					}

					//Building the Y matrix.
					for (int i = 0; i < y.rows_; i++)
					{
						y.elements_[i] = importer_->getYY().at(i + h);
					}

					//Proper training of the algorithm.
					for (int v = 0; v < times + 1; v++)
					{
						for (int w = 0; w < y.rows_; w++)
						{
							buildListX(x.at(w), parall);
						}

						computeLoss(y, lossFunct, parall);

						//Printing the losses.
						if (v % view == 0 && print == 1)
						{
							PRINT("Loss: " << loss_ << "\t after " << v << " iterations. And Genaral Loss: " << generalLoss_ << "\n");
						}

						if (v < times || h < sizeX_ - historyLength_)
						{
							param_->recomputeWeightsBias(alpha);
						}
						else
						{
							//Populating the prediction matrix.
							predictions_.resize(nodeList_.size());

							for (int i = 0; i < predictions_.rows_; i++)
							{
								predictions_.elements_[i] = nodeList_.at(i)->state_->h_.elements_[0];
							}
						}

						listX_.assign(0, std::vector<float>(0));
					}
				}
			}
			else
			{
				PRINT("Review the code or the import paths.") << "\n";
			}
			
			if (exportParam == 1)
			{
				exportWeights();
			}
		}

		void LongShortTermMemory::test()
		{
			//Testing the parameters of the algorithm using default arguments.
			test(0, 0);
		}

		void LongShortTermMemory::test(const int& lossFunct, const int& parall)
		{
			PRINT("The result of the test is: ") << "\n";

			train(0, 1, 0, 1, lossFunct, parall, 0);
		}

		void LongShortTermMemory::classify()
		{
			//Default classification.
			classify(0);
		}

		void LongShortTermMemory::classify(const int& parall)
		{
			//Classifying the elements of a new time series using the Artificial Neural Network.
			train(0, 1, 0, 0, 0, parall, 0);

			float predict = 0.0;

			for (int i = 0; i < nodeList_.size(); i++)
			{
				if (nodeList_.at(i)->state_->h_.elements_[0] > 0.5)
				{
					predict = 1.0;
				}
				else
				{
					predict = 0.0;
				}

				PRINT("  " << predict << "  (" << nodeList_.at(i)->state_->h_.elements_[0] << ")") << "\n";
			}

			std::cout << std::endl;
		}

		void LongShortTermMemory::computeStatistics()
		{
			//Computing the statistics of the training or the test.
			train(0, 1, 0, 1, 0, 0, 0);

			Matrix<float> y(historyLength_);

			for (int i = 0; i < historyLength_; i++)
			{
				y.elements_[i] = importer_->getYY().at(i + importer_->getYY().size() - historyLength_);
			}

			float predict = 0.0f;
			std::string errOrCorr = "wrong";

			int right0to0 = 0;
			int right1to1 = 0;
			int errors0to1 = 0;
			int errors1to0 = 0;

			for (int i = 0; i < nodeList_.size(); i++)
			{
				if (nodeList_.at(i)->state_->h_.elements_[0] > 0.5)
				{
					predict = 1.0f;

					if (y.elements_[i] == 1)
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
					predict = 0.0f;

					if (y.elements_[i] == 0)
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

				PRINT("It's  " << y.elements_[i] << "  while predicted  " << predict << "  (" << nodeList_.at(i)->state_->h_.elements_[0] << ")" << "\t\t" << errOrCorr) << "\n";
			}

			PRINT("\n");
			PRINT("Right1to1: " << right1to1) << "\n";
			PRINT("Right0to0: " << right0to0) << "\n";
			PRINT("Errors0to1: " << errors0to1) << "\n";
			PRINT("Errors1to0: " << errors1to0) << "\n";
			PRINT("\n");
			PRINT("Total correct: " << right0to0 + right1to1) << "\n";
			PRINT("Total errors: " << errors0to1 + errors1to0) << "\n";
			PRINT("\n") << "\n";
		}

		void LongShortTermMemory::exportWeights()
		{
			//Saving the parameters in a txt file.
			std::ofstream file;

			file.open("weightsG.txt");

			if (file)
			{
				for (int i = 0; i < param_->weightsG_.rows_; i++)
				{
					for (int j = 0; j < param_->weightsG_.cols_; j++)
					{
						file << param_->weightsG_.elements_[i * param_->weightsG_.cols_ + j];
						file << "\t";
					}

					file << std::endl;
				}

				file.close();
			}
			else
			{
				PRINT("Problem in opening the weightsG file.") << "\n";
			}

			file.open("weightsI.txt");

			if (file)
			{
				for (int i = 0; i < param_->weightsI_.rows_; i++)
				{
					for (int j = 0; j < param_->weightsI_.cols_; j++)
					{
						file << param_->weightsI_.elements_[i * param_->weightsI_.cols_ + j];
						file << "\t";
					}

					file << std::endl;
				}

				file.close();
			}
			else
			{
				PRINT("Problem in opening the weightsI file.") << "\n";
			}

			file.open("weightsF.txt");

			if (file)
			{
				for (int i = 0; i < param_->weightsF_.rows_; i++)
				{
					for (int j = 0; j < param_->weightsF_.cols_; j++)
					{
						file << param_->weightsF_.elements_[i * param_->weightsF_.cols_ + j];
						file << "\t";
					}

					file << std::endl;
				}

				file.close();
			}
			else
			{
				PRINT("Problem in opening the weightsF file.") << "\n";
			}

			file.open("weightsO.txt");

			if (file)
			{
				for (int i = 0; i < param_->weightsO_.rows_; i++)
				{
					for (int j = 0; j < param_->weightsO_.cols_; j++)
					{
						file << param_->weightsO_.elements_[i * param_->weightsO_.cols_ + j];
						file << "\t";
					}

					file << std::endl;
				}

				file.close();
			}
			else
			{
				PRINT("Problem in opening the weightsO file.") << "\n";
			}

			file.open("biasG.txt");

			if (file)
			{
				for (int i = 0; i < param_->biasG_.rows_; i++)
				{
					file << param_->biasG_.elements_[i];
					file << "\t";
				}

				file.close();
			}
			else
			{
				PRINT("Problem in opening the biasG file.") << "\n";
			}

			file.open("biasI.txt");

			if (file)
			{
				for (int i = 0; i < param_->biasI_.rows_; i++)
				{
					file << param_->biasI_.elements_[i];
					file << "\t";
				}

				file.close();
			}
			else
			{
				PRINT("Problem in opening the biasI file.") << "\n";
			}

			file.open("biasF.txt");

			if (file)
			{
				for (int i = 0; i < param_->biasF_.rows_; i++)
				{
					file << param_->biasF_.elements_[i];
					file << "\t";
				}

				file.close();
			}
			else
			{
				PRINT("Problem in opening the biasF file.") << "\n";
			}

			file.open("biasO.txt");

			if (file)
			{
				for (int i = 0; i < param_->biasO_.rows_; i++)
				{
					file << param_->biasO_.elements_[i];
					file << "\t";
				}

				file.close();
			}
			else
			{
				PRINT("Problem in opening the biasO file.") << "\n";
			}
		}
	}
}






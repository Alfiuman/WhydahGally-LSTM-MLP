
////////////////////////////////////////////////////////////////////////////////////////
//                         WHYDAH GALLY Machine Learning Tool                         //
////////////////////////////////////////////////////////////////////////////////////////

#include <thread>
#include <mutex>
#include <sstream>
#include <Windows.h>

#include "Definitions.h"
#include "LongShortTermMemory.h"
#include "MultiLayerPerceptron.h"
#include "MLPFast.h"

using namespace WhydahGally;
using namespace Base;
using namespace Maths;

//Creating the thread-safe Printer class for outputting the final loss of the machine learning algorithms.
class Printer
{
private:
	std::mutex mu_;
public:
	Printer() { }
	void printError(float loss, float generalLoss, int numThr, const std::string& numNeurons, float alpha)
	{
		std::lock_guard<std::mutex> guard(mu_);
		
		PRINT("Final error for thread " << numThr << " is: "  << loss << " \tand generalLoss is: " << generalLoss << " \twith " << numNeurons << " neurons and " << ((alpha == 0) ? "N/A" : std::to_string(alpha)) << " alpha.\n");
	}
};

//Creating the tasks for multithreading analysis, for the different machine learning algorithms.
void taskLSTM(int numThr, Importer* imp, int numCell, bool importParam, float max, float min, float seedNo, int testTimes, int viewsEach, float alpha, bool print, bool exportParam, int operation, int statistics, int lossFunct, int parall, Printer& printer)
{
	LongShortTermMemory* k = new LongShortTermMemory(*imp, numCell, importParam, max, min, seedNo);

	if (operation == 1)
	{
		k->train(testTimes, viewsEach, alpha, print, lossFunct, parall, exportParam);

		printer.printError(k->getLoss(), k->getGeneralLoss(), numThr, std::to_string(numCell), alpha);
	}
	else if (operation == 2)
	{
		k->test(lossFunct, parall); 
	}
	else if (operation == 3)
	{
		k->classify(parall);
	}

	if (statistics == 1)
	{
		k->computeStatistics();
	}

	delete k;
}

void taskMLP(int numThr, Importer* imp, float limMin, float limMax, float seedNo1, const std::vector<int>& numNeurArr, DistribParamForMLP& distrParam, int ranges[3], int checkPoints[3], int lossFunction, bool plot, bool print, bool importParam, bool exportParam, int operation, int statistics, Printer& printer)
{
	MultiLayerPerceptron* b = new MultiLayerPerceptron(*imp, limMin, limMax, seedNo1, numNeurArr);

	if (importParam == 1)
	{
		b->importWeights();
	}

	if (operation == 1)
	{
		b->train(distrParam, ranges, checkPoints, lossFunction, plot, print);
		
		std::string numNeur;
		
		for (int i = 0; i < b->getNumNeur().size(); ++i)
		{
			numNeur = numNeur + "-" + std::to_string(b->getNumNeur().at(i));
		}

		printer.printError(b->getError(), b->getErrorV(), numThr, numNeur, 0.0f);
	}
	else if (operation == 2)
	{
		b->test(lossFunction);
	}
	else if (operation == 3)
	{
		b->classify();
	}

	if (statistics == 1)
	{
		b->computeStatistics();
	}

	if (exportParam == 1)
	{
		b->exportWeights();
	}

	delete b;
}

void taskMLPFast(int numThr, Importer* imp, float limMin, float limMax, float seedNo1, const std::vector<int>& numNeurArr, DistribParamForMLP& distrParam, int ranges[3], int checkPoints[3], int lossFunction, bool plot, bool print, bool importParam, bool exportParam, int operation, int statistics, Printer& printer, int parall)
{
	MLPFast* c = new MLPFast(*imp, limMin, limMax, seedNo1, numNeurArr);

	if (importParam == 1)
	{
		c->importWeights();
	}

	if (operation == 1)
	{
		c->train(distrParam, ranges, checkPoints, lossFunction, plot, print, parall);

		std::string numNeur;

		for (int i = 0; i < c->getNumNeur().size(); ++i)
		{
			numNeur = numNeur + "-" + std::to_string(c->getNumNeur().at(i));
		}

		printer.printError(c->getError(), c->getErrorV(), numThr, numNeur, 0.0f);
	}
	else if (operation == 2)
	{
		c->test(lossFunction);
	}
	else if (operation == 3)
	{
		c->classify();
	}

	if (statistics == 1)
	{
		c->computeStatistics();
	}

	if (exportParam == 1)
	{
		c->exportWeights();
	}

	delete c;
}

// MAIN ---------------------------------------------------------------------------------------------------------

int main()
{
	PRINT("\n\n==============================================================================\n"
			  "==                                                                          ==\n"
			  "==                   Welcome onboard of the WHYDAH GALLY!                   ==\n"
		      "==                                                                          ==\n"
		      "==============================================================================\n"
																							"\n");

	srand(DEFAULT_SEEDNO);

#if DEBUG
	//General debug parameters.
	int numThreads = 1;

	int hist = 0;	//Recommended 0 for LSTM and 4 for MLP and MLPFast if the time series is under 50 time points.
	std::string nameFile = "Data9";

	bool importParam = 0;
	bool exportParam = 1;

	int algorithm = LSTM;
	int operation = TRAIN;
	bool statistics = 1;
	bool diffThreads = 1;
	int multiThreadCellStdDev = 30;
	float multiThreadAlphaStdDev = 0.5f;
	int parall = CPU;
	float bias = DEFAULT_BIAS;
	bool print = 1;

	Importer* a = new Importer(hist, bias, nameFile);

	if (operation == TEST || operation == CLASSIFY)
	{
		importParam = 1;	//We test or use to classify pre-built parameters.
		exportParam = 0;	//No export for non trained parameters.
		numThreads = 1;		//No need multithreading for test and classify.
	}

	if (numThreads == 1)
	{
		diffThreads = 0;	//One thread cannot be different from itself.
	}
	
	if (operation == CLASSIFY)
	{
		statistics = 0;		//It's not possible to have statistics for a classification.
	}

	if (numThreads > 1)
	{
		exportParam = 0;	//No export if there are more than one thread because we don't know from the beginning the parameter of which tread we want to export.
		statistics = 0;
		print = 0;	//The printing of the intermediate results is not thread safe at the moment.
	}

	std::vector<std::thread> tt(numThreads);

	Printer printer;
	long int start = GetTickCount();

	if (algorithm == LSTM)
	{
		// LSTM -------------------------------------------------------------------------------------------------
		
		//LSTM debug parameters.
		int numCell = 20;
		float max = 1.0f;
		float min = -1.0f;
		float seedNo = DEFAULT_SEEDNO;
		int lossFunct = LOSSFUNCTSIMPLE;

		int testTimes = 5000;
		int viewsEach = 1;
		float alpha = 0.11f;

		//Creating the LSTM debug threads.
		if (diffThreads == 0)
		{
			for (int i = 0; i < numThreads; ++i)
			{
				tt[i] = std::thread(taskLSTM, i, a, numCell, importParam, max, min, seedNo, testTimes, viewsEach, alpha, print, exportParam, operation, statistics, lossFunct, parall, std::ref(printer));
			}
		}
		else
		{
			tt[0] = std::thread(taskLSTM, 0, a, numCell, importParam, max, min, seedNo, testTimes, viewsEach, alpha, print, exportParam, operation, statistics, lossFunct, parall, std::ref(printer));

			for (int i = 1; i < numThreads; ++i)
			{
				tt[i] = std::thread(taskLSTM, i, a, abs(numCell + randNormalDistrib(0, multiThreadCellStdDev)), importParam, max, min, seedNo, testTimes, viewsEach, abs(alpha + randNormalDistrib(0.0f, multiThreadAlphaStdDev)), print, exportParam, operation, statistics, lossFunct, parall, std::ref(printer));
			}
		}
	}
	else if (algorithm == MLP || algorithm == MLPFAST)
	{
		// MLPs --------------------------------------------------------------------------------------------------

		//MLPs debug parameters.
		float limMin = -10.0f;
		float limMax = 10.0f;
		float seedNo1 = 0.0f;
		std::vector<int> numNeurArr{ 18, 24 };

		DistribParamForMLP distrParam;
		distrParam.mu_ = -10.0f;
		distrParam.sigma_ = 10.0f;
		distrParam.ranDistr_ = 0;
		distrParam.epsilon_ = 0.05f;
		distrParam.muAlpha_ = 0.4f;
		distrParam.sigmaAlpha_ = 0.1f;
		distrParam.seedNo_ = DEFAULT_SEEDNO;

		int ranges[3]{ 5000, 5000, 20000 };
		int checkPoints[3]{ 100, 100, 100 };
		int lossFunction = LOSSFUNCTSIMPLE;
		bool plot = 1;

		//Creating the MLP debug threads.
		if (algorithm == MLP)
		{
			if (diffThreads == 0)
			{
				for (int i = 0; i < numThreads; ++i)
				{
					tt[i] = std::thread(taskMLP, i, a, limMin, limMax, seedNo1, numNeurArr, distrParam, ranges, checkPoints, lossFunction, plot, print, importParam, exportParam, operation, statistics, std::ref(printer));
				}
			}
			else
			{
				tt[0] = std::thread(taskMLP, 0, a, limMin, limMax, seedNo1, numNeurArr, distrParam, ranges, checkPoints, lossFunction, plot, print, importParam, exportParam, operation, statistics, std::ref(printer));

				for (int i = 1; i < numThreads; ++i)
				{
					std::vector<int> numNeurArrAlt;

					for (int j = 0; j < numNeurArr.size(); ++j)
					{
						numNeurArrAlt.push_back((rand() % 100 > (100 / (numNeurArr.size() + 2)) + 1) ? abs(numNeurArr.at(j) + randNormalDistrib(0, multiThreadCellStdDev)) : ((j == 0) ? rand() % (numNeurArr.at(0) * 3) : 0));
					}

					//Other two eventual layers more.
					numNeurArrAlt.push_back((rand() % 100 > 65) ? abs(randNormalDistrib(0, multiThreadCellStdDev)) : 0);
					numNeurArrAlt.push_back((rand() % 100 > 79) ? abs(randNormalDistrib(0, multiThreadCellStdDev / 2)) : 0);

					tt[i] = std::thread(taskMLP, i, a, limMin, limMax, seedNo1, numNeurArrAlt, distrParam, ranges, checkPoints, lossFunction, plot, print, importParam, exportParam, operation, statistics, std::ref(printer));
				}
			}
		}
		else if (algorithm == MLPFAST)
		{
			if (diffThreads == 0)
			{
				for (int i = 0; i < numThreads; ++i)
				{
					tt[i] = std::thread(taskMLPFast, i, a, limMin, limMax, seedNo1, numNeurArr, distrParam, ranges, checkPoints, lossFunction, plot, print, importParam, exportParam, operation, statistics, std::ref(printer), parall);
				}
			}
			else
			{
				tt[0] = std::thread(taskMLPFast, 0, a, limMin, limMax, seedNo1, numNeurArr, distrParam, ranges, checkPoints, lossFunction, plot, print, importParam, exportParam, operation, statistics, std::ref(printer), parall);

				for (int i = 1; i < numThreads; ++i)
				{
					std::vector<int> numNeurArrAlt;

					for (int j = 0; j < numNeurArr.size(); ++j)
					{
						numNeurArrAlt.push_back((rand() % 100 > (100 / (numNeurArr.size() + 2)) + 1) ? abs(numNeurArr.at(j) + randNormalDistrib(0, multiThreadCellStdDev)) : ((j == 0) ? rand() % (numNeurArr.at(0) * 3) : 0));
					}
					
					//Other two eventual layers more.
					numNeurArrAlt.push_back((rand() % 100 > 65) ? abs(randNormalDistrib(0, multiThreadCellStdDev)) : 0);
					numNeurArrAlt.push_back((rand() % 100 > 79) ? abs(randNormalDistrib(0, multiThreadCellStdDev / 2)) : 0);

					tt[i] = std::thread(taskMLPFast, i, a, limMin, limMax, seedNo1, numNeurArrAlt, distrParam, ranges, checkPoints, lossFunction, plot, print, importParam, exportParam, operation, statistics, std::ref(printer), parall);
				}
			}
		}
	}
	// FINAL ------------------------------------------------------------------------------------------------

	//Joining the debug threads.
	for (int i = 0; i < numThreads; ++i)
	{
		tt[i].join();
	}

	long int end = GetTickCount();

	std::cout << "\n\nTicks needed: " << end - start << "\n\n";

	delete a;

#else
	//Creating the variables.
	//General variables.
	bool active = 1;
	int algorithm = LSTM;
	int operation = TRAIN;
	int numThreads = 0;
	int hist = 0;
	std::string nameFile = "aaa";
	bool importParam = 0;
	bool exportParam = 0;
	bool statistics = 0;
	bool diffThreads = 0;
	int parall = 0;
	float bias = 0.0f;
	int multiThreadCellStdDev = 0;

	//LSTM variables.
	int numCell = 0;
	float max = 0.0f;
	float min = 0.0f;
	float seedNo = 0.0f;
	int lossFunct = 0;
	int testTimes = 0;
	int viewsEach = 0;
	float alpha = 0.0f;
	bool print = 0;
	float multiThreadAlphaStdDev = 0.0f;

	//MLP variables.
	float limMax = 0.0f;
	float limMin = 0.0f;
	float seedNo1 = 0.0f;
	std::vector<int> numNeurArr;
	DistribParamForMLP distrParam;
	distrParam.mu_ = 0.0f;
	distrParam.sigma_ = 0.0f;
	distrParam.ranDistr_ = 0;
	distrParam.epsilon_ = 0.0f;
	distrParam.muAlpha_ = 0.0f;
	distrParam.sigmaAlpha_ = 0.0f;
	distrParam.seedNo_ = 0.0f;
	int ranges[3]{ 0 };
	int checkPoints[3]{ 0 };
	int lossFunction = LOSSFUNCTSIMPLE;
	bool plot = 0;

	//Asking to the user the various parameters.
	std::string answer;

	while (active)
	{
		PRINT("\nDIGIT THE ANSWERS AND PRESS ENTER:\n\n");

		PRINT("Which Neural Network do you want to use? (1 LSTM, 2 MLP, 3 MLPFst)\n");
		std::getline(std::cin, answer);

		try
		{
			algorithm = std::stoi(answer);
		}
		catch (const std::invalid_argument& e)
		{
			algorithm = LSTM;	//Default option.
		}
		catch (const std::out_of_range& e)
		{
			algorithm = LSTM;	//Default option.
		}

		if (algorithm > 3 || algorithm < 1) //Change if there are more machine learning algorithms.
		{
			algorithm = LSTM;	//Default option.
		}

		PRINT("You chose " << algorithm << ".\n");
		answer.clear();

		PRINT("\nDo you want to TRAIN it, TEST it or use it to CLASSIFY? (1, 2 or 3)\n");
		std::getline(std::cin, answer);

		try
		{
			operation = std::stoi(answer);
		}
		catch (const std::invalid_argument& e)
		{
			operation = TRAIN;	//Default option.
		}
		catch (const std::out_of_range& e)
		{
			operation = TRAIN;	//Default option.
		}

		if (operation > 3 || operation < 1)
		{
			operation = TRAIN;	//Default option.
		}

		PRINT("You chose " << operation << ".\n");
		answer.clear();

		PRINT("\nHow many threads do you want?\n");
		std::getline(std::cin, answer);

		try
		{
			numThreads = std::stoi(answer);
		}
		catch (const std::invalid_argument& e)
		{
			numThreads = 1;	//Default option.
		}
		catch (const std::out_of_range& e)
		{
			numThreads = 1;	//Default option.
		}

		if (numThreads < 1)
		{
			numThreads = 1;	//Default option.
		}

		if (operation == TEST || operation == CLASSIFY)
		{
			numThreads = 1;	//No multithreads for tests or classifications.
		}

		PRINT("You chose " << numThreads << ".\n");
		answer.clear();

		PRINT("\nHow long the history of the time series? (0 for entire history)\n");
		std::getline(std::cin, answer);

		try
		{
			hist = std::stoi(answer);
		}
		catch (const std::invalid_argument& e)
		{
			hist = 0;	//Default option.
		}
		catch (const std::out_of_range& e)
		{
			hist = 0;	//Default option.
		}

		if (hist < 0)
		{
			hist = 0;	//History length cannot be less than 0.
		}

		PRINT("You chose " << hist << ".\n");
		answer.clear();

		PRINT("\nWrite here the name and the path of the file.\n");
		std::getline(std::cin, answer);

		nameFile = answer;

		answer.clear();

		PRINT("\nWhat is the Bias value you want?\n");
		std::getline(std::cin, answer);

		try
		{
			bias = std::stof(answer);
		}
		catch (const std::invalid_argument& e)
		{
			bias = DEFAULT_BIAS;	//Default option.
		}
		catch (const std::out_of_range& e)
		{
			bias = DEFAULT_BIAS;	//Default option.
		}

		PRINT("You chose " << bias << ".\n");
		answer.clear();

		PRINT("\nDo you want to import the parameters? (1 yes, 0 no)\n");
		std::getline(std::cin, answer);

		try
		{
			importParam = (bool)std::stoi(answer);
		}
		catch (const std::invalid_argument& e)
		{
			importParam = 0;	//Default option.
		}
		catch (const std::out_of_range& e)
		{
			importParam = 0;	//Default option.
		}

		if (operation == TEST || operation == CLASSIFY)
		{
			importParam = 1;	//You have to use pre-built parameters to test or classify.
		}

		PRINT("You chose " << importParam << ".\n");
		answer.clear();

		PRINT("\nDo you want to export the parameters? (1 yes, 0 no)\n");
		std::getline(std::cin, answer);

		try
		{
			exportParam = (bool)std::stoi(answer);
		}
		catch (const std::invalid_argument& e)
		{
			exportParam = 0;	//Default option.
		}
		catch (const std::out_of_range& e)
		{
			exportParam = 0;	//Default option.
		}

		if (operation == TEST || operation == CLASSIFY || numThreads > 1)
		{
			exportParam = 0;	//Tests or classifications don't export parameters and also multithreading. 
		}

		PRINT("You chose " << exportParam << ".\n");
		answer.clear();

		PRINT("\nDo you want to print the intermediate results? (1 yes, 0 no)\n");
		std::getline(std::cin, answer);

		try
		{
			print = (bool)std::stoi(answer);
		}
		catch (const std::invalid_argument& e)
		{
			print = 1;	//Default option.
		}
		catch (const std::out_of_range& e)
		{
			print = 1;	//Default option.
		}

		if (numThreads != 1)
		{
			print = 0;	//Printing the intermediate results is not thread safe at the moment.
		}

		PRINT("You chose " << print << ".\n");
		answer.clear();

		PRINT("\nDo you want to show the statistics? (1 yes, 0 no)\n");
		std::getline(std::cin, answer);

		try
		{
			statistics = (bool)std::stoi(answer);
		}
		catch (const std::invalid_argument& e)
		{
			statistics = 0;		//Default option.
		}
		catch (const std::out_of_range& e)
		{
			statistics = 0;		//Default option.
		}

		if (operation == CLASSIFY)
		{
			statistics = 0;		//No statistics if we classify data.
		}

		if (numThreads != 1)
		{
			statistics = 0;		//No statistics for multithreading (they are not thread safe at the moment).
		}

		PRINT("You chose " << statistics << ".\n");
		answer.clear();

		if (numThreads > 1)
		{
			PRINT("\nDo you want different threads? (1 yes, 0 no)\n");
			std::getline(std::cin, answer);

			try
			{
				diffThreads = (bool)std::stoi(answer);
			}
			catch (const std::invalid_argument& e)
			{
				diffThreads = 0;	//Default option.
			}
			catch (const std::out_of_range& e)
			{
				diffThreads = 0;	//Default option.
			}

			PRINT("You chose " << diffThreads << ".\n");
			answer.clear();

			if (diffThreads)
			{
				PRINT("\nWhich value for the multi-thread number of neurons standard deviation do you want?\n");
				std::getline(std::cin, answer);

				try
				{
					multiThreadCellStdDev = std::stoi(answer);
				}
				catch (const std::invalid_argument& e)
				{
					multiThreadCellStdDev = 30;	//Default option.
				}
				catch (const std::out_of_range& e)
				{
					multiThreadCellStdDev = 30;	//Default option.
				}

				PRINT("You chose " << multiThreadCellStdDev << ".\n");
				answer.clear();
			}
		}
#if CUDA
		PRINT("\nWhich kind of computation do you want? (0 for CPU, 1 Global GPU, 2 Shared GPU, 11 Config1.)\n");
		std::getline(std::cin, answer);

		try
		{
			parall = std::stoi(answer);
		}
		catch (const std::invalid_argument& e)
		{
			parall = 0;		//Default option.
		}
		catch (const std::out_of_range& e)
		{
			parall = 0;		//Default option.
		}

		if (parall < 0 || (parall > 2 && parall != 11))
		{
			parall = 0;		//Default option.
		}

		PRINT("You chose " << parall << ".\n");
		answer.clear();
#endif
		if (algorithm == LSTM)
		{
			PRINT("\nHow many cells do you want?\n");
			std::getline(std::cin, answer);

			try
			{
				numCell = std::stoi(answer);
			}
			catch (const std::invalid_argument& e)
			{
				numCell = 20;	//Default option.
			}
			catch (const std::out_of_range& e)
			{
				numCell = 20;	//Default option.
			}

			if (numCell < 0)
			{
				numCell = 20;	//Default option.
			}

			PRINT("You chose " << numCell << ".\n");
			answer.clear();

			PRINT("\nWhat is the MAX value you want for the random initialization?\n");
			std::getline(std::cin, answer);

			try
			{
				max = std::stof(answer);
			}
			catch (const std::invalid_argument& e)
			{
				max = 1.0f;		//Default option.
			}
			catch (const std::out_of_range& e)
			{
				max = 1.0f;		//Default option.
			}

			PRINT("You chose " << max << ".\n");
			answer.clear();

			PRINT("\nWhat is the MIN value you want for the random initialization?\n");
			std::getline(std::cin, answer);

			try
			{
				min = std::stof(answer);
			}
			catch (const std::invalid_argument& e)
			{
				min = -1.0f;	//Default option.
			}
			catch (const std::out_of_range& e)
			{
				min = -1.0f;	//Default option.
			}

			if (min >= max)
			{
				min = max - 1.0f;	//Min has to be less than Max.
			}

			PRINT("You chose " << min << ".\n");
			answer.clear();

			PRINT("\nWhat is the seed number you want to use for the random initialization?\n");
			std::getline(std::cin, answer);

			try
			{
				seedNo = std::stof(answer);
			}
			catch (const std::invalid_argument& e)
			{
				seedNo = DEFAULT_SEEDNO;	//Default option.
			}
			catch (const std::out_of_range& e)
			{
				seedNo = DEFAULT_SEEDNO;	//Default option.
			}

			PRINT("You chose " << seedNo << ".\n");
			answer.clear();

			PRINT("\nWhich loss function do you want to use? (0 simple, 1 log, 2 logPow3, 3 pow3, 4 pow3LogPow3)\n");
			std::getline(std::cin, answer);

			try
			{
				lossFunct = std::stoi(answer);
			}
			catch (const std::invalid_argument& e)
			{
				lossFunct = 0;		//Default option.
			}
			catch (const std::out_of_range& e)
			{
				lossFunct = 0;		//Default option.
			}

			if (lossFunct < 0 || lossFunct > 4)
			{
				lossFunct = 0;		//Default option.
			}

			PRINT("You chose " << lossFunct << ".\n");
			answer.clear();

			PRINT("\nHow many iterations do you want?\n");
			std::getline(std::cin, answer);

			try
			{
				testTimes = std::stoi(answer);
			}
			catch (const std::invalid_argument& e)
			{
				testTimes = 1000;	//Default option.
			}
			catch (const std::out_of_range& e)
			{
				testTimes = 1000;	//Default option.
			}

			if (testTimes < 0)
			{
				testTimes = 1000;	//Default option.
			}

			PRINT("You chose " << testTimes << ".\n");
			answer.clear();

			if (print == 1)
			{
				PRINT("\nHow often do you want to see the intermediate result? (put the number of the iteration)\n");
				std::getline(std::cin, answer);

				try
				{
					viewsEach = std::stoi(answer);
				}
				catch (const std::invalid_argument& e)
				{
					viewsEach = 100;	//Default option.
				}
				catch (const std::out_of_range& e)
				{
					viewsEach = 100;	//Default option.
				}

				if (viewsEach < 0)
				{
					viewsEach = 100;	//Default option.
				}

				PRINT("You chose " << viewsEach << ".\n");
				answer.clear();
			}
			else
			{
				viewsEach = testTimes + 1;
			}
			
			PRINT("\nPlease select the alpha.\n");
			std::getline(std::cin, answer);

			try
			{
				alpha = std::stof(answer);
			}
			catch (const std::invalid_argument& e)
			{
				alpha = 0.11f;	//Default option.
			}
			catch (const std::out_of_range& e)
			{
				alpha = 0.11f;	//Default option.
			}

			PRINT("You chose " << alpha << ".\n");
			answer.clear();

			if (numThreads > 1 && diffThreads == 1)
			{
				PRINT("\nWhich value for the multi-thread alpha standard deviation do you want?\n");
				std::getline(std::cin, answer);

				try
				{
					multiThreadAlphaStdDev = std::stof(answer);
				}
				catch (const std::invalid_argument& e)
				{
					multiThreadAlphaStdDev = 0.5f;	//Default option.
				}
				catch (const std::out_of_range& e)
				{
					multiThreadAlphaStdDev = 0.5f;	//Default option.
				}

				PRINT("You chose " << multiThreadAlphaStdDev << ".\n");
				answer.clear();
			}
		}
		else if (algorithm == MLP || algorithm == MLPFAST)
		{
			PRINT("\nWhat is the MAX value you want for the random initialization?\n");
			std::getline(std::cin, answer);

			try
			{
				limMax = std::stof(answer);
			}
			catch (const std::invalid_argument& e)
			{
				limMax = 1.0f;	//Default option.
			}
			catch (const std::out_of_range& e)
			{
				limMax = 1.0f;	//Default option.
			}

			PRINT("You chose " << limMax << ".\n");
			answer.clear();

			PRINT("\nWhat is the MIN value you want for the random initialization?\n");
			std::getline(std::cin, answer);

			try
			{
				limMin = std::stof(answer);
			}
			catch (const std::invalid_argument& e)
			{
				limMin = -1.0f;	//Default option.
			}
			catch (const std::out_of_range& e)
			{
				limMin = -1.0f;	//Default option.
			}

			if (limMin > limMax)
			{
				limMin = limMax - 1.0f;	//Min has to be less or equal than Max.
			}

			PRINT("You chose " << limMin << ".\n");
			answer.clear();

			PRINT("\nWhat is the seed number you want to use for the random initialization?\n");
			std::getline(std::cin, answer);

			try
			{
				seedNo1 = std::stof(answer);
			}
			catch (const std::invalid_argument& e)
			{
				seedNo1 = DEFAULT_SEEDNO;	//Default option.
			}
			catch (const std::out_of_range& e)
			{
				seedNo1 = DEFAULT_SEEDNO;	//Default option.
			}

			PRINT("You chose " << seedNo1 << ".\n");
			answer.clear();

			bool endOfLayers = 0;
			int counterLayers = 0;

			while (!endOfLayers)
			{
				PRINT("\nHow many neurons do you want in the layer " << counterLayers + 1 << "?\n");
				std::getline(std::cin, answer);

				try
				{
					numNeurArr.push_back(std::stoi(answer));
				}
				catch (const std::invalid_argument& e)	//In Visual Studio 2015 with Release x64 it gives unexpected behaviour when we print it at row 1022. To avoid it, it needs to be catched by value.
				{
					numNeurArr.push_back(10);		//Default option.
				}
				catch (const std::out_of_range& e)
				{
					numNeurArr.push_back(10);		//Default option.
				}

				if (numNeurArr[counterLayers] < 0)
				{
					numNeurArr[counterLayers] = 10;		//Default option.
				}

				if (counterLayers == 0 && numNeurArr[counterLayers] == 0)
				{
					numNeurArr[counterLayers] = 1;		//The first layer should have at least one neuron.
				}

				PRINT("You chose " << numNeurArr[counterLayers] << ".\n");
				answer.clear();

				if (numNeurArr[counterLayers] == 0)
				{
					endOfLayers = 1;
					break;
				}
				
				counterLayers++;
			}

			PRINT("\nWhich random distribution do you want? (0 uniform, 1 normal)\n");
			std::getline(std::cin, answer);

			try
			{
				distrParam.ranDistr_ = std::stoi(answer);
			}
			catch (const std::invalid_argument& e)
			{
				distrParam.ranDistr_ = 0;	//Default option.
			}
			catch (const std::out_of_range& e)
			{
				distrParam.ranDistr_ = 0;	//Default option.
			}

			if (distrParam.ranDistr_ < 0 || distrParam.ranDistr_ > 1)
			{
				distrParam.ranDistr_ = 0;	//Default option.
			}

			PRINT("You chose " << distrParam.ranDistr_ << ".\n");
			answer.clear();

			PRINT("\nWhat the value of Mu do you want?\n");
			std::getline(std::cin, answer);

			try
			{
				distrParam.mu_ = std::stof(answer);
			}
			catch (const std::invalid_argument& e)
			{
				distrParam.mu_ = -10.0f;	//Default option.
			}
			catch (const std::out_of_range& e)
			{
				distrParam.mu_ = -10.0f;	//Default option.
			}

			PRINT("You chose " << distrParam.mu_ << ".\n");
			answer.clear();

			PRINT("\nWhat the value of Sigma do you want?\n");
			std::getline(std::cin, answer);

			try
			{
				distrParam.sigma_ = std::stof(answer);
			}
			catch (const std::invalid_argument& e)
			{
				distrParam.sigma_ = 10.0f;	//Default option.
			}
			catch (const std::out_of_range& e)
			{
				distrParam.sigma_ = 10.0f;	//Default option.
			}

			if (distrParam.sigma_ < distrParam.mu_ && distrParam.ranDistr_ == 0)
			{
				distrParam.sigma_ = distrParam.mu_ + 1.0f;	//If ranDistr is uniform then Min has to be less or equal than Max.
			}

			PRINT("You chose " << distrParam.sigma_ << ".\n");
			answer.clear();

			PRINT("\nHow many iterations do you want for the first phase?\n");
			std::getline(std::cin, answer);

			try
			{
				ranges[0] = std::stoi(answer);
			}
			catch (const std::invalid_argument& e)
			{
				ranges[0] = 1000;	//Default option.
			}
			catch (const std::out_of_range& e)
			{
				ranges[0] = 1000;	//Default option.
			}

			if (ranges[0] < 0)
			{
				ranges[0] = 1000;	//Default option.
			}

			PRINT("You chose " << ranges[0] << ".\n");
			answer.clear();

			PRINT("\nHow many iterations do you want for the second phase?\n");
			std::getline(std::cin, answer);

			try
			{
				ranges[1] = std::stoi(answer);
			}
			catch (const std::invalid_argument& e)
			{
				ranges[1] = 1000;	//Default option.
			}
			catch (const std::out_of_range& e)
			{
				ranges[1] = 1000;	//Default option.
			}

			if (ranges[1] < 0)
			{
				ranges[1] = 1000;	//Default option.
			}

			PRINT("You chose " << ranges[1] << ".\n");
			answer.clear();

			PRINT("\nHow many iterations do you want for the third phase?\n");
			std::getline(std::cin, answer);

			try
			{
				ranges[2] = std::stoi(answer);
			}
			catch (const std::invalid_argument& e)
			{
				ranges[2] = 1000;	//Default option.
			}
			catch (const std::out_of_range& e)
			{
				ranges[2] = 1000;	//Default option.
			}

			if (ranges[2] < 0)
			{
				ranges[2] = 1000;	//Default option.
			}

			PRINT("You chose " << ranges[2] << ".\n");
			answer.clear();

			if (print == 1)
			{
				PRINT("\nAt the multiple of which iteration number do you want the checkpoint for the first phase?\n");
				std::getline(std::cin, answer);

				try
				{
					checkPoints[0] = std::stoi(answer);
				}
				catch (const std::invalid_argument& e)
				{
					checkPoints[0] = 100;	//Default option.
				}
				catch (const std::out_of_range& e)
				{
					checkPoints[0] = 100;	//Default option.
				}

				if (checkPoints[0] < 0)
				{
					checkPoints[0] = 100;	//Default option.
				}

				PRINT("You chose " << checkPoints[0] << ".\n");
				answer.clear();

				PRINT("\nAt the multiple of which iteration number do you want the checkpoint for the second phase?\n");
				std::getline(std::cin, answer);

				try
				{
					checkPoints[1] = std::stoi(answer);
				}
				catch (const std::invalid_argument& e)
				{
					checkPoints[1] = 100;	//Default option.
				}
				catch (const std::out_of_range& e)
				{
					checkPoints[1] = 100;	//Default option.
				}

				if (checkPoints[1] < 0)
				{
					checkPoints[1] = 100;	//Default option.
				}

				PRINT("You chose " << checkPoints[1] << ".\n");
				answer.clear();

				PRINT("\nAt the multiple of which iteration number do you want the checkpoint for the third phase?\n");
				std::getline(std::cin, answer);

				try
				{
					checkPoints[2] = std::stoi(answer);
				}
				catch (const std::invalid_argument& e)
				{
					checkPoints[2] = 100;	//Default option.
				}
				catch (const std::out_of_range& e)
				{
					checkPoints[2] = 100;	//Default option.
				}

				if (checkPoints[2] < 0)
				{
					checkPoints[2] = 100;	//Default option.
				}

				PRINT("You chose " << checkPoints[2] << ".\n");
				answer.clear();
			}
			else
			{
				checkPoints[0] = ranges[0] + 1;
				checkPoints[1] = ranges[1] + 1;
				checkPoints[2] = ranges[2] + 1;
			}

			PRINT("\nWhich value of Epsilon do you want?\n");
			std::getline(std::cin, answer);

			try
			{
				distrParam.epsilon_ = std::stof(answer);
			}
			catch (const std::invalid_argument& e)
			{
				distrParam.epsilon_ = 0.05f;	//Default option.
			}
			catch (const std::out_of_range& e)
			{
				distrParam.epsilon_ = 0.05f;	//Default option.
			}

			if (distrParam.epsilon_ < 0)
			{
				distrParam.epsilon_ = 1.0f;		//Other default option.
			}

			PRINT("You chose " << distrParam.epsilon_ << ".\n");
			answer.clear();

			PRINT("\nWhich value of Mu Alpha do you want?\n");
			std::getline(std::cin, answer);

			try
			{
				distrParam.muAlpha_ = std::stof(answer);
			}
			catch (const std::invalid_argument& e)
			{
				distrParam.muAlpha_ = 0.4f;	//Default option.
			}
			catch (const std::out_of_range& e)
			{
				distrParam.muAlpha_ = 0.4f;	//Default option.
			}

			PRINT("You chose " << distrParam.muAlpha_ << ".\n");
			answer.clear();

			PRINT("\nWhich value of Sigma Alpha do you want?\n");
			std::getline(std::cin, answer);

			try
			{
				distrParam.sigmaAlpha_ = std::stof(answer);
			}
			catch (const std::invalid_argument& e)
			{
				distrParam.sigmaAlpha_ = 0.1f;	//Default option.
			}
			catch (const std::out_of_range& e)
			{
				distrParam.sigmaAlpha_ = 0.1f;	//Default option.
			}

			if (distrParam.sigmaAlpha_ < 0)
			{
				distrParam.sigmaAlpha_ = 0.1f;	//Default option.
			}

			PRINT("You chose " << distrParam.sigmaAlpha_ << ".\n");
			answer.clear();

			PRINT("\nWhich loss function do you want to use? (0 simple, 1 log, 2 logPow3, 3 pow3, 4 pow3LogPow3)\n");
			std::getline(std::cin, answer);

			try
			{
				lossFunction = std::stoi(answer);
			}
			catch (const std::invalid_argument& e)
			{
				lossFunction = 0;	//Default option.
			}
			catch (const std::out_of_range& e)
			{
				lossFunction = 0;	//Default option.
			}

			if (lossFunction < 0 || lossFunction > 4)
			{
				lossFunction = 0;	//Default option.
			}

			PRINT("You chose " << lossFunction << ".\n");
			answer.clear();

			//Plot not implemented yet.
			/*PRINT("\nDo you want to plot the results? (1 yes, 0 no)\n");
			std::getline(std::cin, answer);

			try
			{
				plot = (bool)std::stoi(answer);
			}
			catch (const std::invalid_argument& e)
			{
				plot = 0;
			}
			catch (const std::out_of_range& e)
			{
				plot = 0;
			}

			PRINT("You chose " << plot << ".\n");
			answer.clear();*/

			PRINT("\nWhat is the second seed number you want to use for the random initialization?\n");
			std::getline(std::cin, answer);

			try
			{
				distrParam.seedNo_ = std::stof(answer);
			}
			catch (const std::invalid_argument& e)
			{
				distrParam.seedNo_ = DEFAULT_SEEDNO;	//Default option.
			}
			catch (const std::out_of_range& e)
			{
				distrParam.seedNo_ = DEFAULT_SEEDNO;	//Default option.
			}

			PRINT("You chose " << distrParam.seedNo_ << ".\n");
			answer.clear();
		}

		Importer* a = new Importer(hist, bias, nameFile);

		std::vector<std::thread> tt(numThreads);

		Printer printer;

		if (algorithm == LSTM)
		{
			//Creating the LSTM threads.
			if (diffThreads == 0)
			{
				for (int i = 0; i < numThreads; ++i)
				{
					tt[i] = std::thread(taskLSTM, i, a, numCell, importParam, max, min, seedNo, testTimes, viewsEach, alpha, print, exportParam, operation, statistics, lossFunct, parall, std::ref(printer));
				}
			}
			else
			{
				tt[0] = std::thread(taskLSTM, 0, a, numCell, importParam, max, min, seedNo, testTimes, viewsEach, alpha, print, exportParam, operation, statistics, lossFunct, parall, std::ref(printer));

				for (int i = 1; i < numThreads; ++i)
				{
					tt[i] = std::thread(taskLSTM, i, a, abs(numCell + randNormalDistrib(0, multiThreadCellStdDev)), importParam, max, min, seedNo, testTimes, viewsEach, abs(alpha + randNormalDistrib(0.0f, multiThreadAlphaStdDev)), print, exportParam, operation, statistics, lossFunct, parall, std::ref(printer));
				}
			}
		}
		else if (algorithm == MLP)
		{
			//Creating the MLP threads.
			if (diffThreads == 0)
			{
				for (int i = 0; i < numThreads; ++i)
				{
					tt[i] = std::thread(taskMLP, i, a, limMin, limMax, seedNo1, numNeurArr, distrParam, ranges, checkPoints, lossFunction, plot, print, importParam, exportParam, operation, statistics, std::ref(printer));
				}
			}
			else
			{
				tt[0] = std::thread(taskMLP, 0, a, limMin, limMax, seedNo1, numNeurArr, distrParam, ranges, checkPoints, lossFunction, plot, print, importParam, exportParam, operation, statistics, std::ref(printer));

				for (int i = 1; i < numThreads; ++i)
				{
					std::vector<int> numNeurArrAlt;

					for (int j = 0; j < numNeurArr.size(); ++j)
					{
						numNeurArrAlt.push_back((rand() % 100 > (100 / (numNeurArr.size() + 2)) + 1) ? abs(numNeurArr.at(j) + randNormalDistrib(0, multiThreadCellStdDev)) : ((j == 0) ? rand() % (numNeurArr.at(0) * 3) : 0));
					}

					//Other two eventual layers more.
					numNeurArrAlt.push_back((rand() % 100 > 65) ? abs(randNormalDistrib(0, multiThreadCellStdDev)) : 0);
					numNeurArrAlt.push_back((rand() % 100 > 79) ? abs(randNormalDistrib(0, multiThreadCellStdDev / 2)) : 0);

					tt[i] = std::thread(taskMLP, i, a, limMin, limMax, seedNo1, numNeurArrAlt, distrParam, ranges, checkPoints, lossFunction, plot, print, importParam, exportParam, operation, statistics, std::ref(printer));
				}
			}
		}
		else if (algorithm == MLPFAST)
		{
			//Creating the MLPFast threads.
			if (diffThreads == 0)
			{
				for (int i = 0; i < numThreads; ++i)
				{
					tt[i] = std::thread(taskMLPFast, i, a, limMin, limMax, seedNo1, numNeurArr, distrParam, ranges, checkPoints, lossFunction, plot, print, importParam, exportParam, operation, statistics, std::ref(printer), parall);
				}
			}
			else
			{
				tt[0] = std::thread(taskMLPFast, 0, a, limMin, limMax, seedNo1, numNeurArr, distrParam, ranges, checkPoints, lossFunction, plot, print, importParam, exportParam, operation, statistics, std::ref(printer), parall);

				for (int i = 1; i < numThreads; ++i)
				{
					std::vector<int> numNeurArrAlt;

					for (int j = 0; j < numNeurArr.size(); ++j)
					{
						if (numNeurArr[j] != 0)
						{
							numNeurArrAlt.push_back((rand() % 100 >(100 / (numNeurArr.size() + 2)) + 1) ? abs(numNeurArr.at(j) + randNormalDistrib(0, multiThreadCellStdDev)) : ((j == 0) ? rand() % (numNeurArr.at(0) * 3) : 0));
						}
					}

					//Other two eventual layers more.
					numNeurArrAlt.push_back((rand() % 100 > 65) ? abs(randNormalDistrib(0, multiThreadCellStdDev)) : 0);
					numNeurArrAlt.push_back((rand() % 100 > 79) ? abs(randNormalDistrib(0, multiThreadCellStdDev / 2)) : 0);

					tt[i] = std::thread(taskMLPFast, i, a, limMin, limMax, seedNo1, numNeurArrAlt, distrParam, ranges, checkPoints, lossFunction, plot, print, importParam, exportParam, operation, statistics, std::ref(printer), parall);
				}
			}
		}

		//Joining the threads.
		for (int i = 0; i < numThreads; ++i)
		{
			tt[i].join();
		}

		delete a;

		//Asking if the user wants to perform another analysis.
		PRINT("\nDo you want to perform another analysis? (1 yes, 0 no)\n");
		std::getline(std::cin, answer);

		try
		{
			active = (bool)std::stoi(answer);
		}
		catch (const std::invalid_argument& e)
		{
			active = 0;
		}
		catch (const std::out_of_range& e)
		{
			active = 0;
		}

		PRINT("You chose " << active << ".\n");
		answer.clear();
	} 
#endif
	//End of the program.
	PRINT("\n\nEND\nPress Enter to exit.\n");
	
	std::cin.get();
	return 0;
}




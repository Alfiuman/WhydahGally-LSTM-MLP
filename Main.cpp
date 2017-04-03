
////////////////////////////////////////////////////////////////////////////////////////
//                         WHYDAH GALLY Machine Learning Tool                         //
////////////////////////////////////////////////////////////////////////////////////////

#include <thread>
#include <mutex>

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
	void printError(const float& loss, const float& generalLoss, const int& numThr)
	{
		std::lock_guard<std::mutex> guard(mu_);
	
		PRINT("Final error for thread " << numThr << " is: "  << loss << " and generalLoss is: " << generalLoss) << "\n";
	}
};

//Creating the tasks for multithreading analysis, for the different machine learning algorithms.
void taskLSTM(const int& numThr, Importer* imp, const int& numCell, const bool& importParam, const float& max, const float& min, const float& seedNo, const int& testTimes, const int& viewsEach, const float& alpha, const bool& print, const bool& exportParam, const int& operation, const int& statistics, const int& lossFunct, const int& parall, Printer& printer)
{
	LongShortTermMemory* k = new LongShortTermMemory(*imp, numCell, importParam, max, min, seedNo);

	if (operation == 1)
	{
		k->train(testTimes, viewsEach, alpha, print, lossFunct, parall, exportParam);

		printer.printError(k->getLoss(), k->getGeneralLoss(), numThr);
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

void taskMLP(const int& numThr, Importer* imp, const float& limMin, const float& limMax, const float& seedNo1, int numNeurArr[12], DistribParamForMLP& distrParam, int ranges[3], int checkPoints[3], const int& lossFunction, const bool& plot, const bool& print, const bool& importParam, const bool& exportParam, const int& operation, const int& statistics, Printer& printer)
{
	MultiLayerPerceptron* b = new MultiLayerPerceptron(*imp, limMin, limMax, seedNo1, numNeurArr);

	if (importParam == 1)
	{
		b->importWeights();
	}

	if (operation == 1)
	{
		b->train(distrParam, ranges, checkPoints, lossFunction, plot, print);

		printer.printError(b->getError(), b->getErrorV(), numThr);
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

void taskMLPFst(const int& numThr, Importer* imp, const float& limMin, const float& limMax, const float& seedNo1, int numNeurArr[12], DistribParamForMLP& distrParam, int ranges[3], int checkPoints[3], const int& lossFunction, const bool& plot, const bool& print, const bool& importParam, const bool& exportParam, const int& operation, const int& statistics, Printer& printer, const int& parall)
{
	MLPFast* c = new MLPFast(*imp, limMin, limMax, seedNo1, numNeurArr);

	if (importParam == 1)
	{
		c->importWeights();
	}

	if (operation == 1)
	{
		c->train(distrParam, ranges, checkPoints, lossFunction, plot, print, parall);

		printer.printError(c->getError(), c->getErrorV(), numThr);
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

#if DEBUG
	//General debug parameters.
	int numThreads = 1;

	int hist = 0;	//Recommended 0 for LSTM and 4 for MLP and MLPFast if the time series is under 50 time points.
	std::string nameFile = "Data9";

	bool importParam = 0;
	bool exportParam = 1;

	int algorithm = LSTM;
	int operation = TRAIN;
	bool statistics = 0;
	bool diffThreads = 0;
	int parall = CPU;
	float bias = DEFAULT_BIAS;

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
	}

	if (numThreads > MAX_NUM_THREADS && diffThreads == 1)
	{
		numThreads = MAX_NUM_THREADS;	//There is a max number of threads if they are different.
	}

	std::vector<std::thread> tt(numThreads);

	Printer printer;

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
		bool print = 1;

		if (numThreads != 1)
		{
			print = 0;	//The printing of the intermediate results is not thread safe at the moment.
		}

		//Creating the LSTM debug threads.
		if (diffThreads == 0)
		{
			for (int i = 0; i < numThreads; i++)
			{
				tt[i] = std::thread(taskLSTM, i, a, numCell, importParam, max, min, seedNo, testTimes, viewsEach, alpha, print, exportParam, operation, statistics, lossFunct, parall, std::ref(printer));
			}
		}
		else
		{
			tt[0] = std::thread(taskLSTM, 0, a, 20, importParam, max, min, 0, testTimes, viewsEach, 0.11, print, exportParam, operation, statistics, lossFunct, parall, std::ref(printer));
			
			if (numThreads > 1)
			{
				tt[1] = std::thread(taskLSTM, 1, a, 20, importParam, max, min, 1, testTimes, viewsEach, 0.11, print, exportParam, operation, statistics, lossFunct, parall, std::ref(printer));
			}

			if (numThreads > 2)
			{
				tt[2] = std::thread(taskLSTM, 2, a, 20, importParam, max, min, 0, testTimes, viewsEach, 0.5, print, exportParam, operation, statistics, lossFunct, parall, std::ref(printer));
			}
			
			if (numThreads > 3)
			{
				tt[3] = std::thread(taskLSTM, 3, a, 20, importParam, max, min, 1, testTimes, viewsEach, 0.5, print, exportParam, operation, statistics, lossFunct, parall, std::ref(printer));
			}

			if (numThreads > 4)
			{
				tt[4] = std::thread(taskLSTM, 4, a, 20, importParam, max, min, 0, testTimes, viewsEach, 0.9, print, exportParam, operation, statistics, lossFunct, parall, std::ref(printer));
			}
			
			if (numThreads > 5)
			{
				tt[5] = std::thread(taskLSTM, 5, a, 20, importParam, max, min, 1, testTimes, viewsEach, 0.9, print, exportParam, operation, statistics, lossFunct, parall, std::ref(printer));
			}
			
			if (numThreads > 6)
			{
				tt[6] = std::thread(taskLSTM, 6, a, 40, importParam, max, min, 0, testTimes, viewsEach, 0.11, print, exportParam, operation, statistics, lossFunct, parall, std::ref(printer));
			}
			
			if (numThreads > 7)
			{
				tt[7] = std::thread(taskLSTM, 7, a, 40, importParam, max, min, 1, testTimes, viewsEach, 0.11, print, exportParam, operation, statistics, lossFunct, parall, std::ref(printer));
			}
		}
	}
	else if (algorithm == MLP || algorithm == MLPFAST)
	{
		// MLP --------------------------------------------------------------------------------------------------

		//MLP debug parameters.
		float limMin = -10.0f;
		float limMax = 10.0f;
		float seedNo1 = 0.0f;
		int numNeurArr[MAX_NUM_LAYERS_MLP]{ 18, 24 };

		DistribParamForMLP distrParam;
		distrParam.mu_ = -10.0f;
		distrParam.sigma_ = 10.0f;
		distrParam.ranDistr_ = 0;
		distrParam.epsilon_ = 0.05f;
		distrParam.muAlpha_ = 0.4f;
		distrParam.sigmaAlpha_ = 0.1f;
		distrParam.seedNo_ = DEFAULT_SEEDNO;

		int ranges[3]{ 1000, 1000, 5000 };
		int checkPoints[3]{ 100, 100, 100 };
		int lossFunction = LOSSFUNCTSIMPLE;
		bool print1 = 1;
		bool plot = 1;

		if (numThreads != 1)
		{
			print1 = 0;	//The printing of the intermediate results is not thread safe at the moment.
		}

		//Creating the MLP debug threads.
		if (algorithm == MLP)
		{
			if (diffThreads == 0)
			{
				for (int i = 0; i < numThreads; i++)
				{
					tt[i] = std::thread(taskMLP, i, a, limMin, limMax, seedNo1, numNeurArr, distrParam, ranges, checkPoints, lossFunction, plot, print1, importParam, exportParam, operation, statistics, std::ref(printer));
				}
			}
			else
			{
				tt[0] = std::thread(taskMLP, 0, a, limMin, limMax, seedNo1, numNeurArr, distrParam, ranges, checkPoints, lossFunction, plot, print1, importParam, exportParam, operation, statistics, std::ref(printer));

				if (numThreads > 1)
				{
					int numNeurArrAlt[MAX_NUM_LAYERS_MLP]{ 24, 18 };
					tt[1] = std::thread(taskMLP, 1, a, limMin, limMax, seedNo1, numNeurArrAlt, distrParam, ranges, checkPoints, lossFunction, plot, print1, importParam, exportParam, operation, statistics, std::ref(printer));
				}

				if (numThreads > 2)
				{
					int numNeurArrAlt[MAX_NUM_LAYERS_MLP]{ 18, 18 };
					tt[2] = std::thread(taskMLP, 2, a, limMin, limMax, seedNo1, numNeurArrAlt, distrParam, ranges, checkPoints, lossFunction, plot, print1, importParam, exportParam, operation, statistics, std::ref(printer));
				}

				if (numThreads > 3)
				{
					int numNeurArrAlt[MAX_NUM_LAYERS_MLP]{ 32, 22, 12 };
					tt[3] = std::thread(taskMLP, 3, a, limMin, limMax, seedNo1, numNeurArrAlt, distrParam, ranges, checkPoints, lossFunction, plot, print1, importParam, exportParam, operation, statistics, std::ref(printer));
				}

				if (numThreads > 4)
				{
					int numNeurArrAlt[MAX_NUM_LAYERS_MLP]{ 18, 24, 12 };
					tt[4] = std::thread(taskMLP, 4, a, limMin, limMax, seedNo1, numNeurArrAlt, distrParam, ranges, checkPoints, lossFunction, plot, print1, importParam, exportParam, operation, statistics, std::ref(printer));
				}

				if (numThreads > 5)
				{
					int numNeurArrAlt[MAX_NUM_LAYERS_MLP]{ 24, 18, 6 };
					tt[5] = std::thread(taskMLP, 5, a, limMin, limMax, seedNo1, numNeurArrAlt, distrParam, ranges, checkPoints, lossFunction, plot, print1, importParam, exportParam, operation, statistics, std::ref(printer));
				}

				if (numThreads > 6)
				{
					int numNeurArrAlt[MAX_NUM_LAYERS_MLP]{ 32, 16, 8 };
					tt[6] = std::thread(taskMLP, 6, a, limMin, limMax, seedNo1, numNeurArrAlt, distrParam, ranges, checkPoints, lossFunction, plot, print1, importParam, exportParam, operation, statistics, std::ref(printer));
				}

				if (numThreads > 7)
				{
					int numNeurArrAlt[MAX_NUM_LAYERS_MLP]{ 16 };
					tt[7] = std::thread(taskMLP, 7, a, limMin, limMax, seedNo1, numNeurArrAlt, distrParam, ranges, checkPoints, lossFunction, plot, print1, importParam, exportParam, operation, statistics, std::ref(printer));
				}
			}
		}
		else if (algorithm == MLPFAST)
		{
			if (diffThreads == 0)
			{
				for (int i = 0; i < numThreads; i++)
				{
					tt[i] = std::thread(taskMLPFst, i, a, limMin, limMax, seedNo1, numNeurArr, distrParam, ranges, checkPoints, lossFunction, plot, print1, importParam, exportParam, operation, statistics, std::ref(printer), parall);
				}
			}
			else
			{
				tt[0] = std::thread(taskMLPFst, 0, a, limMin, limMax, seedNo1, numNeurArr, distrParam, ranges, checkPoints, lossFunction, plot, print1, importParam, exportParam, operation, statistics, std::ref(printer), parall);

				if (numThreads > 1)
				{
					int numNeurArrAlt[MAX_NUM_LAYERS_MLP]{ 24, 18 };
					tt[1] = std::thread(taskMLPFst, 1, a, limMin, limMax, seedNo1, numNeurArrAlt, distrParam, ranges, checkPoints, lossFunction, plot, print1, importParam, exportParam, operation, statistics, std::ref(printer), parall);
				}

				if (numThreads > 2)
				{
					int numNeurArrAlt[MAX_NUM_LAYERS_MLP]{ 18, 18 };
					tt[2] = std::thread(taskMLPFst, 2, a, limMin, limMax, seedNo1, numNeurArrAlt, distrParam, ranges, checkPoints, lossFunction, plot, print1, importParam, exportParam, operation, statistics, std::ref(printer), parall);
				}

				if (numThreads > 3)
				{
					int numNeurArrAlt[MAX_NUM_LAYERS_MLP]{ 32, 22, 12 };
					tt[3] = std::thread(taskMLPFst, 3, a, limMin, limMax, seedNo1, numNeurArrAlt, distrParam, ranges, checkPoints, lossFunction, plot, print1, importParam, exportParam, operation, statistics, std::ref(printer), parall);
				}

				if (numThreads > 4)
				{
					int numNeurArrAlt[MAX_NUM_LAYERS_MLP]{ 18, 24, 12 };
					tt[4] = std::thread(taskMLPFst, 4, a, limMin, limMax, seedNo1, numNeurArrAlt, distrParam, ranges, checkPoints, lossFunction, plot, print1, importParam, exportParam, operation, statistics, std::ref(printer), parall);
				}

				if (numThreads > 5)
				{
					int numNeurArrAlt[MAX_NUM_LAYERS_MLP]{ 24, 18, 6 };
					tt[5] = std::thread(taskMLPFst, 5, a, limMin, limMax, seedNo1, numNeurArrAlt, distrParam, ranges, checkPoints, lossFunction, plot, print1, importParam, exportParam, operation, statistics, std::ref(printer), parall);
				}

				if (numThreads > 6)
				{
					int numNeurArrAlt[MAX_NUM_LAYERS_MLP]{ 32, 16, 8 };
					tt[6] = std::thread(taskMLPFst, 6, a, limMin, limMax, seedNo1, numNeurArrAlt, distrParam, ranges, checkPoints, lossFunction, plot, print1, importParam, exportParam, operation, statistics, std::ref(printer), parall);
				}

				if (numThreads > 7)
				{
					int numNeurArrAlt[MAX_NUM_LAYERS_MLP]{ 16 };
					tt[7] = std::thread(taskMLPFst, 7, a, limMin, limMax, seedNo1, numNeurArrAlt, distrParam, ranges, checkPoints, lossFunction, plot, print1, importParam, exportParam, operation, statistics, std::ref(printer), parall);
				}
			}
		}
	}
	// FINAL ------------------------------------------------------------------------------------------------

	//Joining the debug threads.
	for (int i = 0; i < numThreads; i++)
	{
		tt[i].join();
	}

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

	//MLP variables.
	float limMax = 0.0f;
	float limMin = 0.0f;
	float seedNo1 = 0.0f;
	int numNeurArr[12]{ 0 };
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
	bool print1 = 0;
	bool plot = 0;

	//Asking to the user the various parameters.
	std::string answer;

	while (active == 1)
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

		if (numThreads > MAX_NUM_THREADS)
		{
			numThreads = MAX_NUM_THREADS;	//There is a Max.
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
		}
#if CUDA
		PRINT("\nWhich kind of computation do you want? (0 for CPU, 1 Global GPU, 2 Shared GPU)\n");
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

		if (parall < 0 || parall > 2)
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

			PRINT("\nDo you want to print the results? (1 yes, 0 no)\n");
			std::getline(std::cin, answer);

			try
			{
				print = (bool)std::stoi(answer);
			}
			catch (const std::invalid_argument& e)
			{
				print = 0;	//Default option.
			}
			catch (const std::out_of_range& e)
			{
				print = 0;	//Default option.
			}

			if (numThreads != 1)
			{
				print = 0;	//Printing the intermediate results is not thread safe at the moment.
			}

			PRINT("You chose " << print << ".\n");
			answer.clear();
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

			for (int n = 0; n < MAX_NUM_LAYERS_MLP; n++)
			{
				PRINT("\nHow many neurons do you want in the layer " << n + 1 << "?\n");
				std::getline(std::cin, answer);

				try
				{
					numNeurArr[n] = std::stoi(answer);
				}
				catch (const std::invalid_argument& e)
				{
					numNeurArr[n] = 10;		//Default option.
				}
				catch (const std::out_of_range& e)
				{
					numNeurArr[n] = 10;		//Default option.
				}

				if (numNeurArr[n] < 0)
				{
					numNeurArr[n] = 10;		//Default option.
				}

				if (n == 0 && numNeurArr[n] == 0)
				{
					numNeurArr[n] = 1;		//The first layer should have at least one neuron.
				}

				PRINT("You chose " << numNeurArr[n] << ".\n");
				answer.clear();

				if (numNeurArr[n] == 0)
				{
					break;
				}
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

			PRINT("\nDo you want to print the results? (1 yes, 0 no)\n");
			std::getline(std::cin, answer);

			try
			{
				print1 = (bool)std::stoi(answer);
			}
			catch (const std::invalid_argument& e)
			{
				print1 = 0;		//Default option.
			}
			catch (const std::out_of_range& e)
			{
				print1 = 0;		//Default option.
			}

			if (numThreads != 1)
			{
				print1 = 0;		//Printing the intermediate results is not thread safe at the moment.
			}

			PRINT("You chose " << print1 << ".\n");
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

		if (numThreads > 8 && diffThreads == 1)
		{
			numThreads = 8;		//Max number of different threads.
		}

		std::vector<std::thread> tt(numThreads);

		Printer printer;

		if (algorithm == LSTM)
		{
			//Creating the LSTM threads.
			if (diffThreads == 0)
			{
				for (int i = 0; i < numThreads; i++)
				{
					tt[i] = std::thread(taskLSTM, i, a, numCell, importParam, max, min, seedNo, testTimes, viewsEach, alpha, print, exportParam, operation, statistics, lossFunct, parall, std::ref(printer));
				}
			}
			else
			{
				tt[0] = std::thread(taskLSTM, 0, a, 20, importParam, max, min, DEFAULT_SEEDNO, testTimes, viewsEach, 0.11, print, exportParam, operation, statistics, lossFunct, parall, std::ref(printer));

				if (numThreads > 1)
				{
					tt[1] = std::thread(taskLSTM, 1, a, 20, importParam, max, min, 1.0f, testTimes, viewsEach, 0.11, print, exportParam, operation, statistics, lossFunct, parall, std::ref(printer));
				}

				if (numThreads > 2)
				{
					tt[2] = std::thread(taskLSTM, 2, a, 20, importParam, max, min, DEFAULT_SEEDNO, testTimes, viewsEach, 0.5, print, exportParam, operation, statistics, lossFunct, parall, std::ref(printer));
				}

				if (numThreads > 3)
				{
					tt[3] = std::thread(taskLSTM, 3, a, 20, importParam, max, min, 1.0f, testTimes, viewsEach, 0.5, print, exportParam, operation, statistics, lossFunct, parall, std::ref(printer));
				}

				if (numThreads > 4)
				{
					tt[4] = std::thread(taskLSTM, 4, a, 20, importParam, max, min, DEFAULT_SEEDNO, testTimes, viewsEach, 0.9, print, exportParam, operation, statistics, lossFunct, parall, std::ref(printer));
				}

				if (numThreads > 5)
				{
					tt[5] = std::thread(taskLSTM, 5, a, 20, importParam, max, min, 1.0f, testTimes, viewsEach, 0.9, print, exportParam, operation, statistics, lossFunct, parall, std::ref(printer));
				}

				if (numThreads > 6)
				{
					tt[6] = std::thread(taskLSTM, 6, a, 40, importParam, max, min, DEFAULT_SEEDNO, testTimes, viewsEach, 0.11, print, exportParam, operation, statistics, lossFunct, parall, std::ref(printer));
				}

				if (numThreads > 7)
				{
					tt[7] = std::thread(taskLSTM, 7, a, 40, importParam, max, min, 1.0f, testTimes, viewsEach, 0.11, print, exportParam, operation, statistics, lossFunct, parall, std::ref(printer));
				}
			}
		}
		else if (algorithm == MLP)
		{
			//Creating the MLP threads.
			if (diffThreads == 0)
			{
				for (int i = 0; i < numThreads; i++)
				{
					tt[i] = std::thread(taskMLP, i, a, limMin, limMax, seedNo1, numNeurArr, distrParam, ranges, checkPoints, lossFunction, plot, print1, importParam, exportParam, operation, statistics, std::ref(printer));
				}
			}
			else
			{
				tt[0] = std::thread(taskMLP, 0, a, limMin, limMax, seedNo1, numNeurArr, distrParam, ranges, checkPoints, lossFunction, plot, print1, importParam, exportParam, operation, statistics, std::ref(printer));

				if (numThreads > 1)
				{
					int numNeurArrAlt[MAX_NUM_LAYERS_MLP]{ (numNeurArr[1] > 0) ? numNeurArr[1] : 1, numNeurArr[0] };
					tt[1] = std::thread(taskMLP, 1, a, limMin, limMax, seedNo1, numNeurArrAlt, distrParam, ranges, checkPoints, lossFunction, plot, print1, importParam, exportParam, operation, statistics, std::ref(printer));
				}

				if (numThreads > 2)
				{
					int numNeurArrAlt[MAX_NUM_LAYERS_MLP]{ numNeurArr[0], numNeurArr[0] };
					tt[2] = std::thread(taskMLP, 2, a, limMin, limMax, seedNo1, numNeurArrAlt, distrParam, ranges, checkPoints, lossFunction, plot, print1, importParam, exportParam, operation, statistics, std::ref(printer));
				}

				if (numThreads > 3)
				{
					int numNeurArrAlt[MAX_NUM_LAYERS_MLP]{ numNeurArr[0] * 2, numNeurArr[0], numNeurArr[0] / 2 };
					tt[3] = std::thread(taskMLP, 3, a, limMin, limMax, seedNo1, numNeurArrAlt, distrParam, ranges, checkPoints, lossFunction, plot, print1, importParam, exportParam, operation, statistics, std::ref(printer));
				}

				if (numThreads > 4)
				{
					int numNeurArrAlt[MAX_NUM_LAYERS_MLP]{ numNeurArr[0], numNeurArr[1], numNeurArr[1] / 2 };
					tt[4] = std::thread(taskMLP, 4, a, limMin, limMax, seedNo1, numNeurArrAlt, distrParam, ranges, checkPoints, lossFunction, plot, print1, importParam, exportParam, operation, statistics, std::ref(printer));
				}

				if (numThreads > 5)
				{
					int numNeurArrAlt[MAX_NUM_LAYERS_MLP]{ (numNeurArr[1] > 0) ? numNeurArr[1] : 1, numNeurArr[0], numNeurArr[0] / 2 };
					tt[5] = std::thread(taskMLP, 5, a, limMin, limMax, seedNo1, numNeurArrAlt, distrParam, ranges, checkPoints, lossFunction, plot, print1, importParam, exportParam, operation, statistics, std::ref(printer));
				}

				if (numThreads > 6)
				{
					int numNeurArrAlt[MAX_NUM_LAYERS_MLP]{ ((numNeurArr[1] > 0) ? numNeurArr[1] : 1) * 2, numNeurArr[1], numNeurArr[1] / 2 };
					tt[6] = std::thread(taskMLP, 6, a, limMin, limMax, seedNo1, numNeurArrAlt, distrParam, ranges, checkPoints, lossFunction, plot, print1, importParam, exportParam, operation, statistics, std::ref(printer));
				}

				if (numThreads > 7)
				{
					int numNeurArrAlt[MAX_NUM_LAYERS_MLP]{ numNeurArr[0] };
					tt[7] = std::thread(taskMLP, 7, a, limMin, limMax, seedNo1, numNeurArrAlt, distrParam, ranges, checkPoints, lossFunction, plot, print1, importParam, exportParam, operation, statistics, std::ref(printer));
				}
			}
		}
		else if (algorithm == MLPFAST)
		{
			//Creating the MLPFast threads.
			if (diffThreads == 0)
			{
				for (int i = 0; i < numThreads; i++)
				{
					tt[i] = std::thread(taskMLPFst, i, a, limMin, limMax, seedNo1, numNeurArr, distrParam, ranges, checkPoints, lossFunction, plot, print1, importParam, exportParam, operation, statistics, std::ref(printer), parall);
				}
			}
			else
			{
				tt[0] = std::thread(taskMLPFst, 0, a, limMin, limMax, seedNo1, numNeurArr, distrParam, ranges, checkPoints, lossFunction, plot, print1, importParam, exportParam, operation, statistics, std::ref(printer), parall);

				if (numThreads > 1)
				{
					int numNeurArrAlt[MAX_NUM_LAYERS_MLP]{ (numNeurArr[1] > 0) ? numNeurArr[1] : 1, numNeurArr[0] };
					tt[1] = std::thread(taskMLPFst, 1, a, limMin, limMax, seedNo1, numNeurArrAlt, distrParam, ranges, checkPoints, lossFunction, plot, print1, importParam, exportParam, operation, statistics, std::ref(printer), parall);
				}

				if (numThreads > 2)
				{
					int numNeurArrAlt[MAX_NUM_LAYERS_MLP]{ numNeurArr[0], numNeurArr[0] };
					tt[2] = std::thread(taskMLPFst, 2, a, limMin, limMax, seedNo1, numNeurArrAlt, distrParam, ranges, checkPoints, lossFunction, plot, print1, importParam, exportParam, operation, statistics, std::ref(printer), parall);
				}

				if (numThreads > 3)
				{
					int numNeurArrAlt[MAX_NUM_LAYERS_MLP]{ numNeurArr[0] * 2, numNeurArr[0], numNeurArr[0] / 2 };
					tt[3] = std::thread(taskMLPFst, 3, a, limMin, limMax, seedNo1, numNeurArrAlt, distrParam, ranges, checkPoints, lossFunction, plot, print1, importParam, exportParam, operation, statistics, std::ref(printer), parall);
				}

				if (numThreads > 4)
				{
					int numNeurArrAlt[MAX_NUM_LAYERS_MLP]{ numNeurArr[0], numNeurArr[1], numNeurArr[1] / 2 };
					tt[4] = std::thread(taskMLPFst, 4, a, limMin, limMax, seedNo1, numNeurArrAlt, distrParam, ranges, checkPoints, lossFunction, plot, print1, importParam, exportParam, operation, statistics, std::ref(printer), parall);
				}

				if (numThreads > 5)
				{
					int numNeurArrAlt[MAX_NUM_LAYERS_MLP]{ (numNeurArr[1] > 0) ? numNeurArr[1] : 1, numNeurArr[0], numNeurArr[0] / 2 };
					tt[5] = std::thread(taskMLPFst, 5, a, limMin, limMax, seedNo1, numNeurArrAlt, distrParam, ranges, checkPoints, lossFunction, plot, print1, importParam, exportParam, operation, statistics, std::ref(printer), parall);
				}

				if (numThreads > 6)
				{
					int numNeurArrAlt[MAX_NUM_LAYERS_MLP]{ ((numNeurArr[1] > 0) ? numNeurArr[1] : 1) * 2, numNeurArr[1], numNeurArr[1] / 2 };
					tt[6] = std::thread(taskMLPFst, 6, a, limMin, limMax, seedNo1, numNeurArrAlt, distrParam, ranges, checkPoints, lossFunction, plot, print1, importParam, exportParam, operation, statistics, std::ref(printer), parall);
				}

				if (numThreads > 7)
				{
					int numNeurArrAlt[MAX_NUM_LAYERS_MLP]{ numNeurArr[0] };
					tt[7] = std::thread(taskMLPFst, 7, a, limMin, limMax, seedNo1, numNeurArrAlt, distrParam, ranges, checkPoints, lossFunction, plot, print1, importParam, exportParam, operation, statistics, std::ref(printer), parall);
				}
			}
		}

		//Joining the threads.
		for (int i = 0; i < numThreads; i++)
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




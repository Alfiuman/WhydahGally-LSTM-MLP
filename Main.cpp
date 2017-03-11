
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

class Printer
{
private:
	std::mutex mu_;
public:
	Printer() { }
	void printError(const float& loss, const float& generalLoss, const int& numThr)
	{
		std::lock_guard<std::mutex> guard(mu_);
	
		PRINT("Final error for " << numThr << " thread is: "  << loss << " and generalLoss is: " << generalLoss) << "\n";
	}
};

void taskLSTM(const int& numThr, Importer* imp, const int& numCell, const bool& importParam, const float& max, const float& min, const int& seedNo, const int& testTimes, const int& viewsEach, const float& alpha, const bool& print, const bool& exportParam, const int& operation, const int& statistics, const int& lossFunct, const int& parall, Printer& printer)
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

void taskMLP(const int& numThr, Importer* imp, const float& limMin, const float& limMax, const float& seedNo1, const int& numNeur1, const int& numNeur2, const int& numNeur3, const int& numNeur4, const int& numNeur5, const int& numNeur6, const int& numNeur7, const int& numNeur8, const int& numNeur9, const int& numNeur10, const int& numNeur11, const int& numNeur12, const float& mu, const float& sigma, const int& ranDistr, const int& range1, const int& range2, const int& range3, const int& checkPoint1, const int& checkPoint2, const int& checkPoint3, const float& epsilon, const float& muAlpha, const float& sigmaAlpha, const int& lossFunction, const bool& plot, const bool& print, const float& seedNo2, const bool& importParam, const bool& exportParam, const int& operation, const int& statistics, Printer& printer)
{
	MultiLayerPerceptron* b = new MultiLayerPerceptron(*imp, limMin, limMax, seedNo1, numNeur1, numNeur2);

	if (importParam == 1)
	{
		b->importWeights();
	}

	if (operation == 1)
	{
		b->train(mu, sigma, ranDistr, range1, range2, range3, checkPoint1, checkPoint2, checkPoint3, epsilon, muAlpha, sigmaAlpha, lossFunction, plot, print, seedNo2);

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

void taskMLPFst(const int& numThr, Importer* imp, const float& limMin, const float& limMax, const float& seedNo1, const int& numNeur1, const int& numNeur2, const int& numNeur3, const int& numNeur4, const int& numNeur5, const int& numNeur6, const int& numNeur7, const int& numNeur8, const int& numNeur9, const int& numNeur10, const int& numNeur11, const int& numNeur12, const float& mu, const float& sigma, const int& ranDistr, const int& range1, const int& range2, const int& range3, const int& checkPoint1, const int& checkPoint2, const int& checkPoint3, const float& epsilon, const float& muAlpha, const float& sigmaAlpha, const int& lossFunction, const bool& plot, const bool& print, const float& seedNo2, const bool& importParam, const bool& exportParam, const int& operation, const int& statistics, Printer& printer, const int& parall)
{
	MLPFast* c = new MLPFast(*imp, limMin, limMax, seedNo1, numNeur1, numNeur2);

	if (importParam == 1)
	{
		c->importWeights();
	}

	if (operation == 1)
	{
		c->train(mu, sigma, ranDistr, range1, range2, range3, checkPoint1, checkPoint2, checkPoint3, epsilon, muAlpha, sigmaAlpha, lossFunction, plot, print, seedNo2, parall);

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

	int numThreads = 1;

	int hist = 0;
	std::string nameFile = "Data9";

	bool importParam = 0;
	bool exportParam = 0;

	int algorithm = LSTM;
	int operation = TRAIN;
	bool statistics = 0;
	bool diffThreads = 0;
	int parall = 0;
	float bias = 1;

	Importer* a = new Importer(hist, bias, nameFile);

	if (operation == TEST || operation == CLASSIFY)
	{
		importParam = 1;
		exportParam = 0;
		numThreads = 1;
	}

	if (numThreads == 1)
	{
		diffThreads = 0;
	}
	
	if (operation == CLASSIFY)
	{
		statistics = 0;
	}

	if (numThreads > 1)
	{
		exportParam = 0;
	}

	if (numThreads > 8 && diffThreads == 1)
	{
		numThreads = 8;
	}

	std::vector<std::thread> tt(numThreads);

	Printer printer;

	if (algorithm == 1)
	{
		// LSTM -------------------------------------------------------------------------------------------------
		
		int numCell = 20;
		float max = 1.0f;
		float min = -1.0f;
		float seedNo = 0.0f;
		int lossFunct = LOSSFUNCTSIMPLE;

		int testTimes = 50;
		int viewsEach = 1;
		float alpha = 0.11f;
		bool print = 1;

		if (numThreads != 1)
		{
			print = 0;
		}

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
	else if (algorithm == 2 || algorithm == 3)
	{
		// MLP --------------------------------------------------------------------------------------------------

		float limMin = -10.0f;
		float limMax = 10.0f;
		float seedNo1 = 0.0f;
		int numNeur1 = 18;
		int numNeur2 = 24;
		int numNeur3 = 0;
		int numNeur4 = 0;
		int numNeur5 = 0;
		int numNeur6 = 0;
		int numNeur7 = 0;
		int numNeur8 = 0;
		int numNeur9 = 0;
		int numNeur10 = 0;
		int numNeur11 = 0;
		int numNeur12 = 0;

		float mu = -10.0f;
		float sigma = 10.0f;
		int ranDistr = 0;
		int range1 = 1000;
		int range2 = 1000;
		int range3 = 500;
		int checkPoint1 = 100;
		int checkPoint2 = 100;
		int checkPoint3 = 100;
		float epsilon = 0.05f;
		float muAlpha = 0.4f;
		float sigmaAlpha = 0.1f;
		int lossFunction = LOSSFUNCTSIMPLE;
		bool print1 = 1;
		bool plot = 1;
		float seedNo2 = 0.0f;

		if (numThreads != 1)
		{
			print1 = 0;
		}

		if (algorithm == 2)
		{
			if (diffThreads == 0)
			{
				for (int i = 0; i < numThreads; i++)
				{
					tt[i] = std::thread(taskMLP, i, a, limMin, limMax, seedNo1, numNeur1, numNeur2, numNeur3, numNeur4, numNeur5, numNeur6, numNeur7, numNeur8, numNeur9, numNeur10, numNeur11, numNeur12, mu, sigma, ranDistr, range1, range2, range3, checkPoint1, checkPoint2, checkPoint3, epsilon, muAlpha, sigmaAlpha, lossFunction, plot, print1, seedNo2, importParam, exportParam, operation, statistics, std::ref(printer));
				}
			}
			else
			{
				tt[0] = std::thread(taskMLP, 0, a, limMin, limMax, seedNo1, 18, 24, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, mu, sigma, ranDistr, range1, range2, range3, checkPoint1, checkPoint2, checkPoint3, epsilon, muAlpha, sigmaAlpha, lossFunction, plot, print1, seedNo2, importParam, exportParam, operation, statistics, std::ref(printer));

				if (numThreads > 1)
				{
					tt[1] = std::thread(taskMLP, 1, a, limMin, limMax, seedNo1, 18, 12, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, mu, sigma, ranDistr, range1, range2, range3, checkPoint1, checkPoint2, checkPoint3, epsilon, muAlpha, sigmaAlpha, lossFunction, plot, print1, seedNo2, importParam, exportParam, operation, statistics, std::ref(printer));
				}

				if (numThreads > 2)
				{
					tt[2] = std::thread(taskMLP, 2, a, limMin, limMax, seedNo1, 24, 12, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, mu, sigma, ranDistr, range1, range2, range3, checkPoint1, checkPoint2, checkPoint3, epsilon, muAlpha, sigmaAlpha, lossFunction, plot, print1, seedNo2, importParam, exportParam, operation, statistics, std::ref(printer));
				}

				if (numThreads > 3)
				{
					tt[3] = std::thread(taskMLP, 3, a, limMin, limMax, seedNo1, 18, 24, 12, 0, 0, 0, 0, 0, 0, 0, 0, 0, mu, sigma, ranDistr, range1, range2, range3, checkPoint1, checkPoint2, checkPoint3, epsilon, muAlpha, sigmaAlpha, lossFunction, plot, print1, seedNo2, importParam, exportParam, operation, statistics, std::ref(printer));
				}

				if (numThreads > 4)
				{
					tt[4] = std::thread(taskMLP, 4, a, limMin, limMax, seedNo1, 24, 18, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, mu, sigma, ranDistr, range1, range2, range3, checkPoint1, checkPoint2, checkPoint3, epsilon, muAlpha, sigmaAlpha, lossFunction, plot, print1, seedNo2, importParam, exportParam, operation, statistics, std::ref(printer));
				}

				if (numThreads > 5)
				{
					tt[5] = std::thread(taskMLP, 5, a, limMin, limMax, seedNo1, 12, 24, 18, 0, 0, 0, 0, 0, 0, 0, 0, 0, mu, sigma, ranDistr, range1, range2, range3, checkPoint1, checkPoint2, checkPoint3, epsilon, muAlpha, sigmaAlpha, lossFunction, plot, print1, seedNo2, importParam, exportParam, operation, statistics, std::ref(printer));
				}

				if (numThreads > 6)
				{
					tt[6] = std::thread(taskMLP, 6, a, limMin, limMax, seedNo1, 12, 18, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, mu, sigma, ranDistr, range1, range2, range3, checkPoint1, checkPoint2, checkPoint3, epsilon, muAlpha, sigmaAlpha, lossFunction, plot, print1, seedNo2, importParam, exportParam, operation, statistics, std::ref(printer));
				}

				if (numThreads > 7)
				{
					tt[7] = std::thread(taskMLP, 7, a, limMin, limMax, seedNo1, 18, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, mu, sigma, ranDistr, range1, range2, range3, checkPoint1, checkPoint2, checkPoint3, epsilon, muAlpha, sigmaAlpha, lossFunction, plot, print1, seedNo2, importParam, exportParam, operation, statistics, std::ref(printer));
				}
			}
		}
		else if (algorithm == 3)
		{
			if (diffThreads == 0)
			{
				for (int i = 0; i < numThreads; i++)
				{
					tt[i] = std::thread(taskMLPFst, i, a, limMin, limMax, seedNo1, numNeur1, numNeur2, numNeur3, numNeur4, numNeur5, numNeur6, numNeur7, numNeur8, numNeur9, numNeur10, numNeur11, numNeur12, mu, sigma, ranDistr, range1, range2, range3, checkPoint1, checkPoint2, checkPoint3, epsilon, muAlpha, sigmaAlpha, lossFunction, plot, print1, seedNo2, importParam, exportParam, operation, statistics, std::ref(printer), parall);
				}
			}
			else
			{
				tt[0] = std::thread(taskMLPFst, 0, a, limMin, limMax, seedNo1, 18, 24, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, mu, sigma, ranDistr, range1, range2, range3, checkPoint1, checkPoint2, checkPoint3, epsilon, muAlpha, sigmaAlpha, lossFunction, plot, print1, seedNo2, importParam, exportParam, operation, statistics, std::ref(printer), parall);

				if (numThreads > 1)
				{
					tt[1] = std::thread(taskMLPFst, 1, a, limMin, limMax, seedNo1, 18, 12, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, mu, sigma, ranDistr, range1, range2, range3, checkPoint1, checkPoint2, checkPoint3, epsilon, muAlpha, sigmaAlpha, lossFunction, plot, print1, seedNo2, importParam, exportParam, operation, statistics, std::ref(printer), parall);
				}

				if (numThreads > 2)
				{
					tt[2] = std::thread(taskMLPFst, 2, a, limMin, limMax, seedNo1, 24, 12, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, mu, sigma, ranDistr, range1, range2, range3, checkPoint1, checkPoint2, checkPoint3, epsilon, muAlpha, sigmaAlpha, lossFunction, plot, print1, seedNo2, importParam, exportParam, operation, statistics, std::ref(printer), parall);
				}

				if (numThreads > 3)
				{
					tt[3] = std::thread(taskMLPFst, 3, a, limMin, limMax, seedNo1, 18, 24, 12, 0, 0, 0, 0, 0, 0, 0, 0, 0, mu, sigma, ranDistr, range1, range2, range3, checkPoint1, checkPoint2, checkPoint3, epsilon, muAlpha, sigmaAlpha, lossFunction, plot, print1, seedNo2, importParam, exportParam, operation, statistics, std::ref(printer), parall);
				}

				if (numThreads > 4)
				{
					tt[4] = std::thread(taskMLPFst, 4, a, limMin, limMax, seedNo1, 24, 18, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, mu, sigma, ranDistr, range1, range2, range3, checkPoint1, checkPoint2, checkPoint3, epsilon, muAlpha, sigmaAlpha, lossFunction, plot, print1, seedNo2, importParam, exportParam, operation, statistics, std::ref(printer), parall);
				}

				if (numThreads > 5)
				{
					tt[5] = std::thread(taskMLPFst, 5, a, limMin, limMax, seedNo1, 12, 24, 18, 0, 0, 0, 0, 0, 0, 0, 0, 0, mu, sigma, ranDistr, range1, range2, range3, checkPoint1, checkPoint2, checkPoint3, epsilon, muAlpha, sigmaAlpha, lossFunction, plot, print1, seedNo2, importParam, exportParam, operation, statistics, std::ref(printer), parall);
				}

				if (numThreads > 6)
				{
					tt[6] = std::thread(taskMLPFst, 6, a, limMin, limMax, seedNo1, 12, 18, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, mu, sigma, ranDistr, range1, range2, range3, checkPoint1, checkPoint2, checkPoint3, epsilon, muAlpha, sigmaAlpha, lossFunction, plot, print1, seedNo2, importParam, exportParam, operation, statistics, std::ref(printer), parall);
				}

				if (numThreads > 7)
				{
					tt[7] = std::thread(taskMLPFst, 7, a, limMin, limMax, seedNo1, 18, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, mu, sigma, ranDistr, range1, range2, range3, checkPoint1, checkPoint2, checkPoint3, epsilon, muAlpha, sigmaAlpha, lossFunction, plot, print1, seedNo2, importParam, exportParam, operation, statistics, std::ref(printer), parall);
				}
			}
		}
	}
	// FINAL ------------------------------------------------------------------------------------------------

	for (int i = 0; i < numThreads; i++)
	{
		tt[i].join();
	}

	delete a;

#else

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
	float bias = 1.0f;

	int numCell = 0;
	float max = 0.0f;
	float min = 0.0f;
	float seedNo = 0.0f;
	int lossFunct = 0;
	int testTimes = 0;
	int viewsEach = 0;
	float alpha = 0.0f;
	bool print = 0;

	float limMax = 0.0f;
	float limMin = 0.0f;
	float seedNo1 = 0.0f;
	int numNeur1 = 0;
	int numNeur2 = 0;
	int numNeur3 = 0;
	int numNeur4 = 0;
	int numNeur5 = 0;
	int numNeur6 = 0;
	int numNeur7 = 0;
	int numNeur8 = 0;
	int numNeur9 = 0;
	int numNeur10 = 0;
	int numNeur11 = 0;
	int numNeur12 = 0;
	int ranDistr = 0;
	float mu = 0.0f;
	float sigma = 0.0f;
	int range1 = 1000;
	int range2 = 1000;
	int range3 = 2500;
	int checkPoint1 = 100;
	int checkPoint2 = 100;
	int checkPoint3 = 100;
	float epsilon = 0.05f;
	float muAlpha = 0.4f;
	float sigmaAlpha = 0.1f;
	int lossFunction = LOSSFUNCTSIMPLE;
	bool print1 = 0;
	bool plot = 1;
	float seedNo2 = 0.0f;

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
			algorithm = 1;
		}
		catch (const std::out_of_range& e)
		{
			algorithm = 1;
		}

		if (algorithm > 3 || algorithm < 1)
		{
			algorithm = 1;
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
			operation = 2;
		}
		catch (const std::out_of_range& e)
		{
			operation = 2;
		}

		if (operation > 3 || operation < 1)
		{
			operation = 2;
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
			numThreads = 1;
		}
		catch (const std::out_of_range& e)
		{
			numThreads = 1;
		}

		if (numThreads < 1)
		{
			numThreads = 1;
		}

		if (operation == TEST || operation == CLASSIFY)
		{
			numThreads = 1;
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
			hist = 0;
		}
		catch (const std::out_of_range& e)
		{
			hist = 0;
		}

		if (hist < 0)
		{
			hist = 0;
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
			bias = 1.0f;
		}
		catch (const std::out_of_range& e)
		{
			bias = 1.0f;
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
			importParam = 0;
		}
		catch (const std::out_of_range& e)
		{
			importParam = 0;
		}

		if (operation == TEST || operation == CLASSIFY)
		{
			importParam = 1;
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
			exportParam = 0;
		}
		catch (const std::out_of_range& e)
		{
			exportParam = 0;
		}

		if (operation == TEST || operation == CLASSIFY || numThreads > 1)
		{
			exportParam = 0;
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
			statistics = 0;
		}
		catch (const std::out_of_range& e)
		{
			statistics = 0;
		}

		if (operation == CLASSIFY)
		{
			statistics = 0;
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
				diffThreads = 0;
			}
			catch (const std::out_of_range& e)
			{
				diffThreads = 0;
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
			parall = 0;
		}
		catch (const std::out_of_range& e)
		{
			parall = 0;
		}

		if (parall < 0 || parall > 2)
		{
			parall = 0;
		}

		PRINT("You chose " << parall << ".\n");
		answer.clear();
#endif
		if (algorithm == 1)
		{
			PRINT("\nHow many cells do you want?\n");
			std::getline(std::cin, answer);

			try
			{
				numCell = std::stoi(answer);
			}
			catch (const std::invalid_argument& e)
			{
				numCell = 20;
			}
			catch (const std::out_of_range& e)
			{
				numCell = 20;
			}

			if (numCell < 0)
			{
				numCell = 20;
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
				max = 1.0f;
			}
			catch (const std::out_of_range& e)
			{
				max = 1.0f;
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
				min = -1.0f;
			}
			catch (const std::out_of_range& e)
			{
				min = -1.0f;
			}

			if (min >= max)
			{
				min = max - 1.0f;
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
				seedNo = 0.0f;
			}
			catch (const std::out_of_range& e)
			{
				seedNo = 0.0f;
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
				lossFunct = 0;
			}
			catch (const std::out_of_range& e)
			{
				lossFunct = 0;
			}

			if (lossFunct < 0 || lossFunct > 4)
			{
				lossFunct = 0;
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
				testTimes = 1000;
			}
			catch (const std::out_of_range& e)
			{
				testTimes = 1000;
			}

			if (testTimes < 0)
			{
				testTimes = 1000;
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
				viewsEach = 100;
			}
			catch (const std::out_of_range& e)
			{
				viewsEach = 100;
			}

			if (viewsEach < 0)
			{
				viewsEach = 100;
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
				alpha = 0.11f;
			}
			catch (const std::out_of_range& e)
			{
				alpha = 0.11f;
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
				print = 0;
			}
			catch (const std::out_of_range& e)
			{
				print = 0;
			}

			if (numThreads != 1)
			{
				print = 0;
			}

			PRINT("You chose " << print << ".\n");
			answer.clear();
		}
		else if (algorithm == 2 || algorithm == 3)
		{
			PRINT("\nWhat is the MAX value you want for the random initialization?\n");
			std::getline(std::cin, answer);

			try
			{
				limMax = std::stof(answer);
			}
			catch (const std::invalid_argument& e)
			{
				limMax = 1.0f;
			}
			catch (const std::out_of_range& e)
			{
				limMax = 1.0f;
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
				limMin = -1.0f;
			}
			catch (const std::out_of_range& e)
			{
				limMin = -1.0f;
			}

			if (limMin > limMax)
			{
				limMin = limMax - 1.0f;
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
				seedNo1 = 0.0f;
			}
			catch (const std::out_of_range& e)
			{
				seedNo1 = 0.0f;
			}

			PRINT("You chose " << seedNo1 << ".\n");
			answer.clear();

			PRINT("\nHow many neurons do you want in the first layer?\n");
			std::getline(std::cin, answer);

			try
			{
				numNeur1 = std::stoi(answer);
			}
			catch (const std::invalid_argument& e)
			{
				numNeur1 = 10;
			}
			catch (const std::out_of_range& e)
			{
				numNeur1 = 10;
			}

			if (numNeur1 < 0)
			{
				numNeur1 = 10;
			}

			PRINT("You chose " << numNeur1 << ".\n");
			answer.clear();

			PRINT("\nHow many neurons do you want in the second layer?\n");
			std::getline(std::cin, answer);

			try
			{
				numNeur2 = std::stoi(answer);
			}
			catch (const std::invalid_argument& e)
			{
				numNeur2 = 10;
			}
			catch (const std::out_of_range& e)
			{
				numNeur2 = 10;
			}

			if (numNeur2 < 0)
			{
				numNeur2 = 10;
			}

			PRINT("You chose " << numNeur2 << ".\n");
			answer.clear();

			if (numNeur2 != 0)
			{
				PRINT("\nHow many neurons do you want in the third layer?\n");
				std::getline(std::cin, answer);

				try
				{
					numNeur3 = std::stoi(answer);
				}
				catch (const std::invalid_argument& e)
				{
					numNeur3 = 10;
				}
				catch (const std::out_of_range& e)
				{
					numNeur3 = 10;
				}

				if (numNeur3 < 0)
				{
					numNeur3 = 10;
				}

				PRINT("You chose " << numNeur3 << ".\n");
				answer.clear();

				if (numNeur3 != 0)
				{
					PRINT("\nHow many neurons do you want in the fourth layer?\n");
					std::getline(std::cin, answer);

					try
					{
						numNeur4 = std::stoi(answer);
					}
					catch (const std::invalid_argument& e)
					{
						numNeur4 = 10;
					}
					catch (const std::out_of_range& e)
					{
						numNeur4 = 10;
					}

					if (numNeur4 < 0)
					{
						numNeur4 = 10;
					}

					PRINT("You chose " << numNeur4 << ".\n");
					answer.clear();

					if (numNeur4 != 0)
					{
						PRINT("\nHow many neurons do you want in the fifth layer?\n");
						std::getline(std::cin, answer);

						try
						{
							numNeur5 = std::stoi(answer);
						}
						catch (const std::invalid_argument& e)
						{
							numNeur5 = 10;
						}
						catch (const std::out_of_range& e)
						{
							numNeur5 = 10;
						}

						if (numNeur5 < 0)
						{
							numNeur5 = 10;
						}

						PRINT("You chose " << numNeur5 << ".\n");
						answer.clear();

						if (numNeur5 != 0)
						{
							PRINT("\nHow many neurons do you want in the sixth layer?\n");
							std::getline(std::cin, answer);

							try
							{
								numNeur6 = std::stoi(answer);
							}
							catch (const std::invalid_argument& e)
							{
								numNeur6 = 10;
							}
							catch (const std::out_of_range& e)
							{
								numNeur6 = 10;
							}

							if (numNeur6 < 0)
							{
								numNeur6 = 10;
							}

							PRINT("You chose " << numNeur6 << ".\n");
							answer.clear();

							if (numNeur6 != 0)
							{
								PRINT("\nHow many neurons do you want in the seventh layer?\n");
								std::getline(std::cin, answer);

								try
								{
									numNeur7 = std::stoi(answer);
								}
								catch (const std::invalid_argument& e)
								{
									numNeur7 = 10;
								}
								catch (const std::out_of_range& e)
								{
									numNeur7 = 10;
								}

								if (numNeur7 < 0)
								{
									numNeur7 = 10;
								}

								PRINT("You chose " << numNeur7 << ".\n");
								answer.clear();

								if (numNeur7 != 0)
								{
									PRINT("\nHow many neurons do you want in the eighth layer?\n");
									std::getline(std::cin, answer);

									try
									{
										numNeur8 = std::stoi(answer);
									}
									catch (const std::invalid_argument& e)
									{
										numNeur8 = 10;
									}
									catch (const std::out_of_range& e)
									{
										numNeur8 = 10;
									}

									if (numNeur8 < 0)
									{
										numNeur8 = 10;
									}

									PRINT("You chose " << numNeur8 << ".\n");
									answer.clear();

									if (numNeur8 != 0)
									{
										PRINT("\nHow many neurons do you want in the ninth layer?\n");
										std::getline(std::cin, answer);

										try
										{
											numNeur9 = std::stoi(answer);
										}
										catch (const std::invalid_argument& e)
										{
											numNeur9 = 10;
										}
										catch (const std::out_of_range& e)
										{
											numNeur9 = 10;
										}

										if (numNeur9 < 0)
										{
											numNeur9 = 10;
										}

										PRINT("You chose " << numNeur9 << ".\n");
										answer.clear();

										if (numNeur9 != 0)
										{
											PRINT("\nHow many neurons do you want in the tenth layer?\n");
											std::getline(std::cin, answer);

											try
											{
												numNeur10 = std::stoi(answer);
											}
											catch (const std::invalid_argument& e)
											{
												numNeur10 = 10;
											}
											catch (const std::out_of_range& e)
											{
												numNeur10 = 10;
											}

											if (numNeur10 < 0)
											{
												numNeur10 = 10;
											}

											PRINT("You chose " << numNeur10 << ".\n");
											answer.clear();

											if (numNeur10 != 0)
											{
												PRINT("\nHow many neurons do you want in the eleventh layer?\n");
												std::getline(std::cin, answer);

												try
												{
													numNeur11 = std::stoi(answer);
												}
												catch (const std::invalid_argument& e)
												{
													numNeur11 = 10;
												}
												catch (const std::out_of_range& e)
												{
													numNeur11 = 10;
												}

												if (numNeur11 < 0)
												{
													numNeur11 = 10;
												}

												PRINT("You chose " << numNeur11 << ".\n");
												answer.clear();

												if (numNeur11 != 0)
												{
													PRINT("\nHow many neurons do you want in the twelfth layer?\n");
													std::getline(std::cin, answer);

													try
													{
														numNeur12 = std::stoi(answer);
													}
													catch (const std::invalid_argument& e)
													{
														numNeur12 = 10;
													}
													catch (const std::out_of_range& e)
													{
														numNeur12 = 10;
													}

													if (numNeur12 < 0)
													{
														numNeur12 = 10;
													}

													PRINT("You chose " << numNeur12 << ".\n");
													answer.clear();
												}
											}
										}
									}
								}
							}
						}
					}
				}
			}

			PRINT("\nWhich random distribution do you want? (0 uniform, 1 normal)\n");
			std::getline(std::cin, answer);

			try
			{
				ranDistr = std::stoi(answer);
			}
			catch (const std::invalid_argument& e)
			{
				ranDistr = 0;
			}
			catch (const std::out_of_range& e)
			{
				ranDistr = 0;
			}

			if (ranDistr < 0 || ranDistr > 1)
			{
				ranDistr = 0;
			}

			PRINT("You chose " << ranDistr << ".\n");
			answer.clear();

			PRINT("\nWhat the value of mu do you want?\n");
			std::getline(std::cin, answer);

			try
			{
				mu = std::stof(answer);
			}
			catch (const std::invalid_argument& e)
			{
				mu = 0.0f;
			}
			catch (const std::out_of_range& e)
			{
				mu = 0.0f;
			}

			PRINT("You chose " << mu << ".\n");
			answer.clear();

			PRINT("\nWhat the value of sigma do you want?\n");
			std::getline(std::cin, answer);

			try
			{
				sigma = std::stof(answer);
			}
			catch (const std::invalid_argument& e)
			{
				sigma = 0.0f;
			}
			catch (const std::out_of_range& e)
			{
				sigma = 0.0f;
			}

			if (sigma < mu && ranDistr == 0)
			{
				sigma = mu + 1.0f;
			}

			PRINT("You chose " << sigma << ".\n");
			answer.clear();

			PRINT("\nHow many iterations do you ant for the first phase?\n");
			std::getline(std::cin, answer);

			try
			{
				range1 = std::stoi(answer);
			}
			catch (const std::invalid_argument& e)
			{
				range1 = 1000;
			}
			catch (const std::out_of_range& e)
			{
				range1 = 1000;
			}

			if (range1 < 0)
			{
				range1 = 1000;
			}

			PRINT("You chose " << range1 << ".\n");
			answer.clear();

			PRINT("\nHow many iterations do you ant for the second phase?\n");
			std::getline(std::cin, answer);

			try
			{
				range2 = std::stoi(answer);
			}
			catch (const std::invalid_argument& e)
			{
				range2 = 1000;
			}
			catch (const std::out_of_range& e)
			{
				range2 = 1000;
			}

			if (range2 < 0)
			{
				range2 = 1000;
			}

			PRINT("You chose " << range2 << ".\n");
			answer.clear();

			PRINT("\nHow many iterations do you ant for the third phase?\n");
			std::getline(std::cin, answer);

			try
			{
				range3 = std::stoi(answer);
			}
			catch (const std::invalid_argument& e)
			{
				range3 = 1000;
			}
			catch (const std::out_of_range& e)
			{
				range3 = 1000;
			}

			if (range3 < 0)
			{
				range3 = 1000;
			}

			PRINT("You chose " << range3 << ".\n");
			answer.clear();

			PRINT("\nAt the multiple of which iteration number do you want the checkpoint for the first phase?\n");
			std::getline(std::cin, answer);

			try
			{
				checkPoint1 = std::stoi(answer);
			}
			catch (const std::invalid_argument& e)
			{
				checkPoint1 = 100;
			}
			catch (const std::out_of_range& e)
			{
				checkPoint1 = 100;
			}

			if (checkPoint1 < 0)
			{
				checkPoint1 = 100;
			}

			PRINT("You chose " << checkPoint1 << ".\n");
			answer.clear();

			PRINT("\nAt the multiple of which iteration number do you want the checkpoint for the second phase?\n");
			std::getline(std::cin, answer);

			try
			{
				checkPoint2 = std::stoi(answer);
			}
			catch (const std::invalid_argument& e)
			{
				checkPoint2 = 100;
			}
			catch (const std::out_of_range& e)
			{
				checkPoint2 = 100;
			}

			if (checkPoint2 < 0)
			{
				checkPoint2 = 100;
			}

			PRINT("You chose " << checkPoint2 << ".\n");
			answer.clear();

			PRINT("\nAt the multiple of which iteration number do you want the checkpoint for the third phase?\n");
			std::getline(std::cin, answer);

			try
			{
				checkPoint3 = std::stoi(answer);
			}
			catch (const std::invalid_argument& e)
			{
				checkPoint3 = 100;
			}
			catch (const std::out_of_range& e)
			{
				checkPoint3 = 100;
			}

			if (checkPoint3 < 0)
			{
				checkPoint3 = 100;
			}

			PRINT("You chose " << checkPoint3 << ".\n");
			answer.clear();

			PRINT("\nWhich value of epsilon do you want?\n");
			std::getline(std::cin, answer);

			try
			{
				epsilon = std::stof(answer);
			}
			catch (const std::invalid_argument& e)
			{
				epsilon = 1.0f;
			}
			catch (const std::out_of_range& e)
			{
				epsilon = 1.0f;
			}

			if (epsilon < 0)
			{
				epsilon = 1.0f;
			}

			PRINT("You chose " << epsilon << ".\n");
			answer.clear();

			PRINT("\nWhich value of mu Alpha do you want?\n");
			std::getline(std::cin, answer);

			try
			{
				muAlpha = std::stof(answer);
			}
			catch (const std::invalid_argument& e)
			{
				muAlpha = 1.0f;
			}
			catch (const std::out_of_range& e)
			{
				muAlpha = 1.0f;
			}

			PRINT("You chose " << muAlpha << ".\n");
			answer.clear();

			PRINT("\nWhich value of sigma Alpha do you want?\n");
			std::getline(std::cin, answer);

			try
			{
				sigmaAlpha = std::stof(answer);
			}
			catch (const std::invalid_argument& e)
			{
				sigmaAlpha = 1.0f;
			}
			catch (const std::out_of_range& e)
			{
				sigmaAlpha = 1.0f;
			}

			if (sigmaAlpha < 0)
			{
				sigmaAlpha = 1.0f;
			}

			PRINT("You chose " << sigmaAlpha << ".\n");
			answer.clear();

			PRINT("\nWhich loss function do you want to use? (0 simple, 1 log, 2 logPow3, 3 pow3, 4 pow3LogPow3)\n");
			std::getline(std::cin, answer);

			try
			{
				lossFunction = std::stoi(answer);
			}
			catch (const std::invalid_argument& e)
			{
				lossFunction = 0;
			}
			catch (const std::out_of_range& e)
			{
				lossFunction = 0;
			}

			if (lossFunction < 0 || lossFunction > 4)
			{
				lossFunction = 0;
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
				print1 = 0;
			}
			catch (const std::out_of_range& e)
			{
				print1 = 0;
			}

			if (numThreads != 1)
			{
				print1 = 0;
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
				seedNo2 = std::stof(answer);
			}
			catch (const std::invalid_argument& e)
			{
				seedNo2 = 0.0f;
			}
			catch (const std::out_of_range& e)
			{
				seedNo2 = 0.0f;
			}

			PRINT("You chose " << seedNo2 << ".\n");
			answer.clear();
		}

		Importer* a = new Importer(hist, bias, nameFile);

		if (numThreads > 8 && diffThreads == 1)
		{
			numThreads = 8;
		}

		std::vector<std::thread> tt(numThreads);

		Printer printer;

		if (algorithm == 1)
		{
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
		else if (algorithm == 2)
		{
			if (diffThreads == 0)
			{
				for (int i = 0; i < numThreads; i++)
				{
					tt[i] = std::thread(taskMLP, i, a, limMin, limMax, seedNo1, numNeur1, numNeur2, numNeur3, numNeur4, numNeur5, numNeur6, numNeur7, numNeur8, numNeur9, numNeur10, numNeur11, numNeur12, mu, sigma, ranDistr, range1, range2, range3, checkPoint1, checkPoint2, checkPoint3, epsilon, muAlpha, sigmaAlpha, lossFunction, plot, print1, seedNo2, importParam, exportParam, operation, statistics, std::ref(printer));
				}
			}
			else
			{
				tt[0] = std::thread(taskMLP, 0, a, limMin, limMax, seedNo1, 18, 24, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, mu, sigma, ranDistr, range1, range2, range3, checkPoint1, checkPoint2, checkPoint3, epsilon, muAlpha, sigmaAlpha, lossFunction, plot, print1, seedNo2, importParam, exportParam, operation, statistics, std::ref(printer));

				if (numThreads > 1)
				{
					tt[1] = std::thread(taskMLP, 1, a, limMin, limMax, seedNo1, 18, 12, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, mu, sigma, ranDistr, range1, range2, range3, checkPoint1, checkPoint2, checkPoint3, epsilon, muAlpha, sigmaAlpha, lossFunction, plot, print1, seedNo2, importParam, exportParam, operation, statistics, std::ref(printer));
				}

				if (numThreads > 2)
				{
					tt[2] = std::thread(taskMLP, 2, a, limMin, limMax, seedNo1, 24, 12, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, mu, sigma, ranDistr, range1, range2, range3, checkPoint1, checkPoint2, checkPoint3, epsilon, muAlpha, sigmaAlpha, lossFunction, plot, print1, seedNo2, importParam, exportParam, operation, statistics, std::ref(printer));
				}

				if (numThreads > 3)
				{
					tt[3] = std::thread(taskMLP, 3, a, limMin, limMax, seedNo1, 18, 24, 12, 0, 0, 0, 0, 0, 0, 0, 0, 0, mu, sigma, ranDistr, range1, range2, range3, checkPoint1, checkPoint2, checkPoint3, epsilon, muAlpha, sigmaAlpha, lossFunction, plot, print1, seedNo2, importParam, exportParam, operation, statistics, std::ref(printer));
				}

				if (numThreads > 4)
				{
					tt[4] = std::thread(taskMLP, 4, a, limMin, limMax, seedNo1, 24, 18, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, mu, sigma, ranDistr, range1, range2, range3, checkPoint1, checkPoint2, checkPoint3, epsilon, muAlpha, sigmaAlpha, lossFunction, plot, print1, seedNo2, importParam, exportParam, operation, statistics, std::ref(printer));
				}

				if (numThreads > 5)
				{
					tt[5] = std::thread(taskMLP, 5, a, limMin, limMax, seedNo1, 12, 24, 18, 0, 0, 0, 0, 0, 0, 0, 0, 0, mu, sigma, ranDistr, range1, range2, range3, checkPoint1, checkPoint2, checkPoint3, epsilon, muAlpha, sigmaAlpha, lossFunction, plot, print1, seedNo2, importParam, exportParam, operation, statistics, std::ref(printer));
				}

				if (numThreads > 6)
				{
					tt[6] = std::thread(taskMLP, 6, a, limMin, limMax, seedNo1, 12, 18, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, mu, sigma, ranDistr, range1, range2, range3, checkPoint1, checkPoint2, checkPoint3, epsilon, muAlpha, sigmaAlpha, lossFunction, plot, print1, seedNo2, importParam, exportParam, operation, statistics, std::ref(printer));
				}

				if (numThreads > 7)
				{
					tt[7] = std::thread(taskMLP, 7, a, limMin, limMax, seedNo1, 18, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, mu, sigma, ranDistr, range1, range2, range3, checkPoint1, checkPoint2, checkPoint3, epsilon, muAlpha, sigmaAlpha, lossFunction, plot, print1, seedNo2, importParam, exportParam, operation, statistics, std::ref(printer));
				}
			}
		}
		else if (algorithm == 3)
		{
			if (diffThreads == 0)
			{
				for (int i = 0; i < numThreads; i++)
				{
					tt[i] = std::thread(taskMLPFst, i, a, limMin, limMax, seedNo1, numNeur1, numNeur2, numNeur3, numNeur4, numNeur5, numNeur6, numNeur7, numNeur8, numNeur9, numNeur10, numNeur11, numNeur12, mu, sigma, ranDistr, range1, range2, range3, checkPoint1, checkPoint2, checkPoint3, epsilon, muAlpha, sigmaAlpha, lossFunction, plot, print1, seedNo2, importParam, exportParam, operation, statistics, std::ref(printer), parall);
				}
			}
			else
			{
				tt[0] = std::thread(taskMLPFst, 0, a, limMin, limMax, seedNo1, 18, 24, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, mu, sigma, ranDistr, range1, range2, range3, checkPoint1, checkPoint2, checkPoint3, epsilon, muAlpha, sigmaAlpha, lossFunction, plot, print1, seedNo2, importParam, exportParam, operation, statistics, std::ref(printer), parall);

				if (numThreads > 1)
				{
					tt[1] = std::thread(taskMLPFst, 1, a, limMin, limMax, seedNo1, 18, 12, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, mu, sigma, ranDistr, range1, range2, range3, checkPoint1, checkPoint2, checkPoint3, epsilon, muAlpha, sigmaAlpha, lossFunction, plot, print1, seedNo2, importParam, exportParam, operation, statistics, std::ref(printer), parall);
				}

				if (numThreads > 2)
				{
					tt[2] = std::thread(taskMLPFst, 2, a, limMin, limMax, seedNo1, 24, 12, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, mu, sigma, ranDistr, range1, range2, range3, checkPoint1, checkPoint2, checkPoint3, epsilon, muAlpha, sigmaAlpha, lossFunction, plot, print1, seedNo2, importParam, exportParam, operation, statistics, std::ref(printer), parall);
				}

				if (numThreads > 3)
				{
					tt[3] = std::thread(taskMLPFst, 3, a, limMin, limMax, seedNo1, 18, 24, 12, 0, 0, 0, 0, 0, 0, 0, 0, 0, mu, sigma, ranDistr, range1, range2, range3, checkPoint1, checkPoint2, checkPoint3, epsilon, muAlpha, sigmaAlpha, lossFunction, plot, print1, seedNo2, importParam, exportParam, operation, statistics, std::ref(printer), parall);
				}

				if (numThreads > 4)
				{
					tt[4] = std::thread(taskMLPFst, 4, a, limMin, limMax, seedNo1, 24, 18, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, mu, sigma, ranDistr, range1, range2, range3, checkPoint1, checkPoint2, checkPoint3, epsilon, muAlpha, sigmaAlpha, lossFunction, plot, print1, seedNo2, importParam, exportParam, operation, statistics, std::ref(printer), parall);
				}

				if (numThreads > 5)
				{
					tt[5] = std::thread(taskMLPFst, 5, a, limMin, limMax, seedNo1, 12, 24, 18, 0, 0, 0, 0, 0, 0, 0, 0, 0, mu, sigma, ranDistr, range1, range2, range3, checkPoint1, checkPoint2, checkPoint3, epsilon, muAlpha, sigmaAlpha, lossFunction, plot, print1, seedNo2, importParam, exportParam, operation, statistics, std::ref(printer), parall);
				}

				if (numThreads > 6)
				{
					tt[6] = std::thread(taskMLPFst, 6, a, limMin, limMax, seedNo1, 12, 18, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, mu, sigma, ranDistr, range1, range2, range3, checkPoint1, checkPoint2, checkPoint3, epsilon, muAlpha, sigmaAlpha, lossFunction, plot, print1, seedNo2, importParam, exportParam, operation, statistics, std::ref(printer), parall);
				}

				if (numThreads > 7)
				{
					tt[7] = std::thread(taskMLPFst, 7, a, limMin, limMax, seedNo1, 18, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, mu, sigma, ranDistr, range1, range2, range3, checkPoint1, checkPoint2, checkPoint3, epsilon, muAlpha, sigmaAlpha, lossFunction, plot, print1, seedNo2, importParam, exportParam, operation, statistics, std::ref(printer), parall);
				}
			}
		}

		for (int i = 0; i < numThreads; i++)
		{
			tt[i].join();
		}

		delete a;

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
	PRINT("\n\nEND") << "\n";

	std::cin.get();
	return 0;
}




#include "Importer.h"

namespace WhydahGally 
{
	namespace Base
	{
		//It populates the Importer from a selected txt file.
		Importer::Importer(const int& historyLength, const float& bias, const std::string& fileName)
			: fileName_(fileName), numNumbers_(0), numRows_(0), numColumns_(0), historyLength_(historyLength), yMat_(1), bias_(bias)
		{
			//Opening and reading the txt file.
			if (fileName_ == "")
			{
				PRINT("Please enter the File Name: \n");
				getline(std::cin, fileName_);
			}
			
			file_.open(fileName_ + ".txt");

			if (file_)
			{
				while (file_ >> element_)
				{
					numNumbers_++;
				}

				file_.clear();
				file_.seekg(file_.beg);

				while (getline(file_, line_))
				{
					++numRows_;
				}

				file_.clear();
				file_.seekg(file_.beg);

				numColumns_ = numNumbers_ / numRows_;

				headers_.resize(numColumns_);

				std::vector<std::vector<float>> ee(numRows_ - 1, std::vector<float>());

				//Populating the header vector and creating the conditions to populate the other vectors.
				for (int i = 0; i < numNumbers_; ++i)
				{
					if (i < numColumns_)
					{
						file_ >> headers_[i];
					}
					else
					{
						file_ >> number_;
						ee.at((i / numColumns_) - 1).push_back(number_);
					}
				}

				float a = ee.at(0).at(0);
				int counter01 = 0;
				std::vector<std::vector<std::vector<float>>> uu;
				std::vector<std::vector<float>> aa;

				uu.push_back(aa);

				for (int i = 0; i < ee.size(); i++)
				{
					if (ee[i][0] != a)
					{
						counter01++;
						a = ee.at(i).at(0);
						uu.push_back(aa);
					}

					uu[counter01].push_back(ee.at(i));
				}

				std::vector<std::vector<std::vector<std::vector<float*>>>> uu2;

				uu2.resize(counter01 + 1, std::vector<std::vector<std::vector<float*>>>(3));

				for (int i = 0; i <= counter01; i++)
				{
					for (int j = 0; j < 3; j++)
					{
						if (j < 2)
						{
							uu2.at(i).at(j).resize(1, std::vector<float*>(uu.at(i).size()));
							for (int k = 0; k < uu[i].size(); k++)
							{
								uu2.at(i).at(j).at(0).at(k) = &uu.at(i).at(k).at(j);
							}
						}
						else
						{
							uu2.at(i).at(j).resize(uu.at(i).size(), std::vector<float*>(numColumns_ - 2));

							for (int k = 0; k < uu[i].size(); k++)
							{
								for (int m = 0; m < numColumns_ - 2; m++)
								{
									uu2.at(i).at(j).at(k).at(m) = &uu.at(i).at(k).at(m + 2);
								}
							}
						}
					}
				}

				std::vector<std::vector<std::vector<float**>>> xx;
				xx.resize(counter01 + 1);

				if (historyLength_ == 0 || historyLength_ > ee.size())
				{
					historyLength_ = ee.size();
				}

				for (int i = 0; i <= counter01; i++)
				{
					xx[i].resize(uu.at(i).size() - historyLength_ + 1);
					for (int j = 0; j < uu.at(i).size() - historyLength_ + 1; j++)
					{
						xx.at(i).at(j).resize(historyLength_ * (numColumns_ - 2));
					}
				}

				for (int i = 0; i < xx.size(); i++)
				{
					for (int j = 0; j < xx.at(i).size(); j++)
					{
						for (int k = 0; k < xx.at(i).at(j).size(); k++)
						{
							xx.at(i).at(j).at(k) = &uu2.at(i).at(2).at(j + (k % historyLength_)).at(k / historyLength_);
						}
					}
				}

				std::vector<std::vector<float**>> xp;

				for (int i = 0; i < xx.size(); i++)
				{
					for (int j = 0; j < xx.at(i).size(); j++)
					{
						xp.push_back(xx.at(i).at(j));
					}
				}

				std::vector<float*> yp;

				for (int i = 0; i < uu2.size(); i++)
				{
					for (int j = historyLength_ - 1; j < uu2.at(i).at(1).at(0).size(); j++)
					{
						yp.push_back(uu2.at(i).at(1).at(0).at(j));
					}
				}

				//Populating the bias vector.
				for (int i = 0; i < yp.size(); i++)
				{
					bias_.push_back(bias);
				}

				x_.resize(xp.size());

				for (int i = 0; i < xp.size(); i++)
				{
					x_.at(i).resize(xp.at(i).size());
				}

				//Populating the X vector.
				for (int i = 0; i < xp.size(); i++)
				{
					for (int j = 0; j < xp.at(i).size(); j++)
					{
						x_.at(i).at(j) = **xp.at(i).at(j);
					}
				}

				y_.resize(yp.size());

				//Populating the Y vector.
				for (int i = 0; i < yp.size(); i++)
				{
					y_.at(i) = *yp.at(i);
				}

				yMat_.resize(y_.size());

				//Populating the Y matrix.
				for (int i = 0; i < y_.size(); i++)
				{
					yMat_.elements_[i] = y_[i];
				}

				series_.resize(ee.size());

				//Populating the series vector.
				for (int i = 0; i < ee.size(); i++)
				{
					series_.at(i) = ee.at(i).at(0);
				}

				yy_.resize(ee.size());

				//Populating the YY vector.
				for (int i = 0; i < ee.size(); i++)
				{
					yy_.at(i) = ee.at(i).at(1);
				}

				xx_.resize(ee.size());

				for (int i = 0; i < ee.size(); i++)
				{
					xx_.at(i).resize(ee.at(i).size() - 2);
				}

				//Populating the XX vector.
				for (int i = 0; i < ee.size(); i++)
				{
					for (int j = 0; j < ee.at(i).size() - 2; j++)
					{
						xx_.at(i).at(j) = ee.at(i).at(j + 2);
					}
				}

				PRINT("\nData imported.\n\n");
			}
			else
			{
				PRINT("\n\nWrong file name.\n");
				this->~Importer();

				PRINT("\nEND\nPress Enter to exit.\n");
				std::cin.get();
				exit(0);
			}
		}

		Importer::~Importer()
		{
			
		}
	}
}





#pragma once

#include <iostream>
#include <string>
#include <fstream>
#include <vector>

#include "Definitions.h"
#include "Matrix.h"

namespace WhydahGally
{
	namespace Base
	{
		class Importer
		{
		private:
			int numNumbers_;
			int numRows_;
			int numColumns_;
			int historyLength_;
			float number_;

			std::string element_;
			std::string fileName_;
			std::string line_;
			std::ifstream file_;
			std::vector<float> series_;
			std::vector<float> y_;
			std::vector<float> yy_;
			std::vector<float> bias_;
			std::vector<std::vector<float>> x_;
			std::vector<std::vector<float>> xx_;
			std::vector<std::string> headers_;

			Matrix yMat_;

		public:
			Importer(const int& historyLength, const float& bias, const std::string& fileName = "");
			~Importer();

			inline int getNumColumns() const { return numColumns_; }
			inline int getNumRows() const { return numRows_; }
			inline int getHistoryLength() const { return historyLength_; }
			inline float getBias(int n) const { return bias_[n]; }
			inline std::string getHeader(short int n) const { return headers_[n]; }
			inline std::vector<float> getSeries() const { return series_; }
			inline std::vector<float> getY() const { return y_; }
			inline std::vector<float> getYY() const { return yy_; }
			inline std::vector<std::vector<float>> getX() const { return x_; }
			inline std::vector<std::vector<float>> getXX() const { return xx_; }
			inline Matrix getYMat() const { return yMat_; }

			inline void setBias(const float& bias, const int& x) { bias_.at(x) = bias; }
		};
	}
}
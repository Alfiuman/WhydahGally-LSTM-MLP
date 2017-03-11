#pragma once

#include <iostream>

#include "Definitions.h"

namespace WhydahGally
{
	struct Matrix
	{
		int rows_;
		int cols_;
		float* elements_;

		Matrix(int x, int y = 1);
		Matrix(const Matrix& x);
		~Matrix();

		void resize(int x = 1, int y = 1);
		void assign(const float& x);

		Matrix& copy(const Matrix& right);
	};
}
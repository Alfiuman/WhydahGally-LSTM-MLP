#pragma once

#include <iostream>

#include "Definitions.h"

namespace WhydahGally
{
	template<typename T> struct Matrix
	{
		//Struct built around a dynamic allocated array in order to have a proper functioning matrix.
		//Everything is public in order to simulate as much as possible a normal array.
		int rows_;
		int cols_;
		T* elements_;

		Matrix<T>(int x, int y = 1)
		: rows_(x), cols_(y)
		{
			if (x <= 0)
			{
				rows_ = 1;
			}
			else
			{
				rows_ = x;
			}

			if (y <= 0)
			{
				cols_ = 1;
			}
			else
			{
				cols_ = y;
			}

			elements_ = new T[rows_ * cols_];

			std::fill(elements_, elements_ + (rows_ * cols_), 0.0f);
		}

		Matrix<T>(const Matrix& x)
		{
			//Constructing a matrix copying another one.
			if (x.rows_ <= 0)
			{
				rows_ = 1;
			}
			else
			{
				rows_ = x.rows_;
			}

			if (x.cols_ <= 0)
			{
				cols_ = 1;
			}
			else
			{
				cols_ = x.cols_;
			}

			elements_ = new T[rows_ * cols_];

			for (int i = 0; i < rows_; ++i)
			{
				for (int j = 0; j < cols_; ++j)
				{
					elements_[i * cols_ + j] = x.elements_[i * x.cols_ + j];
				}
			}
		}

		~Matrix<T>()
		{
			delete[] elements_;
		}

		void resize(int x = 1, int y = 1)
		{
			//Resizing the matrix and zeroing its elements.
			if (x <= 0)
			{
				rows_ = 1;
			}
			else
			{
				rows_ = x;
			}

			if (y <= 0)
			{
				cols_ = 1;
			}
			else
			{
				cols_ = y;
			}

			delete[] elements_;
			elements_ = new T[rows_ * cols_];

			std::fill(elements_, elements_ + (rows_ * cols_), 0.0f);;
		}

		void assign(T x)
		{
			//Populating the entire matix with a value.
			std::fill(elements_, elements_ + (rows_ * cols_), x);;
		}

		Matrix& copy(const Matrix& right)
		{
			//Reconstructing an existing matrix copying another one.
			if (right.rows_ <= 0)
			{
				rows_ = 1;
			}
			else
			{
				rows_ = right.rows_;
			}

			if (right.cols_ <= 0)
			{
				cols_ = 1;
			}
			else
			{
				cols_ = right.cols_;
			}

			delete[] elements_;
			elements_ = new T[rows_ * cols_];

			for (int i = 0; i < rows_; ++i)
			{
				for (int j = 0; j < cols_; ++j)
				{
					elements_[i * cols_ + j] = right.elements_[i * right.cols_ + j];
				}
			}

			return *this;
		}
	};
}
#include "Matrix.h"

namespace WhydahGally
{
	Matrix::Matrix(int x, int y)
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

		elements_ = new float[rows_ * cols_];

		for (int i = 0; i < (rows_ * cols_); i++)
		{
			elements_[i] = 0.0f;
		}
	}

	Matrix::Matrix(const Matrix& x)
	{
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

		elements_ = new float[rows_ * cols_];

		for (int i = 0; i < rows_; i++)
		{
			for (int j = 0; j < cols_; j++)
			{
				elements_[i * cols_ + j] = x.elements_[i * x.cols_ + j];
			}
		}
	}

	Matrix::~Matrix()
	{
		delete[] elements_;
	}

	void Matrix::resize(int x, int y)
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

		delete[] elements_;
		elements_ = new float[rows_ * cols_];

		for (int i = 0; i < (rows_ * cols_); i++)
		{
			elements_[i] = 0.0f;
		}
	}

	void Matrix::assign(const float& x)
	{
		for (int i = 0; i < (rows_ * cols_); i++)
		{
			elements_[i] = x;
		}
	}

	Matrix& Matrix::copy(const Matrix& right)
	{
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
		elements_ = new float[rows_ * cols_];

		for (int i = 0; i < rows_; i++)
		{
			for (int j = 0; j < cols_; j++)
			{
				elements_[i * cols_ + j] = right.elements_[i * right.cols_ + j];
			}
		}

		return *this;
	}
}
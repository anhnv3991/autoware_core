#ifndef GMATRIX_H_
#define GMATRIX_H_

#include <cuda.h>
#include <cuda_runtime.h>

#include "common.hpp"
#include "point_types.hpp"
#include "data_accessor.hpp"


namespace gpu
{

template <int Rows, int Cols, typename Scalar>
class Matrix
{
public:
  CUDAH Matrix() 
  {
    offset_ = 0;
    buffer_ = nullptr;
  }

  CUDAH Matrix(int offset, double * buffer)
  : offset_(offset), buffer_(buffer)
  {}

  CUDAH Matrix(const Matrix & other)
  : offset_(other.offset_), buffer_(other.buffer_)
  {}

  CUDAH Matrix(Matrix && other)
  : offset_(other.offset_), buffer_(other.buffer_)
  {
    other.offset_ = 0;
    other.buffer_ = nullptr;
  }

  CUDAH Matrix & operator=(const Matrix & other)
  {
    if (*this != other) {
      offset_ = other.offset_;
      buffer_ = other.buffer_;
    }

    return *this;
  }

  CUDAH Matrix & operator=(Matrix && other)
  {
    if (*this != other) {
      offset_ = other.offset_;
      buffer_ = other.buffer_;

      other.offset_ = 0;
      other.buffer_ = nullptr;
    }

    return *this;
  }

  CUDAH int rows() const { return Rows; }
  CUDAH int cols() const { return Cols; }
  CUDAH int offset() const { return offset_; }

  CUDAH Scalar * buffer() const { return buffer_; }

  // Indexing operators
  CUDAH Scalar & operator()(int row, int col) { return buffer_[(row * Cols + col) * offset_]; }

  CUDAH const Scalar & operator()(int row, int col) const
  {
    return buffer_[(row * Cols + col) * offset_];
  }

  CUDAH Scalar & operator()(int index) { return buffer_[index * offset_]; }
  CUDAH const Scalar & operator()(int index) const { return buffer_[index * offset_]; }

  CUDAH Scalar at(int row, int col) const { return at(row * Cols_ + col); }
  CUDAH Scalar at(int index) const { return buffer_[index * offset_]; }

  // Element-wise operations
  CUDAH bool operator*=(Scalar val)
  {
    constexpr int ele_num = Rows * Cols;
    for (int i = 0, idx = 0; i < ele_num; ++i, idx += offset_) 
    {
        buffer_[idx] *= val;
    }

    return true;
  }

  CUDAH bool operator/=(Scalar val)
  {
    if (val == 0) 
    {
      return false;
    }

    constexpr int ele_num = Rows * Cols;

    for (int i = 0, idx = 0; i < ele_num; ++i, idx += offset) 
    {
      buffer_[idx] /= val;
    }

    return true;
  }

  CUDAH bool transpose(Matrix<Cols, Rows, Scalar> & output)
  {
    for (int i = 0; i < Rows_; i++) 
    {
      for (int j = 0; j < Cols_; j++) 
      {
        output(j, i) = (*this)(i, j);
      }
    }

    return true;
  }

  // Only applicable for 3x3 matrix or below
  CUDAH bool inverse(Matrix & output);

  // Extract a row and a column of the matrix
  CUDAH Matrix<Rows, 1, Scalar> col(int index)
  {
    return Matrix<Rows, 1, Scalar>(offset_ * Cols, buffer_ + index * offset_);
  }

  CUDAH Matrix<1, Cols, Scalar> row(int index)
  {
    return Matrix<1, Cols, Scalar>(offset_, buffer_ + index * Cols * offset_);
  }

protected:
  Scalar * buffer_;
  int offset_;
};

// Only applicable for 3x3 matrix or below
template <int Rows, int Cols, typename Scalar>
CUDAH bool Matrix::inverse(Matrix & output)
{
  if (Rows != Cols || Rows == 0 || Cols == 0) return false;

  if (Rows == 1) 
  {
    if (buffer_[0] != 0) {
      output(0, 0) = 1 / buffer_[0];
    }
    else
    {
      return false;
    }
  }

  if (Rows == 2) 
  {
    double det = at(0, 0) * at(1, 1) - at(0, 1) * at(1, 0);

    if (det != 0) 
    {
      output(0, 0) = at(1, 1) / det;
      output(0, 1) = -at(0, 1) / det;

      output(1, 0) = -at(1, 0) / det;
      output(1, 1) = at(0, 0) / det;
    } else
    {
      return false;
    }
  }

  if (Rows == 3) 
  {
    double det = at(0, 0) * at(1, 1) * at(2, 2) + at(0, 1) * at(1, 2) * at(2, 0) +
                 at(1, 0) * at(2, 1) * at(0, 2) - at(0, 2) * at(1, 1) * at(2, 0) -
                 at(0, 1) * at(1, 0) * at(2, 2) - at(0, 0) * at(1, 2) * at(2, 1);
    double idet = 1.0 / det;

    if (det != 0) 
    {
      output(0, 0) = (at(1, 1) * at(2, 2) - at(1, 2) * at(2, 1)) * idet;
      output(0, 1) = -(at(0, 1) * at(2, 2) - at(0, 2) * at(2, 1)) * idet;
      output(0, 2) = (at(0, 1) * at(1, 2) - at(0, 2) * at(1, 1)) * idet;

      output(1, 0) = -(at(1, 0) * at(2, 2) - at(1, 2) * at(2, 0)) * idet;
      output(1, 1) = (at(0, 0) * at(2, 2) - at(0, 2) * at(2, 0)) * idet;
      output(1, 2) = -(at(0, 0) * at(1, 2) - at(0, 2) * at(1, 0)) * idet;

      output(2, 0) = (at(1, 0) * at(2, 1) - at(1, 1) * at(2, 0)) * idet;
      output(2, 1) = -(at(0, 0) * at(2, 1) - at(0, 1) * at(2, 0)) * idet;
      output(2, 2) = (at(0, 0) * at(1, 1) - at(0, 1) * at(1, 0)) * idet;
    } 
    else 
    {
      return false;
    }
  }

  return true;
}

// Specialization for the accessor
template <>
template <>
CUDAH void Accessor<Matrix<3, 1, float>>::store<Point>(Point & in_point)
{
  data_(0) = in_point.x;
  data_(1) = in_point.y;
  data_(2) = in_point.z;
}

template <>
template <>
CUDAH void Accessor<Matrix<3, 1, double>>::store<Point>(Point & in_point)
{
  data_(0) = in_point.x;
  data_(1) = in_point.y;
  data_(2) = in_point.z;
}

}  // namespace gpu

#endif

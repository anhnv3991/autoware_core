#ifndef MATRIX_HOST_H_
#define MATRIX_HOST_H_

#include "matrix.hpp"
#include "matrix_device.hpp"

#include <Eigen/Dense>

#include <thrust/copy.h>
#include <thrust/device_vector.h>

namespace gpu
{

// A pointer-like class used by CUDA kernels
class DeviceMatrixPointer;

// Array of same-sized matrices
// Actually it is a 3D tensor
class MatrixArray
{
public:
  MatrixArray() : rows_(0), cols_(0), mat_num_(0) {}

  MatrixArray(int rows, int cols, int mat_num) : rows_(rows), cols_(cols), mat_num_(mat_num)
  {
    data_.resize(rows_ * cols_ * mat_num_);
  }

  // Get a device-side matrix pointer, used by CUDA kernels
  DeviceMatrixPointer data();

  // For debug, get a matrix at index i
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> at(int i)
  {
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> mat(rows, cols);

    for (int i = 0; i < rows_; ++i) {
      for (int j = 0; j < cols_; ++j) {
        mat(i, j) = data_[(i * cols_ + j) * mat_num_];
      }
    }

    return mat;
  }

private:
  thrust::device_vector<double> data_;
  int rows_, cols_, mat_num_;
};

class DeviceMatrixPointer
{
public:
  DeviceMatrixPointer()
  {
    rows_ = cols_ = mat_num_ = 0;
    buffer_ = nullptr;
  }

  CUDAH DeviceMatrixPointer(int rows, int cols, int mat_num, double * buffer)
  : rows_(rows), cols_(cols), mat_num_(mat_num), buffer_(buffer)
  {
  }

  CUDAH DeviceMatrixPointer(const DeviceMatrixPointer & other)
  : rows_(other.rows_), cols_(other.cols_), mat_num_(other.mat_num_), buffer_(other.buffer_)
  {
  }

  CUDAH DeviceMatrixPointer(DeviceMatrixPointer && other)
  : rows_(other.rows_), cols_(other.cols_), mat_num_(other.mat_num_), buffer_(other.buffer_)
  {
    other.rows_ = other.cols_ = other.mat_num_ = 0;
    other.buffer_ = nullptr;
  }

  CUDAH DeviceMatrixPointer & operator=(const DeviceMatrixPointer & other)
  {
    if (*this != other) {
      rows_ = other.rows_;
      cols_ = other.cols_;
      mat_num_ = other.mat_num_;
      buffer_ = other.buffer_;
    }

    return *this;
  }

  CUDAH DeviceMatrixPointer & operator=(DeviceMatrixPointer && other)
  {
    if (*this != other) {
      rows_ = other.rows_;
      cols_ = other.cols_;
      mat_num_ = other.mat_num_;
      buffer_ = other.buffer_;

      other.rows_ = other.cols_ = other.mat_num_ = 0;
      other.buffer_ = nullptr;
    }

    return *this;
  }

  // Get a matrix at index i
  CUDAH Matrix operator()(int idx)
  {
    return Matrix(rows_, cols_, mat_num_, buffer_ + idx * mat_num_);
  }

private:
  double * buffer_;
  int rows_, cols_, mat_num_;
}

DeviceMatrixPointer
MatrixArray::data()
{
  return DeviceMatrixPointer(rows_, cols_, mat_num_, thrust::raw_pointer_cast(data_.data()));
}

}  // namespace gpu

#endif

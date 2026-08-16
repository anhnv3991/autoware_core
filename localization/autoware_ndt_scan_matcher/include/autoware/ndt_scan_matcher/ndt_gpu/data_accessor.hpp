#ifndef NDT_GPU_DATA_ACCESSOR_HPP_
#define NDT_GPU_DATA_ACCESSOR_HPP_

#include <cuda.h>
#include <cuda_runtime.h>

#include "common.hpp"

namespace gpu
{

// Data accessor to unify the load/store operations of various data types
// By default, it is similar to the usual load/store operations
// For some data types, we may need specialization to optimize the load/store
template <typename T>
struct Accessor
{
    Accessor() = delete;
    CUDAH Accessor(T & data) : data_(data) {}

    template <typename InType>
    CUDAH void store(InType & in_value) { data_ = in_value; }

    template <typename OutType>
    CUDAH OutType load() { return data_; }

    T & data_;
};
}

#endif
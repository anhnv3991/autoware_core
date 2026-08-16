#ifndef NDT_GPU_UTILITY_HPP_
#define NDT_GPU_UTILITY_HPP_

#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <limits>

#include "debug.hpp"
#include "common.hpp"
#include "matrix.hpp"

namespace gpu
{

namespace util
{

template <
    typename DevPtrType, 
    typename EleType, 
    typename MaxOpType, 
    typename MinOpType,
    typename SharedArrayType,
    typename Scalar,
    int tuple_size
>
__global__ void findMinMax(
    const DevPtrType input, 
    size_t ele_num, 
    CmpType cmp, 
    EleType * max_val,
    EleType * min_val
)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    extern __shared__ Scalar sdata[];   // A shared buffer for block-level reduction
    EleType local_min = std::numeric_limits<EleType>::max();
    EleType local_max = std::numeric_limits<EleType>::lowest();
    Matrix<tuple_size, 1, Scalar> shared_min(blockDim.x, sdata + threadIdx.x);
    Matrix<tuple_size, 1, Scalar> shared_max(blockDim.x, sdata + threadIdx.x + blockDim.x * tuple_size);

    for (int i = index; i < ele_num; i += stride) {
        auto ele = input[i];

        local_min = MinOpType(ele, local_min);
        local_max = MaxOpType(ele, local_max);
    }

    // Write the local values to the shared data
    shared_min.store(local_min);




}

    
}

}

#endif
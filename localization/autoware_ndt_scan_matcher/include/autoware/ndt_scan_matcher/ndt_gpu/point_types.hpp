#ifndef NDT_GPU_POINT_TYPES_HPP_
#define NDT_GPU_POINT_TYPES_HPP_

#include <cuda.h>
#include <cuda_runtime.h>

#include "common.hpp"

namespace gpu
{

struct alignas(16) Point
{
    CUDAH Point(float ix, float iy, float iz) : x(ix), y(iy), z(iz) {}
    CUDAH Point& operator=(other Point & p) 
    {
        x = p.x;
        y = p.y;
        z = p.z;

        return *this;
    }

    CUDAH Point& operator=(Point && p) 
    {
        x = p.x;
        y = p.y;
        z = p.z;

        return *this;
    }

    float x, y, z;
};


}

#endif
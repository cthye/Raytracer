#pragma once
#ifndef RAYTRACING_UTILS_H
#define RAYTRACING_UTILS_H

#include <iostream>
#include <cmath>

#undef M_PI
#define M_PI 3.141592653589793f

#define kInfinity FLT_MAX


//return val between low and high, otherwise return the bound
__host__ __device__ float clamp(const float& low, const float& high, const float& v) {
    // return std::max(low, std::min(high, v));
    if(v >= low && v <= high) return v;
    if(v < low) return low;
    return high;
}

//convert degree to rad
__host__ __device__ float deg2rad(const float& degree) { return degree * M_PI / 180.0f; }

// solve quadratic..
__host__ __device__ inline  bool solveQuadratic(const float &a, const float &b, const float &c, float &x0, float &x1)
{
    float discr = b * b - 4 * a * c;
    if (discr < 0) return false;
    else if (discr == 0) x0 = x1 = - 0.5 * b / a;
    else {
        float q = (b > 0) ?
                  -0.5 * (b + sqrt(discr)) :
                  -0.5 * (b - sqrt(discr));
        x0 = q / a;
        x1 = c / q;
    }
    if (x0 > x1) {
        float tmp = x0;
        x0 = x1;
        x1 = tmp;
    };
    return true;
}

#endif //RAYTRACING_UTILS_H
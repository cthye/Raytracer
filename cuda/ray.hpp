#pragma once
#ifndef RAYTRACING_RAY_H
#define RAYTRACING_RAY_H

#include "vector.hpp"

class Ray {
public:
    __host__ __device__ Ray() {}
    __host__ __device__ Ray(const Vector3f& p, const Vector3f& d, float _t = 0.0f) : origin(p), direction(d), t(_t) {
        // direction_inv = Vector3f(1./direction.x, 1./direction.y, 1./direction.z);
        t_min = 0.0f;
        t_max = FLT_MAX;
    }

    __device__ Vector3f operator() (float t) const { return origin + direction * t; }

    Vector3f origin;
    Vector3f direction, direction_inv;
    float t, t_min, t_max;
};

#endif //RAYTRACING_RAY_H
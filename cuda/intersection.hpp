#pragma once
#ifndef RAYTRACING_INTERSECTION_H
#define RAYTRACING_INTERSECTION_H

#include "vector.hpp"

class Material;

class Intersection {
public:
    bool happened;
    Vector3f coords;
    Vector3f normal;
    Vector3f emit;
    Material *m;
    float t;
    __host__ __device__ Intersection() {
        happened = false;
        coords = Vector3f();
        normal = Vector3f();
        t = FLT_MAX;
        m = NULL;
    }
};

#endif //RAYTRACING_INTERSECTION_H
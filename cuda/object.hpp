#pragma once
#ifndef RAYTRACING_OBJECT_H
#define RAYTRACING_OBJECT_H

#include "ray.hpp"
#include "intersection.hpp"
#include "utils.hpp"
#include "vector.hpp"

class Material;

class Object {
public:
    __host__ __device__ Object() {}
    __host__ __device__ Object(Material* _m) : m(_m){}
    __host__ __device__ virtual bool intersect(const Ray& ray, float tMin, float tMax, Intersection& intersec) const = 0;
    Material *m;
};

class Sphere : public Object {
public:
    __host__ __device__ Sphere() {}
    __host__ __device__ Sphere(Vector3f cen, float r, Material *_m) : center(cen), radius(r), Object(_m) {}
    __host__ __device__ virtual bool intersect(const Ray& ray, float tmin, float tmax, Intersection& result) const {
        result.happened = false;
        Vector3f L = ray.origin - center;
        float a = dot(ray.direction, ray.direction);
        float b = 2 * dot(ray.direction, L);
        float c = dot(L, L) - radius * radius;
        float t0, t1;
        if (!solveQuadratic(a, b, c, t0, t1)) return result.happened;
        if (t0 <= tmin) t0 = t1;
        if (t0 <= tmin || t0 >= tmax) {    
            return result.happened;
        }

        result.happened=true;
        result.coords = Vector3f(ray.origin + ray.direction * t0);
        result.normal = normalize(Vector3f(result.coords - center));
        result.m = this->m;
        result.t = t0;
        return result.happened;
    }
    
    Vector3f center;
    float radius;
};

#endif //RAYTRACING_OBJECT_H

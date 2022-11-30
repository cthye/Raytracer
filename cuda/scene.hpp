#pragma once
#ifndef RAYTRACING_SCENE_H
#define RAYTRACING_SCENE_H

#include "object.hpp"

class Scene {
public:
    float imageAspectRatio;
    int width;
    int height;
    Object **objList;
    int objCnt;
    int capacity;
 
    __host__ __device__ Scene() {}
    __host__ __device__ Scene(float w, float isr, Object ** l, int _capacity) {
        width = w;
        imageAspectRatio = isr;
        height = static_cast<int>(width / imageAspectRatio);
        objList = l;
        objCnt = 0;
        capacity = _capacity;
    }
    __host__ __device__ bool Add(Object *obj) {
        if(objCnt == capacity) {
            // std::cerr << "reach scene capacity" << std::endl;
            return false;
        }
        objList[objCnt++] = obj;
        return true;
    }

    //todo: BVH
    __host__ __device__ bool hit(const Ray& r, float t_min, float t_max, Intersection& rec) const {
        bool hit = false;
        Intersection tmp;
        float nearest = t_max;
        for (int i = 0; i < objCnt; i++) {
            if (objList[i]->intersect(r, t_min, nearest, tmp)) {
                hit = true;
                nearest = tmp.t;
                rec = tmp;
            }
        }
        return hit;
    }
};

#endif //RAYTRACING_SCENE_H
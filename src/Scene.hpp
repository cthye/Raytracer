//
// Created by Göksu Güvendiren on 2019-05-14.
//

#pragma once

#include <vector>
#include "Vector.hpp"
#include "Object.hpp"
#include "BVH.hpp"
#include "Ray.hpp"
#include "global.hpp"

class Scene
{
public:
    // setting up options
    int width;
    int height; 

    Vector3f backgroundColor = Vector3f(0.235294, 0.67451, 0.843137);
    int maxDepth = 1;
    float RussianRoulette = 0.8;
    std::vector<std::shared_ptr<Object> > objects;

    Scene(int w, int h) : width(w), height(h) {}
    void Add(std::shared_ptr<Object> object) { objects.push_back(object); }
    const std::vector<std::shared_ptr<Object> >& get_objects() const { return objects; }
    Intersection intersect(const Ray& ray) const;
    BVHAccel *bvh;
    void buildBVH();
    Vector3f castRay(const Ray &ray) const;
    Vector3f shader(Intersection p, Vector3f wo) const;
    void sampleLight(Intersection &pos, float &pdf) const;
};
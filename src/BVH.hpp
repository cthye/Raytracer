//
// Created by LEI XU on 5/16/19.
//

#ifndef RAYTRACING_BVH_H
#define RAYTRACING_BVH_H
#include <vector>
#include <memory>
#include "Object.hpp"
#include "Ray.hpp"
#include "Bounds3.hpp"
#include "Intersection.hpp"
#include "Vector.hpp"


struct BVHBuildNode {
    Bounds3 bounds;
    std::shared_ptr<BVHBuildNode> left;
    std::shared_ptr<BVHBuildNode> right;
    std::shared_ptr<Object> object;
    float area;

public:
    BVHBuildNode(){
        bounds = Bounds3();
        left = nullptr;right = nullptr;
        object = nullptr;
    }
};


class BVHAccel {

public:
    // BVHAccel Public Methods
    BVHAccel(std::vector<std::shared_ptr<Object>> p);
    ~BVHAccel();

    Intersection Intersect(const Ray &ray) const;
    Intersection getIntersection(std::shared_ptr<BVHBuildNode> node, const Ray& ray)const;
    //intersect with primitive
    bool IntersectP(const Ray &ray) const;
    std::shared_ptr<BVHBuildNode> root;
    std::shared_ptr<BVHBuildNode> recursiveBuild(std::vector<std::shared_ptr<Object>>);

    void getSample(std::shared_ptr<BVHBuildNode> node, float p, Intersection &pos, float &pdf);
    void Sample(Intersection &pos, float &pdf);

    std::vector<std::shared_ptr<Object>> primitives;
};



#endif //RAYTRACING_BVH_H

//
// Created by LEI XU on 5/16/19.
//

#ifndef RAYTRACING_INTERSECTION_H
#define RAYTRACING_INTERSECTION_H
#include "Vector.hpp"
#include "Material.hpp"
class Object;
class Sphere;

struct Intersection
{
    Intersection(){
        happened=false;
        normal=Vector3f();
        distance= std::numeric_limits<double>::max();
        m=nullptr;
    }
    bool happened;
    Vector3f coords;
    Vector3f normal;
    Vector3f emit;
    double distance;
    std::shared_ptr<Material> m;
};
#endif //RAYTRACING_INTERSECTION_H

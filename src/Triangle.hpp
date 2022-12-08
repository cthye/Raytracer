#ifndef RAYTRACING_TRIANGLE_H
#define RAYTRACING_TRIANGLE_H

#include "BVH.hpp"
#include "Intersection.hpp"
#include "Material.hpp"
#include "OBJ_Loader.hpp"
#include "Object.hpp"
#include "Triangle.hpp"
#include <cassert>
#include <array>

class Triangle : public Object
{
public:
    Vector3f v0, v1, v2; 
    Vector3f e1, e2;     // 2 edges v1-v0, v2-v0;
    Vector3f normal;
    float area;
    std::shared_ptr<Material> m;

    std::string name; // used for debug

    Triangle(Vector3f _v0, Vector3f _v1, Vector3f _v2, std::shared_ptr<Material> _m = nullptr)
        : v0(_v0), v1(_v1), v2(_v2), m(_m)
    {
        e1 = v1 - v0;
        e2 = v2 - v0;
        normal = normalize(crossProduct(e1, e2));
        area = crossProduct(e1, e2).norm()*0.5f;
    }

    Triangle(Vector3f _v0, Vector3f _v1, Vector3f _v2, std::string _name, std::shared_ptr<Material> _m = nullptr)
        : v0(_v0), v1(_v1), v2(_v2), m(_m), name(_name)
    {
        e1 = v1 - v0;
        e2 = v2 - v0;
        normal = normalize(crossProduct(e1, e2));
        area = crossProduct(e1, e2).norm()*0.5f;
    }

    Intersection getIntersection(Ray ray) override;
    Bounds3 getBounds() override {
        return Union(Bounds3(v0, v1), v2);
    }
    void Sample(Intersection &pos, float &pdf) override {
        float x = std::sqrt(get_random_float()), y = get_random_float();
        float r = std::sqrt(x);
        pos.coords = v0 * (1.0f - r) + v1 * (r * (1.0f - y)) + v2 * (r * y);
        pos.normal = this->normal;
        pos.happened = 1;
        pdf = 1.0f / area;
    }
    float getArea() override {
        return area;
    }
    bool hasEmit() override {
        return m->hasEmission();
    }
    std::string getName() const override {
        return name;
    }
};

inline Intersection Triangle::getIntersection(Ray ray)
{
    Intersection inter;

    if (dotProduct(ray.direction, normal) > 0)
        return inter;
    double u, v, t_tmp = 0;
    Vector3f pvec = crossProduct(ray.direction, e2); // s1
    double det = dotProduct(e1, pvec);
    if (fabs(det) < EPSILON)
        return inter;

    double det_inv = 1. / det;
    Vector3f tvec = ray.origin - v0; // s
    u = dotProduct(tvec, pvec) * det_inv;
    if (u < 0 || u > 1)
        return inter;
    Vector3f qvec = crossProduct(tvec, e1); // s2
    v = dotProduct(ray.direction, qvec) * det_inv;
    if (v < 0 || u + v > 1)
        return inter;
    t_tmp = dotProduct(e2, qvec) * det_inv;

    if (t_tmp < 0)
        return inter;

    inter.coords = ray.origin + t_tmp * ray.direction;
    inter.distance = t_tmp;
    inter.happened = true;
    inter.normal = this->normal;
    inter.m = this->m;

    return inter;
}


# endif
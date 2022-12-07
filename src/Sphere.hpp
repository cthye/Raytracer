#ifndef RAYTRACING_SPHERE_H
#define RAYTRACING_SPHERE_H

#include <string>

#include "Object.hpp"
#include "Vector.hpp"
#include "Bounds3.hpp"
#include "Material.hpp"

class Sphere : public Object{
public:
    Vector3f center;
    float radius, radius2;
    std::shared_ptr<Material> m;
    float area;
    std::string name;
    Sphere(const Vector3f &c, const float &r, std::shared_ptr<Material> mt) : center(c), radius(r), radius2(r * r), m(mt), area(4 * M_PI *r *r) {}
    Intersection getIntersection(Ray ray){
        Intersection result;
        result.happened = false;
        Vector3f L = ray.origin - center;
        float a = dotProduct(ray.direction, ray.direction);
        float b = 2 * dotProduct(ray.direction, L);
        float c = dotProduct(L, L) - radius2;
        float t0, t1;
        if (!solveQuadratic(a, b, c, t0, t1)) return result;
        if (t0 < 0) t0 = t1;
        if (t0 < 0) return result;
        result.happened=true;
        result.coords = Vector3f(ray.origin + ray.direction * t0);
        result.normal = normalize(Vector3f(result.coords - center));
        result.m = this->m;
        result.distance = t0;
        return result;

    }
    Vector3f evalDiffuseColor(const Vector2f &st)const {
        //return m->getColor();
    }
    Bounds3 getBounds(){
        return Bounds3(center - Vector3f(radius, radius, radius),
                       center + Vector3f(radius, radius, radius));
    }
    void Sample(Intersection &pos, float &pdf){
        float theta = 2.0 * M_PI * get_random_float(), phi = M_PI * get_random_float();
        Vector3f dir(std::cos(phi), std::sin(phi)*std::cos(theta), std::sin(phi)*std::sin(theta));
        pos.coords = center + radius * dir;
        pos.normal = dir;
        pos.emit = m->getEmission();
        pdf = 1.0f / area;
    }
    float getArea(){
        return area;
    }
    bool hasEmit(){
        return m->hasEmission();
    }

    std::string getName() const{
        return name;
    }
};




#endif //RAYTRACING_SPHERE_H

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

struct SAHBox {
    Bounds3 bounds;
    int primitiveCount;
    std::vector<std::shared_ptr<Object> > objs;

public:
    SAHBox() {
        bounds = Bounds3();
        primitiveCount = 0;
    }
    void joinPrimitive(std::shared_ptr<Object> obj) {
        bounds = Union(bounds, obj->getBounds());
        primitiveCount++;
        objs.push_back(obj);
    }
};

inline float computeSAH(std::shared_ptr<SAHBox> box1, std::shared_ptr<SAHBox> box2) {
    return box1->bounds.SurfaceArea()*box1->primitiveCount + box2->bounds.SurfaceArea()*box2->primitiveCount; 
}

class BVHAccel {

public:
    // BVHAccel Public Methods
    enum class SplitMethod { NAIVE, SAH };

    BVHAccel(std::vector<std::shared_ptr<Object>> p, SplitMethod splitMethod = SplitMethod::NAIVE);
    ~BVHAccel();
    Intersection Intersect(const Ray &ray) const;
    void Sample(Intersection &pos, float &pdf);

private:
    Intersection getIntersection(std::shared_ptr<BVHBuildNode> node, const Ray& ray)const;
    std::shared_ptr<BVHBuildNode> recursiveBuild(std::vector<std::shared_ptr<Object>>);
    void getSample(std::shared_ptr<BVHBuildNode> node, float p, Intersection &pos, float &pdf);
    void NAIVE(std::vector<std::shared_ptr<Object> > objects, std::shared_ptr<BVHBuildNode> node, int dim);
    void SAH(std::vector<std::shared_ptr<Object> > objects, Bounds3 centroidBounds, std::shared_ptr<BVHBuildNode> node, int dim);

    std::vector<std::shared_ptr<Object>> primitives;
    std::shared_ptr<BVHBuildNode> root;
    SplitMethod splitMethod;
};


#endif //RAYTRACING_BVH_H

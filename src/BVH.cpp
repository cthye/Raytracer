#include "BVH.hpp"

#include <algorithm>
#include <cassert>

BVHAccel::BVHAccel(std::vector<Object*> p, int maxPrimsInNode,
                   SplitMethod splitMethod)
    : maxPrimsInNode(std::min(255, maxPrimsInNode)),
      splitMethod(splitMethod),
      primitives(std::move(p)) {
    time_t start, stop;
    time(&start);
    if (primitives.empty()) return;

    root = recursiveBuild(primitives);

    time(&stop);
    double diff = difftime(stop, start);
    int hrs = (int)diff / 3600;
    int mins = ((int)diff / 60) - (hrs * 60);
    int secs = (int)diff - (hrs * 3600) - (mins * 60);

    printf(
        "\rBVH Generation complete: \nTime Taken: %i hrs, %i mins, %i secs\n\n",
        hrs, mins, secs);
}

BVHBuildNode* BVHAccel::recursiveBuild(std::vector<Object*> objects) {
    BVHBuildNode* node = new BVHBuildNode();

    // Compute bounds of all primitives in BVH node
    Bounds3 bounds;
    for (int i = 0; i < objects.size(); ++i)
        bounds = Union(bounds, objects[i]->getBounds());
    if (objects.size() == 1) {
        // Create leaf _BVHBuildNode_
        node->bounds = objects[0]->getBounds();
        node->object = objects[0];
        node->left = nullptr;
        node->right = nullptr;
        node->area = objects[0]->getArea();
        return node;
    } else if (objects.size() == 2) {
        node->left = recursiveBuild(std::vector{objects[0]});
        node->right = recursiveBuild(std::vector{objects[1]});

        node->bounds = Union(node->left->bounds, node->right->bounds);
        node->area = node->left->area + node->right->area;
        return node;
    } else {
        Bounds3 centroidBounds;
        for (int i = 0; i < objects.size(); ++i)
            centroidBounds =
                Union(centroidBounds, objects[i]->getBounds().Centroid());
        int dim = centroidBounds.maxExtent();
        switch (dim) {
            case 0:
                std::sort(objects.begin(), objects.end(), [](auto f1, auto f2) {
                    return f1->getBounds().Centroid().x <
                           f2->getBounds().Centroid().x;
                });
                break;
            case 1:
                std::sort(objects.begin(), objects.end(), [](auto f1, auto f2) {
                    return f1->getBounds().Centroid().y <
                           f2->getBounds().Centroid().y;
                });
                break;
            case 2:
                std::sort(objects.begin(), objects.end(), [](auto f1, auto f2) {
                    return f1->getBounds().Centroid().z <
                           f2->getBounds().Centroid().z;
                });
                break;
        }

        auto beginning = objects.begin();
        auto middling = objects.begin() + (objects.size() / 2);
        auto ending = objects.end();

        auto leftshapes = std::vector<Object*>(beginning, middling);
        auto rightshapes = std::vector<Object*>(middling, ending);

        assert(objects.size() == (leftshapes.size() + rightshapes.size()));

        node->left = recursiveBuild(leftshapes);
        node->right = recursiveBuild(rightshapes);

        node->bounds = Union(node->left->bounds, node->right->bounds);
        node->area = node->left->area + node->right->area;
    }

    return node;
}

Intersection BVHAccel::Intersect(const Ray& ray) const {
    Intersection isect;
    if (!root) return isect;
    isect = BVHAccel::getIntersection(root, ray);
    return isect;
}

Intersection BVHAccel::getIntersection(BVHBuildNode* node,
                                       const Ray& ray) const {
    // TODO Traverse the BVH to find intersection

    Intersection isect;
    std::array<int, 3> isDirNeg;
    isDirNeg.fill(0);
    if (ray.direction.x > 0) {
        isDirNeg[0] = 1;
    }
    if (ray.direction.y > 0) {
        isDirNeg[1] = 1;
    }
    if (ray.direction.z > 0) {
        isDirNeg[2] = 1;
    }

    if (!node->bounds.IntersectP(ray, ray.direction_inv, isDirNeg))
        return isect;

    if (node->left == nullptr && node->right == nullptr) {
        //* leaf node
        // std::cout << "leaf:" << node->object->getName() << std::endl;
        return node->object->getIntersection(ray);
    }

    Intersection l_isect = getIntersection(node->left, ray);
    Intersection r_isect = getIntersection(node->right, ray);
   
    //* intersect with 2 leaf, return the one with shorter distance
    if (l_isect.distance < r_isect.distance) {
        return l_isect;
    } else {
        return r_isect;
    }
}

//* 随机选一个mesh triangle, pdf = 1/S_triangle * leaf_node_area /
// root_node_area
//* node_area为这个节点内的所有三角形面积之和
void BVHAccel::getSample(BVHBuildNode* node, float p, Intersection& pos,
                         float& pdf) {
    if (node->left == nullptr || node->right == nullptr) {
        node->object->Sample(pos, pdf);
        pdf *= node->area;
        return;
    }
    if (p < node->left->area)
        getSample(node->left, p, pos, pdf);
    else
        getSample(node->right, p - node->left->area, pos, pdf);
}

void BVHAccel::Sample(Intersection& pos, float& pdf) {
    float p = std::sqrt(get_random_float()) * root->area;
    getSample(root, p, pos, pdf);
    pdf /= root->area;
}
#include "BVH.hpp"

#include <algorithm>
#include <cassert>
#include <vector>

BVHAccel::BVHAccel(std::vector<std::shared_ptr<Object>> p, SplitMethod _splitMethod)
    : primitives(std::move(p)), splitMethod(_splitMethod)
{
    if (primitives.empty())
        return;
    root = recursiveBuild(primitives);
}

std::shared_ptr<BVHBuildNode> BVHAccel::recursiveBuild(std::vector<std::shared_ptr<Object>> objects)
{
    auto node = std::make_shared<BVHBuildNode>();
    if (objects.size() == 1)
    {
        node->bounds = objects[0]->getBounds();
        node->object = objects[0];
        node->left = nullptr;
        node->right = nullptr;
        node->area = objects[0]->getArea();
        return node;
    }
    Bounds3 centroidBounds;
    for (int i = 0; i < objects.size(); ++i)
        centroidBounds =
            Union(centroidBounds, objects[i]->getBounds().Centroid());
    int dim = centroidBounds.maxExtent();
    if (splitMethod == BVHAccel::SplitMethod::NAIVE)
    {
        NAIVE(objects, node, dim);
    }
    else
    {
        SAH(objects, centroidBounds, node, dim);
    }

    return node;
}

Intersection BVHAccel::Intersect(const Ray &ray) const
{
    Intersection isect;
    if (!root)
        return isect;
    isect = BVHAccel::getIntersection(root, ray);
    return isect;
}

Intersection BVHAccel::getIntersection(std::shared_ptr<BVHBuildNode> node,
                                       const Ray &ray) const
{
    //* Traverse the BVH to find intersection
    Intersection isect;

    if (!node->bounds.IntersectP(ray))
        return isect;

    if (node->left == nullptr && node->right == nullptr)
    {
        //* leaf node
        // std::cout << "leaf:" << node->object->getName() << std::endl;
        return node->object->getIntersection(ray);
    }

    Intersection l_isect = getIntersection(node->left, ray);
    Intersection r_isect = getIntersection(node->right, ray);

    //* intersect with 2 leaf, return the one with shorter distance
    if (l_isect.distance < r_isect.distance)
    {
        return l_isect;
    }
    else
    {
        return r_isect;
    }
}

//* 随机选一个mesh triangle, pdf = 1/S_triangle * leaf_node_area /
// root_node_area
//* node_area为这个节点内的所有三角形面积之和
void BVHAccel::getSample(std::shared_ptr<BVHBuildNode> node, float p, Intersection &pos,
                         float &pdf)
{
    if (node->left == nullptr || node->right == nullptr)
    {
        node->object->Sample(pos, pdf);
        pdf *= node->area;
        return;
    }
    if (p < node->left->area)
        getSample(node->left, p, pos, pdf);
    else
        getSample(node->right, p - node->left->area, pos, pdf);
}

void BVHAccel::Sample(Intersection &pos, float &pdf)
{
    float p = std::sqrt(get_random_float()) * root->area;
    getSample(root, p, pos, pdf);
    pdf /= root->area;
}

void BVHAccel::NAIVE(std::vector<std::shared_ptr<Object>> objects, std::shared_ptr<BVHBuildNode> node, int dim)
{
    auto xComparator = [](std::shared_ptr<Object> f1, std::shared_ptr<Object> f2)
    { return f1->getBounds().Centroid().x < f2->getBounds().Centroid().x; };
    auto yComparator = [](std::shared_ptr<Object> f1, std::shared_ptr<Object> f2)
    { return f1->getBounds().Centroid().y < f2->getBounds().Centroid().y; };
    auto zComparator = [](std::shared_ptr<Object> f1, std::shared_ptr<Object> f2)
    { return f1->getBounds().Centroid().z < f2->getBounds().Centroid().z; };
    auto comparator = (dim == 0) ? xComparator : (dim == 1) ? yComparator
                                                            : zComparator;
    std::sort(objects.begin(), objects.end(), comparator);

    auto leftshapes = std::vector<std::shared_ptr<Object>>(objects.begin(), objects.begin() + (objects.size() / 2));
    auto rightshapes = std::vector<std::shared_ptr<Object>>(objects.begin() + (objects.size() / 2), objects.end());

    assert(objects.size() == (leftshapes.size() + rightshapes.size()));

    node->left = recursiveBuild(leftshapes);
    node->right = recursiveBuild(rightshapes);

    node->bounds = Union(node->left->bounds, node->right->bounds);
    node->area = node->left->area + node->right->area;
}

#define SplitCntPerAxis 13

void BVHAccel::SAH(std::vector<std::shared_ptr<Object>> objects, Bounds3 centroidBounds, std::shared_ptr<BVHBuildNode> node, int axis)
{
    std::vector<std::shared_ptr<SAHBox>> boxList;
    for (int i = 0; i < SplitCntPerAxis; i++)
    {
        std::shared_ptr<SAHBox> b = std::make_shared<SAHBox>();
        boxList.push_back(b);
    }
    for (int i = 0; i < objects.size(); ++i)
    {
        Vector3f centroid = objects[i]->getBounds().Centroid();
        int index = clamp(0, SplitCntPerAxis - 1, std::floor(centroidBounds.Offset(centroid)[axis] * SplitCntPerAxis - 1));
        boxList[index]->joinPrimitive(objects[i]);
    }

    float SAHCost = std::numeric_limits<float>::max();
    std::vector<std::shared_ptr<Object>> leftshapes;
    std::vector<std::shared_ptr<Object>> rightshapes;
    for (int i = 0; i < SplitCntPerAxis; i++)
    {
        std::shared_ptr<SAHBox> leftBox = std::make_shared<SAHBox>();
        for (int j = 0; j <= i; j++)
        {
            leftBox->bounds = Union(leftBox->bounds, boxList[j]->bounds);
            leftBox->primitiveCount += boxList[j]->primitiveCount;
            leftBox->objs.insert(leftBox->objs.end(), boxList[j]->objs.begin(), boxList[j]->objs.end());
        }

        std::shared_ptr<SAHBox> rightBox = std::make_shared<SAHBox>();
        for (int j = i + 1; j < SplitCntPerAxis; j++)
        {
            rightBox->bounds = Union(rightBox->bounds, boxList[j]->bounds);
            rightBox->primitiveCount += boxList[j]->primitiveCount;
            rightBox->objs.insert(rightBox->objs.end(), boxList[j]->objs.begin(), boxList[j]->objs.end());
        }
        float cost = computeSAH(leftBox, rightBox);
       
        if (cost < SAHCost)
        {
            SAHCost = cost;
            leftshapes = leftBox->objs;
            rightshapes = rightBox->objs;
        }
    }
    node->left = recursiveBuild(leftshapes);
    node->right = recursiveBuild(rightshapes);

    node->bounds = Union(node->left->bounds, node->right->bounds);
    node->area = node->left->area + node->right->area;
}
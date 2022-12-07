#pragma once

#include <string>

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
    Vector3f v0, v1, v2; // vertices A, B ,C , counter-clockwise order
    Vector3f e1, e2;     // 2 edges v1-v0, v2-v0;
    Vector3f t0, t1, t2; // texture coords
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
    Vector3f evalDiffuseColor(const Vector2f&) const override;
    Bounds3 getBounds() override;
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

class MeshTriangle : public Object
{
public:
    MeshTriangle(const std::string& filename, std::shared_ptr<Material> mt)
    {
        objl::Loader loader;
        loader.LoadFile(filename);
        area = 0;
        m = mt;
        assert(loader.LoadedMeshes.size() == 1);
        auto mesh = loader.LoadedMeshes[0];

        Vector3f min_vert = Vector3f{std::numeric_limits<float>::infinity(),
                                     std::numeric_limits<float>::infinity(),
                                     std::numeric_limits<float>::infinity()};
        Vector3f max_vert = Vector3f{-std::numeric_limits<float>::infinity(),
                                     -std::numeric_limits<float>::infinity(),
                                     -std::numeric_limits<float>::infinity()};
        for (int i = 0; i < mesh.Vertices.size(); i += 3) {
            std::array<Vector3f, 3> face_vertices;

            for (int j = 0; j < 3; j++) {
                auto vert = Vector3f(mesh.Vertices[i + j].Position.X,
                                     mesh.Vertices[i + j].Position.Y,
                                     mesh.Vertices[i + j].Position.Z);
                face_vertices[j] = vert;

                min_vert = Vector3f(std::min(min_vert.x, vert.x),
                                    std::min(min_vert.y, vert.y),
                                    std::min(min_vert.z, vert.z));
                max_vert = Vector3f(std::max(max_vert.x, vert.x),
                                    std::max(max_vert.y, vert.y),
                                    std::max(max_vert.z, vert.z));
            }

            triangles.emplace_back(std::make_shared<Triangle>(face_vertices[0], face_vertices[1],
                                   face_vertices[2], mt));
        }

        bounding_box = Bounds3(min_vert, max_vert);

        std::vector<std::shared_ptr<Object>> ptrs;
        for (auto& tri : triangles){
            ptrs.push_back(tri);
            area += tri->area;
        }
        bvh = new BVHAccel(ptrs);
    }

    MeshTriangle(const std::string& filename, std::string _name, std::shared_ptr<Material> mt)
    {
        objl::Loader loader;
        loader.LoadFile(filename);
        area = 0;
        m = mt;
        name = _name;
        assert(loader.LoadedMeshes.size() == 1);
        auto mesh = loader.LoadedMeshes[0];

        Vector3f min_vert = Vector3f{std::numeric_limits<float>::infinity(),
                                     std::numeric_limits<float>::infinity(),
                                     std::numeric_limits<float>::infinity()};
        Vector3f max_vert = Vector3f{-std::numeric_limits<float>::infinity(),
                                     -std::numeric_limits<float>::infinity(),
                                     -std::numeric_limits<float>::infinity()};
        for (int i = 0; i < mesh.Vertices.size(); i += 3) {
            std::array<Vector3f, 3> face_vertices;

            for (int j = 0; j < 3; j++) {
                auto vert = Vector3f(mesh.Vertices[i + j].Position.X,
                                     mesh.Vertices[i + j].Position.Y,
                                     mesh.Vertices[i + j].Position.Z);
                face_vertices[j] = vert;

                min_vert = Vector3f(std::min(min_vert.x, vert.x),
                                    std::min(min_vert.y, vert.y),
                                    std::min(min_vert.z, vert.z));
                max_vert = Vector3f(std::max(max_vert.x, vert.x),
                                    std::max(max_vert.y, vert.y),
                                    std::max(max_vert.z, vert.z));
            }

            triangles.emplace_back(std::make_shared<Triangle>(face_vertices[0], face_vertices[1],
                                   face_vertices[2], _name, mt));
        }

        bounding_box = Bounds3(min_vert, max_vert);

        std::vector<std::shared_ptr<Object>> ptrs;
        for (auto& tri : triangles){
            ptrs.push_back(tri);
            area += tri->area;
        }
        bvh = new BVHAccel(ptrs);
    }

    MeshTriangle(const std::string& filename, Vector3f tran, Vector3f scale, Vector3f xRotate, Vector3f yRotate, Vector3f zRotate, std::shared_ptr<Material> mt)
    {
        objl::Loader loader;
        loader.LoadFile(filename);
        area = 0;
        m = mt;
        assert(loader.LoadedMeshes.size() == 1);
        auto mesh = loader.LoadedMeshes[0];

        Vector3f min_vert = Vector3f{std::numeric_limits<float>::infinity(),
                                     std::numeric_limits<float>::infinity(),
                                     std::numeric_limits<float>::infinity()};
        Vector3f max_vert = Vector3f{-std::numeric_limits<float>::infinity(),
                                     -std::numeric_limits<float>::infinity(),
                                     -std::numeric_limits<float>::infinity()};
        for (int i = 0; i < mesh.Vertices.size(); i += 3) {
            std::array<Vector3f, 3> face_vertices;

            for (int j = 0; j < 3; j++) {
                auto vert = Vector3f(mesh.Vertices[i + j].Position.X,
                                     mesh.Vertices[i + j].Position.Y,
                                     mesh.Vertices[i + j].Position.Z);
                
                vert.x = dotProduct(vert, xRotate);
                vert.y = dotProduct(vert, yRotate);
                vert.z = dotProduct(vert, zRotate);

                vert = scale * vert + tran;
                face_vertices[j] = vert;
                
                min_vert = Vector3f(std::min(min_vert.x, vert.x),
                                    std::min(min_vert.y, vert.y),
                                    std::min(min_vert.z, vert.z));
                max_vert = Vector3f(std::max(max_vert.x, vert.x),
                                    std::max(max_vert.y, vert.y),
                                    std::max(max_vert.z, vert.z));
            }

            triangles.emplace_back(std::make_shared<Triangle>(face_vertices[0], face_vertices[1],
                                   face_vertices[2], mt));
        }

        bounding_box = Bounds3(min_vert, max_vert);

        std::vector<std::shared_ptr<Object>> ptrs;
        for (auto& tri : triangles){
            ptrs.push_back(tri);
            area += tri->area;
        }
        bvh = new BVHAccel(ptrs);
    }

    Bounds3 getBounds() { return bounding_box; }

    Vector3f evalDiffuseColor(const Vector2f& st) const
    {
        float scale = 5;
        float pattern =
            (fmodf(st.x * scale, 1) > 0.5) ^ (fmodf(st.y * scale, 1) > 0.5);
        return lerp(Vector3f(0.815, 0.235, 0.031),
                    Vector3f(0.937, 0.937, 0.231), pattern);
    }

    Intersection getIntersection(Ray ray)
    {
        Intersection intersec;

        if (bvh) {
            intersec = bvh->Intersect(ray);
        }

        return intersec;
    }
    
    void Sample(Intersection &pos, float &pdf){
        bvh->Sample(pos, pdf);
        pos.emit = m->getEmission();
        pos.m = m;
    }
    float getArea(){
        return area;
    }
    bool hasEmit(){
        return m->hasEmission();
    }

    std::string getName() const {
        return name;
    }

    Bounds3 bounding_box;
    std::shared_ptr<Vector3f[]> vertices;
    uint32_t numTriangles;
    std::shared_ptr<uint32_t[]> vertexIndex;
    std::shared_ptr<Vector2f[]> stCoordinates;

    std::vector<std::shared_ptr<Triangle>> triangles;

    BVHAccel* bvh;
    float area;

    std::shared_ptr<Material> m;

    std::string name; // used for debug
};

inline Bounds3 Triangle::getBounds() { return Union(Bounds3(v0, v1), v2); }

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

inline Vector3f Triangle::evalDiffuseColor(const Vector2f&) const
{
    return Vector3f(0.5, 0.5, 0.5);
}

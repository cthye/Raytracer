#ifndef RAYTRACING_MESHTRIANGLE_H
#define RAYTRACING_MESHTRIANGLE_H

#include "Triangle.hpp"
#include "BVH.hpp"
#include "Intersection.hpp"
#include "Material.hpp"
#include "OBJ_Loader.hpp"
#include "Object.hpp"

class MeshTriangle : public Object
{
public:
    Bounds3 bounding_box;
    std::shared_ptr<Vector3f[]> vertices;
    std::vector<std::shared_ptr<Triangle>> triangles;
    BVHAccel* bvh;
    float area;
    std::shared_ptr<Material> m;
    std::string name; // used for debug
    
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
};

# endif
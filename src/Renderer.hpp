//
// Created by goksu on 2/25/20.
//
#include "Scene.hpp"
#include "Camera.hpp"

#pragma once
struct hit_payload
{
    float tNear;
    uint32_t index;
    Vector2f uv;
    Object* hit_obj;
};

class Renderer
{
public:
    std::vector<Vector3f>* framebuffer;
    int spp = 16;
    // int spp = 512;
    int THREAD_NUMBER = 16;
    int renderedPixels = 0;

    void Render(const Scene& scene, const Camera& cam);
    void MultiThreadRender(int tid, const Scene& scene, const Camera& cam);
private:
};

//
// Created by goksu on 2/25/20.
//
#include "Scene.hpp"

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
    Vector3f eye_pos = Vector3f(278., 273., -800.);
    std::vector<Vector3f>* framebuffer;
    // int spp = 16;
    int spp = 512;
    int THREAD_NUMBER = 16;
    int renderedPixels = 0;

    void Render(const Scene& scene);
    void MultiThreadRender(int tid, const Scene& scene);
private:
};

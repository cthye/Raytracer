#ifndef RAYTRACING_RENDERER_H
#define RAYTRACING_RENDERER_H

#include "Scene.hpp"
#include "Camera.hpp"
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

#endif //RAYTRACING_RENDERER_H
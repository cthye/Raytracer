#include "Renderer.hpp"

#include <fstream>
#include <mutex>
#include <thread>
#include <omp.h>
#include "Scene.hpp"

const float EPSILON = 0.00001;
std::mutex lock;

void Renderer::Render(const Scene& scene, const Camera& cam) {
    framebuffer = new std::vector<Vector3f>(scene.width * scene.height);
    std::cout << "SPP: " << spp << "\n";

    //* threads
    std::thread t[THREAD_NUMBER];
    // for (int i = 0; i < THREAD_NUMBER; i++) {
    //     t[i] =
    //         std::thread(&Renderer::MultiThreadRender, this, i, std::ref(scene));
    // }
    // for (int i = 0; i < THREAD_NUMBER; i++) {
    //     t[i].join();
    // }
    #pragma omp parallel for
        for (int i = 0; i < THREAD_NUMBER; i++) {
            Renderer::MultiThreadRender(i, std::ref(scene), std::ref(cam));
        }
    UpdateProgress(1.f);

    // save framebuffer to file
    FILE* fp = fopen("binary.ppm", "wb");
    (void)fprintf(fp, "P6\n%d %d\n255\n", scene.width, scene.height);
    for (auto i = 0; i < scene.height * scene.width; ++i) {
        static unsigned char color[3];
        //* gamma correction and convert HDR to TDR
        color[0] =
            (unsigned char)(255 *
                            std::pow(clamp(0, 1, (*framebuffer)[i].x), 0.6f));
        color[1] =
            (unsigned char)(255 *
                            std::pow(clamp(0, 1, (*framebuffer)[i].y), 0.6f));
        color[2] =
            (unsigned char)(255 *
                            std::pow(clamp(0, 1, (*framebuffer)[i].z), 0.6f));
        fwrite(color, 1, 3, fp);
    }
    fclose(fp);
}

void Renderer::MultiThreadRender(int tid, const Scene& scene, const Camera& cam) {
    for (uint32_t j = 0; j < scene.height; ++j) {
        for (uint32_t i = 0; i < scene.width; ++i) {
            int m = j * scene.height + i;
            if (m % THREAD_NUMBER == tid) {
                // generate primary ray direction
                // std::cout << "thread" << tid << "is at the " << m << std::endl;  
                // std::cout << "thread" << tid << "'s dir" << dir  << std::endl;
                //MSAA抗锯齿
                //todo: get random float可能会有点慢...用spp % width可能好一点（前提是width=height，spp可开根）
                for (int k = 0; k < spp; k++) {
                    float x = 2 * (i + get_random_float()) / (float)scene.width - 1;
                    float y = 1 - 2 * (j + get_random_float()) / (float)scene.height;

                    //为什么不用减去eye_pos，因为以eye_pos作为原点啊
                    Vector3f dir  = normalize(x * cam.horizontal + y * cam.vertical - cam.w);
                    // std::cout << dir << std::endl;
                    (*framebuffer)[m] +=
                        scene.castRay(Ray(cam.eye_pos, dir));
                }
                (*framebuffer)[m] = (*framebuffer)[m] / spp;
                lock.lock();
                UpdateProgress(++renderedPixels /
                               ((float)scene.height * scene.width));
                lock.unlock();
            }
        }
    }
}
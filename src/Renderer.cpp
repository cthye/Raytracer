//
// Created by goksu on 2/25/20.
//

#include "Renderer.hpp"

#include <fstream>
#include <mutex>
#include <thread>
#include <omp.h>
#include "Scene.hpp"

const float EPSILON = 0.00001;
std::mutex lock;

// The main render function. This where we iterate over all pixels in the image,
// generate primary rays and cast these rays into the scene. The content of the
// framebuffer is saved to a file.
void Renderer::Render(const Scene& scene) {
    framebuffer = new std::vector<Vector3f>(scene.width * scene.height);
    // std::vector<Vector3f> framebuffer(scene.width * scene.height);

    // float scale = tan(deg2rad(scene.fov * 0.5));
    // float imageAspectRatio = scene.width / (float)scene.height;
    // Vector3f eye_pos(278, 273, -800);

    // change the spp (samples per pixel) value to change sample amount
    // int spp = 16;
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
            Renderer::MultiThreadRender(i, std::ref(scene));
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

void Renderer::MultiThreadRender(int tid, const Scene& scene) {
    for (uint32_t j = 0; j < scene.height; ++j) {
        for (uint32_t i = 0; i < scene.width; ++i) {
            int m = j * scene.height + i;
            if (m % THREAD_NUMBER == tid) {
                // generate primary ray direction
                // std::cout << "thread" << tid << "is at the " << m << std::endl;
               
                // std::cout << "thread" << tid << "'s dir" << dir  << std::endl;
                int width, height;
                float step = 1. / width;
                // width = height = sqrt(spp);
                //MSAA抗锯齿
                for (int k = 0; k < spp; k++) {
                    // float x = (2 * (i + step / 2 + step * (k % width)) / (float)scene.width - 1) *
                    //       scene.imageAspectRatio * scene.scale;
                    // float y = (1 - 2 * (j + step / 2 + step * (k / height)) / (float)scene.height) * scene.scale;
                    float x = (2 * (i + get_random_float()) / (float)scene.width - 1) *
                          scene.imageAspectRatio * scene.scale;
                    float y = (1 - 2 * (j + get_random_float()) / (float)scene.height) * scene.scale;
                    //? 为什么不用减去eye_pos?
                    Vector3f dir = normalize(Vector3f(-x, y, 1));

                    // float x = 2 * (i + 0.5 + get_random_float()) / (float)scene.width - 1;
                    // float y = 1 - 2 * (j + 0.5 + get_random_float()) / (float)scene.height;
                    // Vector3f dir  = normalize(scene.lower_left_corner + x*scene.horizontal + y*scene.vertical - scene.eye_pos);
                    // std::cout << dir << std::endl;
                    (*framebuffer)[m] +=
                        scene.castRay(Ray(scene.eye_pos, dir), 0);
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
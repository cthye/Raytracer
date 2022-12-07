#ifndef RAYTRACING_CAMERA_H
#define RAYTRACING_CAMERA_H

#include "Vector.hpp"
#include "global.hpp"

class Camera {
public:
    Camera(float a, float f, Vector3f e, Vector3f t, Vector3f v) : imageAspectRatio(a), fov(f), eye_pos(e), target(t), vup(v) {
        scale = tan(deg2rad(fov * 0.5));
        viewportHeight = scale * 2.f; // y [-1, 1], z = 1
        viewportWidth = imageAspectRatio * viewportHeight;
        w = normalize(eye_pos - target);
        u = normalize(crossProduct(vup, w));
        v = crossProduct(w, u);
        horizontal = viewportWidth * u;
        vertical = viewportHeight * v;
        scale = tan(deg2rad(fov * 0.5));
        std::cout << "horizontal " << horizontal << std::endl;
        std::cout << "vertical " << vertical << std::endl;
        std::cout << "w " << this->w << std::endl;
        std::cout << "u " << u << std::endl;
        std::cout << "v " << v << std::endl;
    }

    Vector3f target; 
    Vector3f eye_pos;
    Vector3f vup; 
    double fov;
    float scale;
    float imageAspectRatio;
    float viewportHeight;
    float viewportWidth;
    Vector3f w;
    Vector3f u;
    Vector3f v;
    Vector3f horizontal;
    Vector3f vertical;
};

#endif 

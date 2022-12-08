#ifndef RAYTRACING_CAMERA_H
#define RAYTRACING_CAMERA_H

#include "vector.hpp"
#include "utils.hpp"

class Camera {
public:
    Vector3f eye;
    Vector3f target;
    Vector3f u, v, w;
    Vector3f horizontal;
    Vector3f vertical;
    Vector3f lowerLeftCorner;


    __host__ __device__ Camera() {}
    __host__ __device__ Camera(Vector3f lookfrom, Vector3f lookat, Vector3f vup, float vfov, float aspect) {
        eye = lookfrom;
        target = lookat;
        w = normalize(eye - target);
        u = normalize(cross(vup, w));
        v = cross(w, u);

        float viewPortHeight = 2.0f * tan(deg2rad(vfov / 2.0f));
        float viewPortWidth = aspect * viewPortHeight;
        horizontal = viewPortWidth * u;
        vertical = viewPortHeight * v;
        lowerLeftCorner = eye - horizontal/2.0f - vertical/2.0f - w;
    }

    __host__ __device__ Ray getRay(double s, double t) const {
        return Ray(eye, lowerLeftCorner + s*horizontal + t*vertical - eye);
    }
};
#endif //RAYTRACING_CAMERA_H
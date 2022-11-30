#pragma once
#ifndef RAYTRACING_MATERIAL_H
#define RAYTRACING_MATERIAL_H

#include "vector.hpp"
#include "utils.hpp"
#include "ray.hpp"
#include "intersection.hpp"

class Intersection;


enum MaterialType { DIFFUSE, METAL, DIELECTRIC};

__host__ __device__ Vector3f reflect(const Vector3f& I, const Vector3f& N) {
    return I - 2.0f * dot(I, N) * N;
}

__host__ __device__ float schlick(float cosine, float ref_idx) {
    float r0 = (1.0f-ref_idx) / (1.0f+ref_idx);
    r0 = r0*r0;
    return r0 + (1.0f-r0)*pow((1.0f - cosine),5.0f);
}

__host__ __device__ bool refract(const Vector3f& v, const Vector3f& n, float ni_over_nt, Vector3f& refracted) {
    Vector3f uv = normalize(v);
    float dt = dot(uv, n);
    float discriminant = 1.0f - ni_over_nt*ni_over_nt*(1-dt*dt);
    if (discriminant > 0) {
        refracted = ni_over_nt*(uv - n*dt) - n*sqrt(discriminant);
        return true;
    }
    else
        return false;
}

// __host__ __device Vector3f refract(const Vector3f &I, const Vector3f &N,
//                      const float &ior) const {
//     float cosi = clamp(-1, 1, dot(I, N));
//     float etai = 1, etat = ior;
//     Vector3f n = N;
//     if (cosi < 0) {
//         cosi = -cosi;
//     } else {
//         std::swap(etai, etat);
//         n = -N;
//     }
//     float eta = etai / etat;
//     float k = 1 - eta * eta * (1 - cosi * cosi);
//     return k < 0 ? 0 : eta * I + (eta * cosi - sqrtf(k)) * n;
// }

//* sample in sphere
__device__ Vector3f random_in_unit_sphere(curandState *local_rand_state) {
    Vector3f p;
    do{
        p = 2.0 * Vector3f(curand_uniform(local_rand_state), 
                       curand_uniform(local_rand_state), 
                       curand_uniform(local_rand_state)) - Vector3f(1, 1, 1);
    } while(p.squareLength() >= 1.0f);
    return p;
}

__device__ Vector3f random_unit_vector(curandState *local_rand_state) {
    return normalize(random_in_unit_sphere(local_rand_state));
}

class Material {
public:
    __host__ __device__ Material() {}
    __host__ __device__ Material(MaterialType t) : mType(t) {}
    __device__ virtual bool scatter(const Ray& r_in, 
                                    const Intersection& intersec, 
                                    Vector3f& attenuation, 
                                    Ray& scattered,
                                    curandState* local_rand_state) const = 0;
    // __host__ __device__ virtual vec3 emitted(float u, 
    //                                 float v, 
    //                                 const vec3& p) const { 
    //     return vec3(0, 0, 0); 
    // }
    MaterialType mType;
};

class Diffuse : public Material {
public:
    __host__ __device__ Diffuse(const Vector3f& a) : albedo(a), Material(DIFFUSE) {}
    __device__ bool scatter(const Ray& r_in, 
                                    const Intersection& intersec, 
                                    Vector3f& attenuation, 
                                    Ray& scattered,
                                    curandState* local_rand_state) const {
                                        Vector3f scatter_direction = intersec.normal + random_unit_vector(local_rand_state);
                                        if (scatter_direction.nearZero())
                                            scatter_direction = intersec.normal;
                                    
                                        scattered = Ray(intersec.coords, scatter_direction);
                                
                                        attenuation = albedo;
                                        return true;
                                    }
    Vector3f albedo;
};

class Metal : public Material {
public:
    __host__ __device__ Metal(const Vector3f& a, float f) : albedo(a), fuzz(f), Material(METAL) {}
    __device__ bool scatter(const Ray& r_in, 
                                    const Intersection& intersec, 
                                    Vector3f& attenuation, 
                                    Ray& scattered,
                                    curandState* local_rand_state) const {
                    Vector3f reflected = reflect(normalize(r_in.direction), intersec.normal);
                    scattered = Ray(intersec.coords, reflected + fuzz*random_unit_vector(local_rand_state));
                    attenuation = albedo;
                    return (dot(scattered.direction, intersec.normal) > 0.0f);
                                    }
    Vector3f albedo;
    float fuzz;
};

//todo: come back and refract this part..
class Dielectric : public Material {
public:
    __host__ __device__ Dielectric(float _ir) : ir(_ir), Material(DIELECTRIC) {}
    __device__ bool scatter(const Ray& r_in, 
                                    const Intersection& intersec, 
                                    Vector3f& attenuation, 
                                    Ray& scattered,
                                    curandState* local_rand_state) const {
        Vector3f outward_normal;
        Vector3f reflected = reflect(r_in.direction, intersec.normal);
        float ni_over_nt;
        attenuation = Vector3f(1.0, 1.0, 1.0);
        Vector3f refracted;
        float reflect_prob;
        float cosine;
        if (dot(r_in.direction, intersec.normal) > 0.0f) {
            outward_normal = -intersec.normal;
            ni_over_nt = ir;
            cosine = dot(r_in.direction, intersec.normal) / r_in.direction.length();
            cosine = sqrt(1.0f - ir*ir*(1-cosine*cosine));
        }
        else {
            outward_normal = intersec.normal;
            ni_over_nt = 1.0f / ir;
            cosine = -dot(r_in.direction, intersec.normal) / r_in.direction.length();
        }
        if (refract(r_in.direction, outward_normal, ni_over_nt, refracted))
            reflect_prob = schlick(cosine, ir);
        else
            reflect_prob = 1.0f;
        if (curand_uniform(local_rand_state) < reflect_prob)
            scattered = Ray(intersec.coords, reflected);
        else
            scattered = Ray(intersec.coords, refracted);
        return true;
    }
    float ir;
};


#endif //RAYTRACING_MATERIAL_H

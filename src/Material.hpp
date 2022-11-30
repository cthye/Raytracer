//
// Created by LEI XU on 5/16/19.
//

#ifndef RAYTRACING_MATERIAL_H
#define RAYTRACING_MATERIAL_H

#include "Vector.hpp"

enum MaterialType { DIFFUSE, MICROFACET };

class Material {
   private:
    // Compute reflection direction
    Vector3f reflect(const Vector3f &I, const Vector3f &N) const {
        return I - 2 * dotProduct(I, N) * N;
    }

    // Compute refraction direction using Snell's law
    //
    // We need to handle with care the two possible situations:
    //
    //    - When the ray is inside the object
    //
    //    - When the ray is outside.
    //
    // If the ray is outside, you need to make cosi positive cosi = -N.I
    //
    // If the ray is inside, you need to invert the refractive indices and
    // negate the normal N
    Vector3f refract(const Vector3f &I, const Vector3f &N,
                     const float &ior) const {
        float cosi = clamp(-1, 1, dotProduct(I, N));
        float etai = 1, etat = ior;
        Vector3f n = N;
        if (cosi < 0) {
            cosi = -cosi;
        } else {
            std::swap(etai, etat);
            n = -N;
        }
        float eta = etai / etat;
        float k = 1 - eta * eta * (1 - cosi * cosi);
        return k < 0 ? 0 : eta * I + (eta * cosi - sqrtf(k)) * n;
    }

    // Compute Fresnel equation
    //
    // \param I is the incident view direction
    //
    // \param N is the normal at the intersection point
    //
    // \param ior is the material refractive index
    //
    // \param[out] kr is the amount of light reflected
    void fresnel(const Vector3f &I, const Vector3f &N, const float &ior,
                 float &kr) const {
        float cosi = clamp(-1, 1, dotProduct(I, N));
        float etai = 1, etat = ior;
        if (cosi > 0) {
            std::swap(etai, etat);
        }
        // Compute sini using Snell's law
        float sint = etai / etat * sqrtf(std::max(0.f, 1 - cosi * cosi));
        // Total internal reflection
        if (sint >= 1) {
            kr = 1;
        } else {
            float cost = sqrtf(std::max(0.f, 1 - sint * sint));
            cosi = fabsf(cosi);
            float Rs = ((etat * cosi) - (etai * cost)) /
                       ((etat * cosi) + (etai * cost));
            float Rp = ((etai * cosi) - (etat * cost)) /
                       ((etai * cosi) + (etat * cost));
            kr = (Rs * Rs + Rp * Rp) / 2;
        }
        // As a consequence of the conservation of energy, transmittance is
        // given by: kt = 1 - kr;
    }

    // convert local to world..
    Vector3f toWorld(const Vector3f &a, const Vector3f &N) {
        Vector3f B, C;
        if (std::fabs(N.x) > std::fabs(N.y)) {
            float invLen = 1.0f / std::sqrt(N.x * N.x + N.z * N.z);
            C = Vector3f(N.z * invLen, 0.0f, -N.x * invLen);
        } else {
            float invLen = 1.0f / std::sqrt(N.y * N.y + N.z * N.z);
            C = Vector3f(0.0f, N.z * invLen, -N.y * invLen);
        }
        B = crossProduct(C, N);
        return a.x * B + a.y * C + a.z * N;
    }

   public:
    MaterialType m_type;
    // Vector3f m_color;
    Vector3f m_emission;
    float ior;
    Vector3f Kd, Ks;
    float specularExponent;
    float alpha =
        0.07;  // coefficient of normal distribution, distribution[0.05, 0.5],
               // smaller -> glossy, larger -> diffuse
    // Texture tex;

    inline Material(MaterialType t = DIFFUSE, Vector3f e = Vector3f(0, 0, 0));
    inline MaterialType getType();
    // inline Vector3f getColor();
    inline Vector3f getColorAt(double u, double v);
    inline Vector3f getEmission();
    inline bool hasEmission();

    // sample a ray by Material properties
    inline Vector3f sample(const Vector3f &wi, const Vector3f &N);
    inline Vector3f importance_sample(const Vector3f &wi, const Vector3f &N);

    // given a ray, calculate the PdF of this ray
    inline float pdf(const Vector3f &wi, const Vector3f &wo, const Vector3f &N);
    // given a ray, calculate the contribution of this ray
    inline Vector3f eval(const Vector3f &wi, const Vector3f &wo,
                         const Vector3f &N);
    inline void normal_distribution(const Vector3f &N, const Vector3f &h,
                                    float &res);
    inline void geometry_term(const Vector3f &N, const Vector3f &h,
                              const Vector3f &wi, const Vector3f &wo,
                              float &res);
};

Material::Material(MaterialType t, Vector3f e) {
    m_type = t;
    // m_color = c;
    m_emission = e;
}

MaterialType Material::getType() { return m_type; }
/// Vector3f Material::getColor(){return m_color;}
Vector3f Material::getEmission() { return m_emission; }
bool Material::hasEmission() {
    if (m_emission.norm() > EPSILON)
        return true;
    else
        return false;
}

Vector3f Material::getColorAt(double u, double v) { return Vector3f(); }

Vector3f Material::sample(const Vector3f &wi, const Vector3f &N) {
    switch (m_type) {
        case DIFFUSE: {
            // uniform sample on the hemisphere
            //* 但这种采样方法其实不太均匀，不能很好地将半圆坐标转化成直角坐标
            float x_1 = get_random_float(), x_2 = get_random_float();
            float z = std::fabs(1.0f - 2.0f * x_1);
            float r = std::sqrt(1.0f - z * z), phi = 2 * M_PI * x_2;
            Vector3f localRay(r * std::cos(phi), r * std::sin(phi), z);
            return toWorld(localRay, N);

            break;
        }
        // case MICROFACET: {
        //     float x_1 = get_random_float(), x_2 = get_random_float();
        //     float tmp = -alpha * alpha * std::logl(1 - x_1);
        //     if (tmp < 0)
        //         return 0.f;
        //     else {
        //         float theta_h = std::atan(std::sqrt(tmp));
        //         float phi_h = 2 * M_PI * x_2;

        //         Vector3f h = Vector3f(theta_h, phi_h);
        //         return toWorld(reflect(wi, h), N);
        //     }
        // }
    }
}

Vector3f Material::importance_sample(const Vector3f &wi, const Vector3f &N) {
    switch (m_type) {
        case DIFFUSE: {
            // consine importance sample on the hemisphere
            // reference: https://zhuanlan.zhihu.com/p/360420413
            float x_1 = get_random_float(), x_2 = get_random_float();

            float theta = std::acos(1 - x_1);
            float phi = 2 * M_PI * x_2;
            Vector3f localRay(std::sin(theta) * std::cos(phi),
                              std::sin(theta) * std::sin(phi), std::cos(theta));
            return toWorld(localRay, N);

            break;
        }
    }
}

float Material::pdf(const Vector3f &wi, const Vector3f &wo, const Vector3f &N) {
    switch (m_type) {
        case DIFFUSE: {

            // uniform sample probability 1 / (2 * PI)
            if (dotProduct(wo, N) > 0.0f)
                return 0.5f / M_PI;
            else
                return 0.0f;
            break;
        }
        // case MICROFACET: {
        //     if (dotProduct(wo, N) <= 0.0f) {
        //         return 0.f;
        //     } else {
        //         Vector3f half = (wi + wo).normalized();
        //         float theta_h = std::acos(half.y);
        //         float phi_h = std::atan(half.z / half.x);
        //         float phi_pdf = 1 / (2 * M_PI);
        //         float theta_pdf =
        //             2 * std::sin(theta_h) /
        //             (alpha * alpha * std::pow(std::cos(theta_h), 3)) *
        //             std::exp(-std::pow(std::tan(theta_h), 2) / (alpha * alpha));
        //         float h_pdf = theta_pdf * theta_h / std::cos(theta_h);
        //         float w_pdf = h_pdf / (4 * dotProduct(wo, N));
        //         return w_pdf;
        //     }
        //     break;
        // } 
    }
}

// BRDF
Vector3f Material::eval(const Vector3f &wi, const Vector3f &wo,
                        const Vector3f &N) {
    switch (m_type) {
        case DIFFUSE: {
            // calculate the contribution of diffuse model
            float cosalpha = dotProduct(N, wo);
            if (cosalpha > 0.0f) {
                Vector3f diffuse = Kd / M_PI;  // albedo / pi
                return diffuse;
            } else
                return Vector3f(0.0f);
            break;
        }
        case MICROFACET: {
            //* reference:
            //* https://www.pbr-book.org/3ed-2018/Reflection_Models/Microfacet_Models
            //* https://cs184.eecs.berkeley.edu/sp18/article/26
            float cosalpha = dotProduct(N, wo);
            float NdotI = dotProduct(N, wi);
            float NdotO = dotProduct(N, wo);
            if (cosalpha <= 0.0f || NdotI <= 0 || NdotO <= 0) {
                return Vector3f(0.f);
            } else {
                Vector3f half = (wi + wo).normalized();
                float f, d, g;
                fresnel(wi, N, ior, f);
                normal_distribution(N, half, d);
                geometry_term(N, half, wi, wo, g);
                return Kd * g * f * d / (4 * dotProduct(N, wi) * dotProduct(N, wo));
            }
            break;
        }
    }
}

void Material::normal_distribution(const Vector3f &N, const Vector3f &h,
                                   float &res) {
    float theta = std::acos(dotProduct(N, h));
    res = std::exp(-std::tan(theta) * std::tan(theta) / alpha / alpha) /
          (M_PI * alpha * alpha * std::pow(std::cos(theta), 4));
    return;
}

void Material::geometry_term(const Vector3f &N, const Vector3f &h,
                             const Vector3f &wi, const Vector3f &wo,
                             float &res) {
    //* reference:
    //http://simonstechblog.blogspot.com/2011/12/microfacet-brdf.html
    //* implicit
    res = dotProduct(N, wi) * dotProduct(N, wo);
    return;
}

#endif  // RAYTRACING_MATERIAL_H

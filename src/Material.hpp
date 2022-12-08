#ifndef RAYTRACING_MATERIAL_H
#define RAYTRACING_MATERIAL_H

#include "Vector.hpp"
// #include "global.hpp"

enum MaterialType { DIFFUSE, MICROFACET };
enum SampleType { UNIFORMAL, IMPORTANCE };


class Material {
public:
    MaterialType m_type;
    Vector3f m_emission;
    float ior;
    Vector3f Kd, Ks;

    // coefficient of normal distribution, distribution[0.05, 0.5],
    // smaller -> glossy, larger -> diffuse
    float alpha = 0.07;

    inline Material(MaterialType t = DIFFUSE, Vector3f e = Vector3f(0, 0, 0));
    inline Material(MaterialType t, Vector3f kd, Vector3f e);
    inline Vector3f getEmission();
    inline bool hasEmission();

    /* ==================================== 
    functions relative to scatter a ray
    ====================================*/
    inline Vector3f scatter(const Vector3f &N, SampleType t = UNIFORMAL);
    inline Vector3f cosineWeightedHemiSample(const Vector3f &N);
    inline Vector3f uniformHemiSample(const Vector3f &N);
    inline Vector3f toWorld(const Vector3f &a, const Vector3f &N);

    /* ==================================== 
    functions relative to compute pdf
    ====================================*/
    inline float pdf(const Vector3f &wi, const Vector3f &wo, const Vector3f &N, SampleType t = UNIFORMAL);
    inline float cosineWeightedHemiPdf(const Vector3f &wo, const Vector3f &N);
    inline float uniformHemiPdf(const Vector3f &wo, const Vector3f &N);

    /* ==================================== 
    functions relative to brdf
    ====================================*/
    inline Vector3f eval(const Vector3f &wi, const Vector3f &wo,
                         const Vector3f &N);

    /* ==================================== 
    functions relative to microfacet model
    ====================================*/
    inline void normal_distribution(const Vector3f &N, const Vector3f &h,
                                    float &res);
    inline void geometry_term(const Vector3f &N, const Vector3f &h,
                              const Vector3f &wi, const Vector3f &wo,
                              float &res);

    inline Vector3f reflect(const Vector3f &I, const Vector3f &N);
    inline Vector3f refract(const Vector3f &I, const Vector3f &N, const float &ior);
    inline void fresnel(const Vector3f &I, const Vector3f &N, const float &ior, float &kr);
    
};

Material::Material(MaterialType t, Vector3f e) {
    m_type = t;
    m_emission = e;
}

Material::Material(MaterialType t, Vector3f _kd, Vector3f e) {
    m_type = t;
    Kd = _kd;
    m_emission = e;
}


/// Vector3f Material::getColor(){return m_color;}
Vector3f Material::getEmission() { return m_emission; }
bool Material::hasEmission() {
    if (m_emission.norm() > EPSILON)
        return true;
    else
        return false;
}

Vector3f Material::cosineWeightedHemiSample(const Vector3f &N) {
    float s = get_random_float(), t = get_random_float();
    float theta = std::acos(1 - t);
    float u = 2 * M_PI * s;
    Vector3f localRay(std::sin(theta) * std::cos(u),
                        std::sin(theta) * std::sin(u), 1-t);
    return toWorld(localRay, N);
}

Vector3f Material::uniformHemiSample(const Vector3f &N) {
    float s = get_random_float(), t = get_random_float();
    float z = std::fabs(1.0f - 2.0f * t);
    float v = std::sqrt(1.0f - z * z);
    float u = 2 * M_PI * s;
    Vector3f localRay(v * std::cos(u), v * std::sin(u), z);
    return toWorld(localRay, N);
}

Vector3f Material::scatter(const Vector3f &N, SampleType t) {
    switch (m_type) {
        case DIFFUSE: {
            // uniform sample on the hemisphere
            if(t == UNIFORMAL)
                return uniformHemiSample(N);
            else if(t == IMPORTANCE)
                return cosineWeightedHemiSample(N);
        }
        case MICROFACET: {
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
            return Vector3f();
        }
    }
}

float Material::pdf(const Vector3f &wi, const Vector3f &wo, const Vector3f &N, SampleType t) {
    switch (m_type) {
        case DIFFUSE: {
            if(t == UNIFORMAL)
                return uniformHemiPdf(wo, N);
            else if (t == IMPORTANCE)
                return cosineWeightedHemiPdf(wo, N);
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

float Material::cosineWeightedHemiPdf(const Vector3f &wo, const Vector3f &N) {
    float costhetah = dotProduct(wo, N);
    if (costhetah > 0.0f) {
        return costhetah / M_PI;
    } else
        return 0.0f;
}

float Material::uniformHemiPdf(const Vector3f &wo, const Vector3f &N) {
    if (dotProduct(wo, N) > 0.0f)
        return 0.5f / M_PI;
    else
        return 0.0f;
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


// convert local to world..
// Vector3f Material::toWorld(const Vector3f &a, const Vector3f &N) {
//     Vector3f B, C;
//     if (std::fabs(N.x) > std::fabs(N.y)) {
//         float invLen = 1.0f / std::sqrt(N.x * N.x + N.z * N.z);
//         C = Vector3f(N.z * invLen, 0.0f, -N.x * invLen);
//     } else {
//         float invLen = 1.0f / std::sqrt(N.y * N.y + N.z * N.z);
//         C = Vector3f(0.0f, N.z * invLen, -N.y * invLen);
//     }
//     B = crossProduct(C, N);
//     return a.x * B + a.y * C + a.z * N;
// }

// convert local to world..
Vector3f Material::toWorld(const Vector3f &a, const Vector3f &N) {
    Vector3f vup;
    if(N.x > 0.9) {
        vup = Vector3f(0, 1, 0);
    } else {
        vup = Vector3f(1, 0, 0);
    }

    Vector3f t = normalize(crossProduct(vup, N));
    Vector3f s = normalize(crossProduct(t, N));
    return a.x * s + a.y * t + a.z * normalize(N);
}

// Compute reflection direction
Vector3f Material::reflect(const Vector3f &I, const Vector3f &N) {
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
Vector3f Material::refract(const Vector3f &I, const Vector3f &N,
                    const float &ior) {
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
void Material::fresnel(const Vector3f &I, const Vector3f &N, const float &ior,
                float &kr) {
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
#endif  // RAYTRACING_MATERIAL_H

#pragma once
#ifndef RAYTRACING_VECTOR_H
#define RAYTRACING_VECTOR_H

#include <iostream>
#include <cmath>
#include <algorithm>

class Vector3f {
public:
    float x, y, z;
    __host__ __device__ Vector3f() {}
    __host__ __device__ Vector3f(float f) : x(f), y(f), z(f) {}
    __host__ __device__ Vector3f(float xx, float yy, float zz): x(xx), y(yy), z(zz) {}

    // return the length of the vector
    __host__ __device__ float length() const { return std::sqrt(x * x + y * y + z*z); }
    __host__ __device__ float squareLength() const { return x * x + y * y + z*z; }


    // return ||v1 - v2||^2
    __host__ __device__ float secondNorm(const Vector3f& v1, const Vector3f& v2) { 
        Vector3f tmp = v1 - v2;
        return tmp.x * tmp.x + tmp.y * tmp.y + tmp.z * tmp.z; 
    }

    
    __host__ __device__ Vector3f normalize() {
        float n = this->length();
        return Vector3f(x / n, y / n, z / n);
    }

    __host__ __device__ Vector3f operator * (const float& r) const { return Vector3f(x * r, y * r, z * r); }
    __host__ __device__ friend Vector3f operator * (const float& r, const Vector3f& v) { return Vector3f(v.x * r, v.y * r, v.z * r); }

    __host__ __device__ Vector3f operator / (const float& r) const { return Vector3f(x / r, y / r, z / r); }
    __host__ __device__ Vector3f& operator /= (const float& r) { x /= r, y /= r, z /= r; return *this; }


    __host__ __device__ Vector3f operator * (const Vector3f& v) const { return Vector3f(x * v.x, y * v.y, z * v.z); }
    __host__ __device__ Vector3f operator + (const Vector3f& v) const { return Vector3f(x + v.x, y + v.y, z + v.z); }
    __host__ __device__ Vector3f operator - (const Vector3f& v) const { return Vector3f(x - v.x, y - v.y, z - v.z); }

    __host__ __device__ Vector3f operator - () const { return Vector3f(-x, -y, -z); }

    __host__ __device__ Vector3f& operator += (const Vector3f& v)  { x += v.x, y += v.y, z += v.z; return *this; }
    __host__ __device__ Vector3f& operator -= (const Vector3f& v)  { x -= v.x, y -= v.y, z -= v.z; return *this; }
    __host__ __device__ Vector3f& operator *= (const Vector3f& v)  { x *= v.x, y *= v.y, z *= v.z; return *this; }

    __host__ __device__ float operator[](int i) const { return (&x)[i]; }
    __host__ __device__ float& operator[](int i) { return (&x)[i]; }


    __host__ __device__ inline float r() const { return x; }
    __host__ __device__ inline float g() const { return y; }
    __host__ __device__ inline float b() const { return z; }

    
    __host__ __device__ inline bool nearZero() const {
        // Return true if the vector is close to zero in all dimensions.
        const float s = 1e-8;
        return (fabs(x) < s) && (fabs(y) < s) && (fabs(z) < s);
    }
    __host__ friend std::ostream& operator << (std::ostream &os, const Vector3f &v) {
        return os << v.x << ' ' << v.y << ' ' << v.z;
    }
};


__host__ __device__ Vector3f normalize(const Vector3f& v) {
    float n = sqrt(v.x * v.x + v.y * v.y + v.z*v.z);
    return Vector3f(v.x / n, v.y / n, v.z / n);
}

__host__ __device__ inline Vector3f cross(const Vector3f &v1, const Vector3f &v2) {
    return Vector3f((v1.y*v2.z - v1.z*v2.y),
            (-(v1.x*v2.z - v1.z*v2.x)),
            (v1.x*v2.y - v1.y*v2.z));
}

__host__ __device__ inline float dot(const Vector3f &v1, const Vector3f &v2) {
    return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
}

#endif //RAYTRACING_VECTOR_H
#pragma once

#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <thrust/random.h>

#include "glm/gtx/norm.hpp"

__forceinline__ __device__ float CosTheta(const glm::vec3& w) { return w.z; }
__forceinline__ __device__ float Cos2Theta(const glm::vec3& w) { float c = w.z; return c * c; }
__forceinline__ __device__ float Sin2Theta(const glm::vec3& w) { float c = w.z; return fmaxf(0.f, 1.f - c * c); }
__forceinline__ __device__ float SinTheta(const glm::vec3& w) { return sqrtf(Sin2Theta(w)); }
__forceinline__ __device__ float AbsCosTheta(const glm::vec3& w) { return fabsf(w.z); }
__forceinline__ __device__ float tanTheta(glm::vec3 w) { return SinTheta(w) / CosTheta(w); }
__forceinline__ __device__ float tan2Theta(glm::vec3 w) { return Sin2Theta(w) / Cos2Theta(w); }

__forceinline__ __device__ float CosPhi(glm::vec3 w) {
    float sinTheta = SinTheta(w);
    return (sinTheta == 0) ? 1 : glm::clamp(w.x / sinTheta, -1.f, 1.f);
}

__forceinline__ __device__ float SinPhi(glm::vec3 w) {
    float sinTheta = SinTheta(w);
    return (sinTheta == 0) ? 0 : glm::clamp(w.y / sinTheta, -1.f, 1.f);
}

__forceinline__ __device__ glm::vec3 SphericalDirection(float st, float ct, float phi) {
    return glm::vec3(st * cosf(phi), st * sinf(phi), ct);
}

struct Frame {
    glm::vec3 x, y, z;
    __host__ __device__ Frame() {};
    __host__ __device__ Frame(const glm::vec3& x_, const glm::vec3& y_, const glm::vec3& z_)
        : x(x_), y(y_), z(z_) {
    }
    __host__ __device__ static Frame FromZ(const glm::vec3& z) {
        float sign = copysignf(1.f, z.z);
        float a = -1.f / (sign + z.z);
        float b = z.x * z.y * a;
        glm::vec3 x(1.f + sign * z.x * z.x * a, sign * b, -sign * z.x);
        glm::vec3 y(b, sign + z.y * z.y * a, -z.y);
        // optional renorm
        return Frame(glm::normalize(x), glm::normalize(y), z);
    }
    __host__ __device__ glm::vec3 ToLocal(const glm::vec3& v) const { return { glm::dot(v,x), glm::dot(v,y), glm::dot(v,z) }; }
    __host__ __device__ glm::vec3 FromLocal(const glm::vec3& v) const { return v.x * x + v.y * y + v.z * z; }
};
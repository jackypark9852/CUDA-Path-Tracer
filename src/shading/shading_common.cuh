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

__forceinline__ __device__ void branchlessONB(const glm::vec3& n, glm::vec3& b1, glm::vec3& b2)
{
    float sign = copysignf(1.0f, n.z);
    const float a = -1.0f / (sign + n.z);
    const float b = n.x * n.y * a;
    b1 = glm::vec3(1.0f + sign * n.x * n.x * a, sign * b, -sign * n.x);
    b2 = glm::vec3(b, sign + n.y * n.y * a, -n.y);
}

__device__ __forceinline__ void worldToLocal(const glm::vec3& n, const glm::vec3& w, glm::vec3& wl) {
    glm::vec3 t, b; branchlessONB(n, t, b);
    wl = glm::vec3(glm::dot(w, t), glm::dot(w, b), glm::dot(w, n));
}

__device__ __forceinline__ glm::vec3 localToWorld(const glm::vec3& n, const glm::vec3& wl) {
    glm::vec3 t, b; branchlessONB(n, t, b);
    return wl.x * t + wl.y * b + wl.z * n;
}
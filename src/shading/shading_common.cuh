#pragma once

#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <thrust/random.h>
#include <cuda_runtime.h> 
#include "glm/gtx/norm.hpp"
#include "../sceneStructs.h"

__device__ __forceinline__ float CosTheta(const glm::vec3& w) { return w.z; }
__device__ __forceinline__ float Cos2Theta(const glm::vec3& w) { float c = w.z; return c * c; }
__device__ __forceinline__ float Sin2Theta(const glm::vec3& w) { float c = w.z; return fmaxf(0.f, 1.f - c * c); }
__device__ __forceinline__ float SinTheta(const glm::vec3& w) { return sqrtf(Sin2Theta(w)); }
__device__ __forceinline__ float AbsCosTheta(const glm::vec3& w) { return fabsf(w.z); }
__device__ __forceinline__ float TanTheta(glm::vec3 w) { return SinTheta(w) / CosTheta(w); }
__device__ __forceinline__ float Tan2Theta(glm::vec3 w) { return Sin2Theta(w) / Cos2Theta(w); }

__device__ __forceinline__ float CosPhi(glm::vec3 w) {
    float sinTheta = SinTheta(w);
    return (sinTheta == 0) ? 1 : glm::clamp(w.x / sinTheta, -1.f, 1.f);
}

__device__ __forceinline__ float Cos2Phi(glm::vec3 w) { return CosPhi(w) * CosPhi(w); }

__device__ __forceinline__ float SinPhi(glm::vec3 w) {
    float sinTheta = SinTheta(w);
    return (sinTheta == 0) ? 0 : glm::clamp(w.y / sinTheta, -1.f, 1.f);
}

__device__ __forceinline__ float Sin2Phi(glm::vec3 w) { return SinPhi(w) * SinPhi(w); }

__device__ __forceinline__ glm::vec3 SphericalDirection(float st, float ct, float phi) {
    return glm::vec3(st * cosf(phi), st * sinf(phi), ct);
}

__device__ __forceinline__ void branchlessONB(const glm::vec3& n, glm::vec3& b1, glm::vec3& b2)
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

__device__ __forceinline__ bool SameHemisphere(glm::vec3 w, glm::vec3 wp) {
    return w.z * wp.z > 0;
}

__device__ __forceinline__ bool HasBaseColorTex(const Material* mat) {
    return (mat->baseColorTex >= 0); 
}

__device__ __forceinline__ bool HasNormalTex(const Material* mat) {
    return (mat->normalTex >= 0);
}

__device__ __forceinline__ bool HasMetallicRoughnessTex(const Material* mat) {
    return (mat->metallicRoughnessTex >= 0);
}

__device__ __forceinline__ bool HasEmissiveTex(const Material* mat) {
    return (mat->emissiveTex >= 0);
}

__device__ __forceinline__ glm::vec3 MakeVec3(float4 c) {
    return glm::vec3(c.x, c.y, c.z);
}

__device__ __forceinline__ float fast_pow_pos(float a, float b) {
    return (a <= 0.0f) ? 0.0f : exp2f(b * log2f(a));
}

__device__ __forceinline__ float linear_to_srgb_f(float x) {
    x = fmaxf(x, 0.0f);
    return (x <= 0.0031308f)
        ? (12.92f * x)
        : (1.055f * fast_pow_pos(x, 1.0f / 2.4f) - 0.055f);
}

__device__ __forceinline__ float srgb_to_linear(float c) {
    c = fminf(fmaxf(c, 0.0f), 1.0f);
    return (c <= 0.04045f)
        ? (c / 12.92f)
        : fast_pow_pos((c + 0.055f) / 1.055f, 2.4f);
}

__device__ __forceinline__ glm::vec3 aces_v3(glm::vec3 x) {
    const float a = 2.51f, b = 0.03f, c = 2.43f, d = 0.59f, e = 0.14f;
    glm::vec3 num = x * (a * x + b);
    glm::vec3 den = x * (c * x + d) + e;
    glm::vec3 y = num / den;
    return glm::clamp(y, 0.0f, 1.0f);
}

__device__ __forceinline__ glm::vec3 reinhard_v3(glm::vec3 c, float exposure = 1.0f) {
    c *= exposure;
    c = c / (glm::vec3(1.0f) + c);
    return glm::clamp(c, 0.0f, 1.0f);
}

__device__ __forceinline__ glm::vec3 to_display_v3(glm::vec3 c) {
    // aces tonemap, then srgb oetf
    c = aces_v3(c);
    return glm::vec3(
        linear_to_srgb_f(c.x),
        linear_to_srgb_f(c.y),
        linear_to_srgb_f(c.z)
    );
}

__device__ __forceinline__ unsigned char to_u8(float x) {
    x = fminf(fmaxf(x, 0.0f), 1.0f);
    return static_cast<unsigned char>(x * 255.0f + 0.5f);
}

// from https://www.realtimerendering.com/raytracing/Ray%20Tracing%20in%20a%20Weekend.pdf
__device__ __forceinline__ glm::vec3 RandomInUnitDisk(thrust::default_random_engine& rng) {
    glm::vec3 p; 
    thrust::uniform_real_distribution<float> u01(0, 1);
    
    do {
        p = 2.0f * glm::vec3(u01(rng), u01(rng), 0) - glm::vec3(1.0f, 1.0f, 0.0f); 
    } while (dot(p, p) >= 1.0f);
    
    return p; 
}

template <typename T>
__device__ __forceinline__ void dswap(T& a, T& b) {
    T tmp = a;
    a = b;
    b = tmp;
}
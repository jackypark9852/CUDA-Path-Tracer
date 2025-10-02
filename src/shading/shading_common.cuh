#pragma once

#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <thrust/random.h>

#include "glm/gtx/norm.hpp"

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

// sample base color: uses texture if present, else constant
__device__ __forceinline__ glm::vec3 SampleBaseColor(const Material* mat, const cpt::Texture2D* textures, const glm::vec2& uv) {
    if (HasBaseColorTex(mat)) {
        cudaTextureObject_t texObj = textures[mat->baseColorTex].texObj; 
        float4 texel = tex2D<float4>(texObj, uv.x, uv.y);
        return MakeVec3(texel);
    }
    return mat->baseColor;
}
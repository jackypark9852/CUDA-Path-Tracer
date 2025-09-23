#pragma once

#include <glm/glm.hpp>
#include <cuda_runtime.h>
#include "../utilities.h"

#define DEVICE_INLINE __device__ __forceinline__

DEVICE_INLINE glm::vec3 fresnelSchlick(const glm::vec3& F0, float cosTheta) {
    float m = fmaxf(0.f, 1.f - cosTheta);
    float m2 = m * m, m5 = m2 * m2 * m;
    return F0 + (glm::vec3(1.f) - F0) * m5;
}

DEVICE_INLINE float D_GGX(float alpha, float NdotH) {
    NdotH = fmaxf(NdotH, 0.f);
    float a2 = alpha * alpha;
    float d = (NdotH * NdotH) * (a2 - 1.f) + 1.f;
    return a2 / (PI * d * d + 1e-7f);
}

DEVICE_INLINE float G_SmithGGX(float alpha, float NdotV, float NdotL) {
    auto lambda = [alpha](float NdotX) {
        NdotX = fmaxf(NdotX, 1e-6f);
        float a2 = alpha * alpha;
        float t = (1.f - NdotX * NdotX) / (NdotX * NdotX);
        return 0.5f * (sqrtf(1.f + a2 * t) - 1.f);
    };
    return 1.f / (1.f + lambda(NdotV) + lambda(NdotL) + 1e-7f);
}

#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <thrust/random.h>

#include "../utilities.h"

#include "device_launch_parameters.h"
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
#include "shading_kernels.cuh"
#include "shading_common.cuh" 

// each kernel shades one material type
__global__ void KernShadeEmissive(int iter, int n, ShadeableIntersection* s, PathSegment* p, Material* m) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) ShadeEmissiveImpl(iter, idx, s, p, m);
}

__global__ void KernShadeDiffuse(int iter, int n, ShadeableIntersection* s, PathSegment* p, Material* m) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) ShadeDiffuseImpl(iter, idx, s, p, m);
}

__global__ void KernShadeSpecular(int iter, int n, ShadeableIntersection* s, PathSegment* p, Material* m) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) ShadeSpecularImpl(iter, idx, s, p, m);
}

__global__ void KernShadeTransmissive(int iter, int n, ShadeableIntersection* s, PathSegment* p, Material* m) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) ShadeTransmissiveImpl(iter, idx, s, p, m);
}

__global__ void KernShadePbr(int iter, int n, ShadeableIntersection* s, PathSegment* p, Material* m) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) ShadePbrImpl(iter, idx, s, p, m);
}

// shades environment for rays that missed; terminates the path
__global__ void KernShadeEnvMap(int iter, int n, ShadeableIntersection* s, PathSegment* p, const cpt::Texture2D envMap) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) ShadeEnvMapImpl(iter, idx, s, p, envMap);
}

// writes a magenta error color for unknown material types
__global__ void KernShadeError(int iter, int n, ShadeableIntersection* s, PathSegment* p) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) ShadeErrorImpl(iter, idx, s, p);
}

// single-pass kernel that dispatches per-material shading
__global__ void KernShadeAllMaterials(int iter, int n, ShadeableIntersection* s, PathSegment* p, Material* m, cpt::Texture2D envMap) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    PathSegment* seg = p + idx;
    if (seg->shouldTerminate) return;

    ShadeableIntersection isect = s[idx];

    // rays that miss geometry are shaded by the environment
    if (isect.t < 0.0f) {
        ShadeEnvMapImpl(iter, idx, s, p, envMap);
        return;
    }

    // stop paths that exceeded bounce budget
    if (seg->remainingBounces <= 0) {
        seg->color = glm::vec3(0.0f);
        seg->shouldTerminate = true;
        return;
    }

    Material mat = m[isect.materialId];
    switch (mat.type) {
    case MaterialType::EMISSIVE:      ShadeEmissiveImpl(iter, idx, s, p, m);      break;
    case MaterialType::DIFFUSE:       ShadeDiffuseImpl(iter, idx, s, p, m);       break;
    case MaterialType::SPECULAR:      ShadeSpecularImpl(iter, idx, s, p, m);      break;
    case MaterialType::TRANSMISSIVE:  ShadeTransmissiveImpl(iter, idx, s, p, m);  break;
    case MaterialType::PBR:           ShadePbrImpl(iter, idx, s, p, m);           break;
    default:                          ShadeErrorImpl(iter, idx, s, p);            break;
    }
}

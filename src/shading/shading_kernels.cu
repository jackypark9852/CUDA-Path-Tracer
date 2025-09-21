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

__global__ void kernShadeEmissive(int iter, int n, ShadeableIntersection* s, PathSegment* p, Material* m) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) shadeEmissive_impl(iter, idx, s, p, m);
}

__global__ void kernShadeDiffuse(int iter, int n, ShadeableIntersection* s, PathSegment* p, Material* m) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) shadeDiffuse_impl(iter, idx, s, p, m);
}

__global__ void kernShadeSpecular(int iter, int n, ShadeableIntersection* s, PathSegment* p, Material* m) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) shadeSpecular_impl(iter, idx, s, p, m);
}

__global__ void kernShadeTransmissive(int iter, int n, ShadeableIntersection* s, PathSegment* p, Material* m) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) shadeTransmissive_impl(iter, idx, s, p, m);
}

__global__ void kernShadeMetallic(int iter, int n, ShadeableIntersection* s, PathSegment* p, Material* m) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) shadeMetallic_impl(iter, idx, s, p, m);
}

__global__ void kernShadeDielectric(int iter, int n, ShadeableIntersection* s, PathSegment* p, Material* m) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) shadeDielectric_impl(iter, idx, s, p, m);
}

__global__ void kernShadeAllMaterials(int iter, int n, ShadeableIntersection* s, PathSegment* p, Material* m) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    PathSegment* seg = p + idx;
    if (seg->shouldTerminate) return;

    ShadeableIntersection isect = s[idx];
    if (isect.t <= 0.0f || seg->remainingBounces <= 0) {
        seg->color = glm::vec3(0.0f);
        seg->shouldTerminate = true;
        return;
    }

    Material mat = m[isect.materialId];
    switch (mat.type) {
        case MaterialType::EMISSIVE:      shadeEmissive_impl(iter, idx, s, p, m);      break;
        case MaterialType::DIFFUSE:       shadeDiffuse_impl(iter, idx, s, p, m);       break;
        case MaterialType::SPECULAR:      shadeSpecular_impl(iter, idx, s, p, m);      break;
        case MaterialType::TRANSMISSIVE:  shadeTransmissive_impl(iter, idx, s, p, m);  break;
        case MaterialType::METALLIC:      shadeMetallic_impl(iter, idx, s, p, m);      break; 
        case MaterialType::DIELECTRIC:    shadeDielectric_impl(iter, idx, s, p, m);    break; 
        default: seg->shouldTerminate = true; break;
    }
}
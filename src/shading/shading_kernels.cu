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

__global__ void kernShadePbr(int iter, int n, ShadeableIntersection* s, PathSegment* p, Material* m) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) shadePbr_impl(iter, idx, s, p, m);
}

__global__ void kernrShadeEnvMap(int iter, int n, ShadeableIntersection* s, PathSegment* p, const cpt::Texture2D envMap)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) shadeEnvMap_impl(iter, idx, s, p, envMap);
}

__global__ void kernShadeAllMaterials(int iter, int n, ShadeableIntersection* s, PathSegment* p, Material* m, cpt::Texture2D envMap) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    PathSegment* seg = p + idx;
    if (seg->shouldTerminate) return;

    ShadeableIntersection isect = s[idx];
    // rays shot into the void should be shaded using the env map
    if (isect.t < 0.0f) {
        shadeEnvMap_impl(iter, idx, s, p, envMap);
        return;
    }
    
    // terminate rays that bounced too many times
    if (seg->remainingBounces <= 0) {
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
        case MaterialType::PBR:           shadePbr_impl(iter, idx, s, p, m);      break; 
        default: seg->shouldTerminate = true; break;
    }
}
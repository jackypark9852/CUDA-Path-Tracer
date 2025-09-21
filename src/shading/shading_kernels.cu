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

__global__ void shadePbrMaterial(
    int iter,
    int num_paths,
    ShadeableIntersection* shadeableIntersections,
    PathSegment* pathSegments,
    Material* materials)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_paths) return;

    ShadeableIntersection intersection = shadeableIntersections[idx];
    PathSegment* pathSegment = pathSegments + idx;

    if (pathSegment->shouldTerminate) return;

    if (intersection.t <= 0.0f || pathSegment->remainingBounces <= 0) {
        // miss or out of bounces
        pathSegment->color = glm::vec3(0.0f);
        pathSegment->shouldTerminate = true;
        return;
    }

    Material material = materials[intersection.materialId];
    glm::vec3 materialColor = material.baseColor;

    // NOTE: remove this if we want to use pbr and non-pbr kernel together 
    //       otherwise this pbr kernel will just paint the non-pbr surfaces magenta
    if (material.type != MaterialType::PBR) {
        glm::vec3 n = glm::normalize(intersection.surfaceNormal);
        pathSegment->color = 0.5f * (n + glm::vec3(1.0f));
        pathSegment->shouldTerminate = true;
        return;
    }

    // Set up the RNG
    thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, pathSegment->remainingBounces);
    thrust::uniform_real_distribution<float> u01(0, 1);

    // TODO: change this diffuse shading to actual pbr shading 
    glm::vec3 n = intersection.surfaceNormal;
    glm::vec3 wi = calculateRandomDirectionInHemisphere(n, rng);
    float cosIn = glm::max(0.f, dot(n, wi));
    float pdf = cosIn / glm::pi<float>();
    glm::vec3 f = material.baseColor / glm::pi<float>();
    pathSegment->color *= f * (cosIn / pdf);
    glm::vec3 hitP = pathSegment->ray.origin +
        pathSegment->ray.direction * intersection.t;
    pathSegment->ray.origin = hitP + n * EPSILON;
    pathSegment->ray.direction = wi;
    --pathSegment->remainingBounces;
    return;
}

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
        default: seg->shouldTerminate = true; break;
    }
}
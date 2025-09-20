#include <cstdio>
#include <cuda.h>
#include <cmath>

#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
#include "shading_kernels.cuh"
#include "shading_common.cuh" 
#include <thrust/random.h>
#include "../utilities.h"

__host__ __device__
thrust::default_random_engine makeSeededRandomEngine(int iter, int index, int depth)
{
    int h = utilhash((1 << 31) | (depth << 22) | iter) ^ utilhash(index);
    return thrust::default_random_engine(h);
};


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
    glm::vec3 materialColor = material.color;

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
    glm::vec3 f = material.color / glm::pi<float>();
    pathSegment->color *= f * (cosIn / pdf);
    glm::vec3 hitP = pathSegment->ray.origin +
        pathSegment->ray.direction * intersection.t;
    pathSegment->ray.origin = hitP + n * EPSILON;
    pathSegment->ray.direction = wi;
    --pathSegment->remainingBounces;
    return;
}

__global__ void shadePerfectMaterial(
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

    // Set up the RNG
    thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, pathSegment->remainingBounces);
    thrust::uniform_real_distribution<float> u01(0, 1);

    Material material = materials[intersection.materialId];
    glm::vec3 materialColor = material.color;

    switch (material.type) {
    case MaterialType::EMISSIVE:
        pathSegment->color *= material.color * material.emittance;
        pathSegment->shouldTerminate = true;
        return;

    case MaterialType::DIFFUSE: {
        glm::vec3 n = intersection.surfaceNormal;
        glm::vec3 wi = calculateRandomDirectionInHemisphere(n, rng);
        pathSegment->color *= material.color;
        glm::vec3 hitP = pathSegment->ray.origin +
            pathSegment->ray.direction * intersection.t;
        pathSegment->ray.origin = hitP + n * EPSILON;
        pathSegment->ray.direction = wi;
        --pathSegment->remainingBounces;
        return;
    }

    case MaterialType::SPECULAR: {
        pathSegment->color *= material.color;

        // reflect the ray
        glm::vec3 n = intersection.surfaceNormal;
        glm::vec3 wi = glm::reflect(pathSegment->ray.direction, n);
        glm::vec3 hitP = pathSegment->ray.origin +
            pathSegment->ray.direction * intersection.t;
        pathSegment->ray.origin = hitP + n * EPSILON;
        pathSegment->ray.direction = wi;
        --pathSegment->remainingBounces;
        return;
    }

    case MaterialType::TRANSMISSIVE: {
        // Hard-coded air glass
        const float etaA = 1.0f;
        const float etaB = material.indexOfRefraction;  // e.g., 1.55

        glm::vec3 n = glm::normalize(intersection.surfaceNormal);
        glm::vec3 I = glm::normalize(pathSegment->ray.direction);
        glm::vec3 wo = -I;

        const bool entering = glm::dot(wo, n) > 0.0f;
        glm::vec3 orientedN = entering ? n : -n;

        const float etaI = entering ? etaA : etaB;
        const float etaT = entering ? etaB : etaA;
        const float eta = etaI / etaT;

        glm::vec3 wi = glm::refract(I, orientedN, eta);

        glm::vec3 hitP = pathSegment->ray.origin + pathSegment->ray.direction * intersection.t;

        if (glm::length2(wi) == 0.0f) {
            wi = glm::reflect(I, orientedN);
            pathSegment->ray.origin = hitP + orientedN * EPSILON;
            pathSegment->ray.direction = glm::normalize(wi);
        }
        else {
            pathSegment->ray.origin = hitP - orientedN * EPSILON;
            pathSegment->ray.direction = glm::normalize(wi);
        }

        --pathSegment->remainingBounces;
        return;
    }

    default:
        pathSegment->shouldTerminate = true;
        return;
    }
}

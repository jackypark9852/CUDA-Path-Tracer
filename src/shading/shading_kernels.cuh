#pragma once
#include "../sceneStructs.h"
#include "../intersections.h"
#include "../interactions.h"
#include "../utilities.h"
#include "../texture.h"

#include "glm/gtx/norm.hpp"
#include "shading_bsdf.cuh"
#include "shading_common.cuh"
#include "shading_kernels.cuh"
#include <cuda_runtime.h>


#include <thrust/random.h>


#define DEVICE_INLINE __device__ __forceinline__


DEVICE_INLINE thrust::default_random_engine makeSeededRandomEngine(int iter, int index, int depth)
{
    int h = utilhash((1 << 31) | (depth << 22) | iter) ^ utilhash(index);
    return thrust::default_random_engine(h);
};


DEVICE_INLINE void shadeEmissive_impl(
    int iter, int idx,
    ShadeableIntersection* s,
    PathSegment* p,
    Material* m)
{
    ShadeableIntersection isect = s[idx];
    PathSegment* seg = p + idx;
    if (seg->shouldTerminate) return;
    if (isect.t <= 0.0f || seg->remainingBounces <= 0) {
        seg->color = glm::vec3(0.0f);   
        seg->shouldTerminate = true;
        return;
    }
    Material mat = m[isect.materialId];
    seg->color *= mat.baseColor * mat.emittance;
    seg->shouldTerminate = true;
}

DEVICE_INLINE void shadeDiffuse_impl(
    int iter, int idx,
    ShadeableIntersection* s,
    PathSegment* p,
    Material* m)
{
    ShadeableIntersection isect = s[idx];
    PathSegment* seg = p + idx;
    if (seg->shouldTerminate) return;
    if (isect.t <= 0.0f || seg->remainingBounces <= 0) {
        seg->color = glm::vec3(0.0f);
        seg->shouldTerminate = true;
        return;
    }
    thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, seg->remainingBounces);
    Material mat = m[isect.materialId];
    glm::vec3 n = isect.surfaceNormal;
    glm::vec3 wi = calculateRandomDirectionInHemisphere(n, rng);
    seg->color *= mat.baseColor;
    glm::vec3 hitP = seg->ray.origin + seg->ray.direction * isect.t;
    seg->ray.origin = hitP + n * EPSILON;
    seg->ray.direction = wi;
    --seg->remainingBounces;
}

DEVICE_INLINE void shadeSpecular_impl(
    int iter, int idx,
    ShadeableIntersection* s,
    PathSegment* p,
    Material* m)
{
    ShadeableIntersection isect = s[idx];
    PathSegment* seg = p + idx;
    if (seg->shouldTerminate) return;
    if (isect.t <= 0.0f || seg->remainingBounces <= 0) {
        seg->color = glm::vec3(0.0f);
        seg->shouldTerminate = true;
        return;
    }
    Material mat = m[isect.materialId];
    seg->color *= mat.baseColor;
    glm::vec3 n = isect.surfaceNormal;
    glm::vec3 wi = glm::reflect(seg->ray.direction, n);
    glm::vec3 hitP = seg->ray.origin + seg->ray.direction * isect.t;
    seg->ray.origin = hitP + n * EPSILON;
    seg->ray.direction = wi;
    --seg->remainingBounces;
}

DEVICE_INLINE void shadeTransmissive_impl(
    int iter, int idx,
    ShadeableIntersection* s,
    PathSegment* p,
    Material* m)
{
    ShadeableIntersection isect = s[idx];
    PathSegment* seg = p + idx;
    if (seg->shouldTerminate) return;
    if (isect.t <= 0.0f || seg->remainingBounces <= 0) {
        seg->color = glm::vec3(0.0f);
        seg->shouldTerminate = true;
        return;
    }
    Material mat = m[isect.materialId];
    const float etaA = 1.0f;
    const float etaB = mat.ior;
    glm::vec3 n = glm::normalize(isect.surfaceNormal);
    glm::vec3 I = glm::normalize(seg->ray.direction);
    glm::vec3 wo = -I;
    const bool entering = glm::dot(wo, n) > 0.0f;
    glm::vec3 orientedN = entering ? n : -n;
    const float etaI = entering ? etaA : etaB;
    const float etaT = entering ? etaB : etaA;
    const float eta = etaI / etaT;
    glm::vec3 wi = glm::refract(I, orientedN, eta);
    glm::vec3 hitP = seg->ray.origin + seg->ray.direction * isect.t;
    if (glm::length2(wi) < EPSILON) {
        wi = glm::reflect(I, orientedN);
        seg->ray.origin = hitP + orientedN * EPSILON;
        seg->ray.direction = glm::normalize(wi);
    }
    else {
        seg->ray.origin = hitP - orientedN * EPSILON;
        seg->ray.direction = glm::normalize(wi);
    }
    --seg->remainingBounces;
}

DEVICE_INLINE void shadePbr_impl(
    int iter, int idx,
    ShadeableIntersection* s,
    PathSegment* p,
    Material* m)
{
    ShadeableIntersection isect = s[idx];
    Material* mat = m + isect.materialId;
    PathSegment* seg = p + idx;
    BSDFSample sample{}; sample.pdf = 0.f;
    thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, seg->remainingBounces);

    float alpha = fmaxf(1e-4f, mat->roughness * mat->roughness);

    glm::vec3 F0_dielectric(0.04f);
    glm::vec3 F0 = (1.f - mat->metallic) * F0_dielectric + mat->metallic * mat->baseColor;

    glm::vec3 n = isect.surfaceNormal;
    glm::vec3 v = -seg->ray.direction;

    float NdotV = fmaxf(glm::dot(n, v), 0.f);
    glm::vec3 F_view = fresnelSchlick(F0, NdotV);
    float F_avg = (F_view.x + F_view.y + F_view.z) * (1.f / 3.f);

    // unnormalized lobe weights
    float wDiffuse = (1.f - mat->metallic) * (1.f - mat->transmission) * (1.f - F_avg);
    float wRefl = F_avg;
    float wTrans = (1.f - F_avg) * mat->transmission;

    // calculate probabilities from weights
    float wSum = wDiffuse + wRefl + wTrans + 1e-7f;
    float pDiffuse = wDiffuse / wSum;
    float pRefl = wRefl / wSum;
    float pTrans = wTrans / wSum;

    thrust::uniform_real_distribution<float> u01(0, 1);
    float xi = u01(rng); 
    if (xi < pDiffuse) {
        // do some diffuse sampling here
    }

    seg->color = glm::vec3(pDiffuse);
    seg->shouldTerminate = true;
}

// https://en.wikipedia.org/wiki/File:Equirectangular_projection_SW.jpg
DEVICE_INLINE glm::vec2 sphere2mapUV_Equirectangular(glm::vec3 p)
{
    return glm::vec2(
        atan2(p.x, -p.z) / (2 * PI) + .5,
        -p.y * .5 + .5
    );
}


DEVICE_INLINE void shadeEnvMap_impl(
    int iter, int idx,
    ShadeableIntersection* s,
    PathSegment* p,
    const cpt::Texture2D envMap
) 
{
    PathSegment* seg = p + idx; 
    glm::vec2 uv = sphere2mapUV_Equirectangular(normalize(seg->ray.direction));
    float4 texel = tex2D<float4>(envMap.texObj, uv.x, uv.y);
    
    seg->color *= glm::vec3(texel.x, texel.y, texel.z); 
    seg->shouldTerminate = true; 
}


DEVICE_INLINE void shadeError_impl(
    int iter, int idx,
    ShadeableIntersection* s,
    PathSegment* p
)
{
    PathSegment* seg = p + idx;

    glm::vec3 errorColor = glm::vec3(1.0f, 0.0f, 1.0f); // magenta
    seg->color = errorColor; 
    seg->shouldTerminate = true;
}


__global__ void kernShadeEmissive(
    int iter, int n, 
    ShadeableIntersection* s, 
    PathSegment* p, 
    Material* m); 

__global__ void kernShadeDiffuse(
    int iter, int n, 
    ShadeableIntersection* s, 
    PathSegment* p, 
    Material* m); 

__global__ void kernShadeSpecular(
    int iter, int n, 
    ShadeableIntersection* s, 
    PathSegment* p, 
    Material* m);

__global__ void kernShadeTransmissive(
    int iter, int n, 
    ShadeableIntersection* s, 
    PathSegment* p, 
    Material* m);

__global__ void kernShadePbr(
    int iter, int n,
    ShadeableIntersection* s,
    PathSegment* p,
    Material* m);

__global__ void kernShadeEnvMap(
    int iter, int n,
    ShadeableIntersection* s,
    PathSegment* p,
    const cpt::Texture2D envMap);

__global__ void kernShadeError(
    int iter, int n,
    ShadeableIntersection* s,
    PathSegment* p);

__global__ void kernShadeAllMaterials(
    int iter, int num_paths,
    ShadeableIntersection* shadeableIntersections,
    PathSegment* pathSegments,
    Material* materials,
    const cpt::Texture2D envMap);

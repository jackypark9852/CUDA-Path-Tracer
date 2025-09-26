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

DEVICE_INLINE BSDFSample sampleBSDF(
    ShadeableIntersection* isect,
    PathSegment* seg,
    Material* mat,
    thrust::default_random_engine& rng)
{
    BSDFSample sample{}; sample.pdf = 0.f;
    thrust::uniform_real_distribution<float> u01(0, 1);

    // always work with unit vectors for ggx math
    glm::vec3 n = glm::normalize(isect->surfaceNormal);
    glm::vec3 woWorld = glm::normalize(-seg->ray.direction);
    glm::vec3 woLocal; worldToLocal(n, woWorld, woLocal);

    // f0 follows disney: dielectric from ior, metallic uses basecolor
    glm::vec3 F0_dielectric = f0_from_ior(mat->ior);
    glm::vec3 F0 = glm::mix(F0_dielectric, mat->baseColor, mat->metallic);

    float NdotV = fmaxf(glm::dot(n, woWorld), 0.f);
    glm::vec3 F_view = fresnelSchlick(F0, NdotV);
    float F_avg = (F_view.x + F_view.y + F_view.z) * (1.f / 3.f);

    // two lobes only: diffuse + specular
    float wDiffuse = (1.f - mat->metallic) * (1.f - F_avg);
    float wRefl = F_avg;

    // optional floor so specular still gets sampled for low-f0 dielectrics
    float wRefl_sample = fmaxf(0.08f, wRefl);

    //float wSum = wDiffuse + wRefl_sample + 1e-7f;
    //float pDiffuse = wDiffuse / wSum;
    //float pRefl = wRefl_sample / wSum;

    float wSum = wDiffuse + wRefl_sample + 1e-7f;
    float pDiffuse = 0.0f; 
    float pRefl = 1.0f;


    float xi = u01(rng);

    // ---- diffuse path ----
    //if (xi < pDiffuse) {
    //    glm::vec3 wi = calculateRandomDirectionInHemisphere(n, rng); // cosine-weighted, world
    //    wi = glm::normalize(wi);

    //    float NdotL = fmaxf(glm::dot(n, wi), 0.0f);
    //    float fd_fres = disney_diffuse_fresnel(NdotL, NdotV);

    //    // bsdf value: disney-ish lambert scaled by (1 - metallic) and fresnel factor
    //    glm::vec3 fd = (1.f - mat->metallic) * fd_fres * lambertBRDF(mat->baseColor);

    //    sample.incomingDir = wi;               // world
    //    sample.bsdfValue = fd;               // brdf value
    //    sample.pdf = lambertPDF(NdotL) * pDiffuse; // include lobe prob
    //    sample.isDelta = false;
    //    return sample;
    //}

    // ---- specular reflection path ----
    {
        float alpha = mat->roughness * mat->roughness;
        glm::vec3 wiWorld;
        float pdf_lobe = 0.f;

        glm::vec3 f_spec = Sample_f_microfacet_refl_vndf(
            F0, n, woLocal, alpha, u01(rng), u01(rng), wiWorld, pdf_lobe);

        if (pdf_lobe <= 0.f || !isfinite(pdf_lobe)) { sample.pdf = 0.f; return sample; }

        sample.incomingDir = glm::normalize(wiWorld);
        sample.bsdfValue = f_spec;
        sample.pdf = pdf_lobe * pRefl;   // <-- include mixture prob!
        sample.isDelta = false;
        return sample;
    }

}


DEVICE_INLINE void shadePbr_impl(
    int iter, int idx,
    ShadeableIntersection* s,
    PathSegment* p,
    Material* m)
{
    ShadeableIntersection* isect = s + idx;
    PathSegment* seg = p + idx;
    Material* mat = m + isect->materialId;
    if (seg->shouldTerminate) return;

    // early exit if no valid hit or no bounces left
    if (isect->t <= 0.f || seg->remainingBounces <= 0) {
        seg->color = glm::vec3(0.f);
        seg->shouldTerminate = true;
        return;
    }

    thrust::default_random_engine rng =
        makeSeededRandomEngine(iter, idx, seg->remainingBounces);

    BSDFSample sample = sampleBSDF(isect, seg, mat, rng);

    glm::vec3 n = glm::normalize(isect->surfaceNormal);
    glm::vec3 wi = sample.incomingDir;
    float pdf = sample.pdf;
    glm::vec3 f = sample.bsdfValue;

    float NdotWi = fmaxf(glm::dot(n, wi), 0.f);

    // guard against invalid sampling cases
    if (pdf <= 1e-7f) {
        seg->color = glm::vec3(1.f, 0.f, 0.f); // red = bad pdf
        seg->shouldTerminate = true;
        return;
    }

    if (NdotWi <= 0.f) {
        seg->color = glm::vec3(0.f, 1.f, 0.f); // green = backfacing / wrong hemi
        seg->shouldTerminate = true;
        return;
    }

    if (!isfinite(pdf)) {
        seg->color = glm::vec3(0.f, 0.f, 1.f); // blue = nan/inf pdf
        seg->shouldTerminate = true;
        return;
    }

    if (!isfinite(f.x) || !isfinite(f.y) || !isfinite(f.z)) {
        seg->color = glm::vec3(1.f, 1.f, 0.f); // yellow = nan/inf bsdf value
        seg->shouldTerminate = true;
        return;
    }

    if (!isfinite(wi.x) || !isfinite(wi.y) || !isfinite(wi.z)) {
        seg->color = glm::vec3(1.f, 0.f, 1.f); // magenta = nan/inf dir
        seg->shouldTerminate = true;
        return;
    }

    // optional clamp to control fireflies
    float ratio = fminf(NdotWi / pdf, 1e4f);

    // advance ray
    glm::vec3 hitP = seg->ray.origin + seg->ray.direction * isect->t;
    seg->ray.origin = hitP + n * EPSILON;
    seg->ray.direction = glm::normalize(wi);

    // accumulate throughput
    seg->color *= f * fminf(NdotWi / pdf, 1e4f);


    --seg->remainingBounces;
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

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

// device-inline helper for cuda kernels
#define DEVICE_INLINE static __device__ __forceinline__

// makes a deterministic per-path rng seeded by iteration, index, and depth
DEVICE_INLINE thrust::default_random_engine MakeSeededRandomEngine(int iter, int index, int depth) {
    int h = utilhash((1 << 31) | (depth << 22) | iter) ^ utilhash(index);
    return thrust::default_random_engine(h);
}

// shades purely emissive hits and terminates the path
DEVICE_INLINE void ShadeEmissiveImpl(
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
    seg->color *= mat.baseColor * mat.emissiveStrength;
    seg->shouldTerminate = true;
}

// cosine-weighted diffuse bounce
DEVICE_INLINE void ShadeDiffuseImpl(
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
    thrust::default_random_engine rng = MakeSeededRandomEngine(iter, idx, seg->remainingBounces);
    Material mat = m[isect.materialId];
    glm::vec3 n = isect.surfaceNormal;
    // uses project-provided cosine hemisphere sampler (world space)
    glm::vec3 wi = CalculateRandomDirectionInHemisphere(n, rng);
    seg->color *= mat.baseColor;
    glm::vec3 hitP = seg->ray.origin + seg->ray.direction * isect.t;
    seg->ray.origin = hitP + n * EPSILON;
    seg->ray.direction = wi;
    --seg->remainingBounces;
}

// perfect mirror reflection
DEVICE_INLINE void ShadeSpecularImpl(
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

// simple transmission using glm::refract with eta from material ior
DEVICE_INLINE void ShadeTransmissiveImpl(
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

    // maintains correct normal orientation for inside/outside
    const bool entering = glm::dot(wo, n) > 0.0f;
    glm::vec3 orientedN = entering ? n : -n;

    const float etaI = entering ? etaA : etaB;
    const float etaT = entering ? etaB : etaA;
    const float eta = etaI / etaT;

    glm::vec3 wi = glm::refract(I, orientedN, eta);
    glm::vec3 hitP = seg->ray.origin + seg->ray.direction * isect.t;

    // total internal reflection fallback
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

struct LobeInputs {
    glm::vec3 baseColor;
    float metallic;
    float roughness;    // [0..1]
    glm::vec3 nW;       // world shading normal (after normal map)
    glm::vec3 woW;      // world outgoing (toward camera)
    glm::vec3 woL;      // local outgoing (+z is n)
    float alpha;        // roughness^2
    glm::vec3 F0;       // conductor/dielectric mix
};

// builds common inputs
DEVICE_INLINE LobeInputs BuildLobeInputs(
    const ShadeableIntersection* isect,
    const PathSegment* seg,
    const Material* mat,
    const cpt::Texture2D* tex)
{
    LobeInputs li{};
    li.baseColor = SampleBaseColor(mat, tex, isect->uv);
    li.metallic = SampleMetallic(mat, tex, isect->uv).x;
    li.roughness = SampleRoughness(mat, tex, isect->uv).x;
    li.nW = ApplyNormalMap(mat, tex, isect);
    li.woW = glm::normalize(-seg->ray.direction);
    worldToLocal(li.nW, li.woW, li.woL);
    glm::vec3 F0d = F0FromIOR(mat->ior);
    li.F0 = glm::mix(F0d, li.baseColor, li.metallic);
    li.alpha = li.roughness * li.roughness;
    return li;
}

// opaque: diffuse + ggx refl (+ optional ms)
DEVICE_INLINE BSDFSample SampleOpaqueBSDF(
    const ShadeableIntersection* isect,
    const PathSegment* seg,
    const Material* mat,
    const cpt::Texture2D* tex,
    thrust::default_random_engine& rng)
{
    BSDFSample s{}; s.pdf = 0.f;
    thrust::uniform_real_distribution<float> U(0, 1);

    LobeInputs li = BuildLobeInputs(isect, seg, mat, tex);
    float NdotV = fmaxf(glm::dot(li.nW, li.woW), 0.f);
    glm::vec3 Fv = FresnelSchlick(li.F0, NdotV);
    float Favg = (Fv.x + Fv.y + Fv.z) * (1.f / 3.f);

    float wDiffuse = (1.f - li.metallic) * (1.f - Favg);
    float wRefl = Favg;
    float wMS = MicrofacetMSWeight(li.alpha, Favg);

    float wReflS = fmaxf(0.08f, wRefl);
    float wSum = wDiffuse + wReflS + wMS + 1e-7f;
    float pDiffuse = wDiffuse / wSum;
    float pRefl = wReflS / wSum;
    float pMS = wMS / wSum;

    float xi = U(rng);

    if (xi < pDiffuse) {
        glm::vec3 wi = glm::normalize(CalculateRandomDirectionInHemisphere(li.nW, rng));
        float NdotL = fmaxf(glm::dot(li.nW, wi), 0.f);
        float fdFr = DisneyDiffuseFresnel(NdotL, NdotV);
        glm::vec3 fd = (1.f - li.metallic) * fdFr * LambertBRDF(li.baseColor);

        s.incomingDir = wi;
        s.bsdfValue = fd;
        s.pdf = LambertPDF(NdotL) * pDiffuse;
        s.isDelta = false;
        return s;
    }
    else if (xi < pDiffuse + pRefl) {
        glm::vec3 wiW; float pdfLobe = 0.f;
        glm::vec3 fSpec = SampleMicrofacetReflVNDF(
            li.F0, li.nW, li.woL, li.alpha, U(rng), U(rng), wiW, pdfLobe);
        s.incomingDir = glm::normalize(wiW);
        s.bsdfValue = fSpec;
        s.pdf = pdfLobe * pRefl;
        s.isDelta = false;
        return s;
    }
    else {
        // optional microfacet ms as diffuse-like fallback
        glm::vec3 wi = glm::normalize(CalculateRandomDirectionInHemisphere(li.nW, rng));
        float NdotL = fmaxf(glm::dot(li.nW, wi), 0.f);
        glm::vec3 msTint = MicrofacetMSTint(li.baseColor, li.metallic);
        glm::vec3 fms = wMS * MicrofacetMSBrdf(msTint);

        s.incomingDir = wi;
        s.bsdfValue = fms;
        s.pdf = LambertPDF(NdotL) * pMS;
        s.isDelta = false;
        return s;
    }
}

DEVICE_INLINE BSDFSample SampleDielectricBSDF(
    const ShadeableIntersection* isect,
    const PathSegment* seg,
    const Material* mat,
    const cpt::Texture2D* tex,
    thrust::default_random_engine& rng)
{
    BSDFSample s{}; s.pdf = 0.f;
    thrust::uniform_real_distribution<float> U(0, 1);

    LobeInputs li = BuildLobeInputs(isect, seg, mat, tex);

    float etaI = 1.f, etaT = mat->ior;
    if (li.woL.z < 0.f) { etaI = mat->ior; etaT = 1.f; }

    bool tir = false;
    float F_exact = FresnelDielectricExact(fabsf(glm::dot(li.woW, li.nW)), etaI, etaT, tir);

    float wRefl = F_exact;
    float wTrans = (!tir ? (mat->transmission * (1.f - F_exact)) : 0.f);

    float wReflS = fmaxf(0.08f, wRefl);
    float wSum = wReflS + wTrans + 1e-7f;
    float pRefl = wReflS / wSum;
    float pTrans = wTrans / wSum;

    float xi = U(rng);

    if (xi < pRefl) {
        glm::vec3 wiW; float pdfLobe = 0.f;
        glm::vec3 fSpec = SampleMicrofacetReflVNDF(
            li.F0, li.nW, li.woL, li.alpha, U(rng), U(rng), wiW, pdfLobe);
        s.incomingDir = glm::normalize(wiW);
        s.bsdfValue = fSpec;
        s.pdf = pdfLobe * pRefl;
        s.isDelta = false;
        return s;
    }
    else {
        glm::vec3 wiW, fTrans; float pdfLobe = 0.f;
        bool ok = SampleMicrofacetTransmission_GGX(
            li.nW, li.woW, li.alpha, etaI, etaT, U(rng), U(rng), wiW, fTrans, pdfLobe);
        if (!ok || pdfLobe <= 0.f) { s.pdf = 0.f; return s; }

        glm::vec3 tint = li.baseColor;
        glm::vec3 value = mat->transmission * tint * fTrans;

        s.incomingDir = glm::normalize(wiW);
        s.bsdfValue = value;
        s.pdf = pdfLobe * pTrans;
        s.isDelta = false;
        return s;
    }
}

// single shade that advances ray and applies throughput
DEVICE_INLINE void ShadeUnifiedAdvance(
    const ShadeableIntersection* isect,
    PathSegment* seg,
    const Material* mat,
    const cpt::Texture2D* tex,
    const BSDFSample& samp)
{
    glm::vec3 nW = ApplyNormalMap(mat, tex, isect);
    glm::vec3 wi = glm::normalize(samp.incomingDir);

    glm::vec3 hitP = seg->ray.origin + seg->ray.direction * isect->t;

    glm::vec3 woW = glm::normalize(-seg->ray.direction);
    bool entering = glm::dot(woW, nW) > 0.0f;
    glm::vec3 orientedN = entering ? nW : -nW;

    bool isTransmission = (glm::dot(nW, woW) * glm::dot(nW, wi)) < 0.0f;
    glm::vec3 offsetN = isTransmission ? (-orientedN) : orientedN;

    seg->ray.origin = hitP + offsetN * EPSILON;
    seg->ray.direction = wi;

    float cosNI = fabsf(glm::dot(nW, wi));
    seg->color *= samp.bsdfValue * fminf(cosNI / fmaxf(samp.pdf, 1e-8f), FLT_MAX);
    --seg->remainingBounces;
}

DEVICE_INLINE void ShadePbrImpl(
    int iter, int idx,
    ShadeableIntersection* isects,
    PathSegment* paths,
    Material* mats,
    cpt::Texture2D* texs)
{
    ShadeableIntersection* isect = isects + idx;
    PathSegment* seg = paths + idx;
    Material* mat = mats + isect->materialId;

    if (seg->shouldTerminate) return;
    if (isect->t <= 0.f || seg->remainingBounces <= 0) { seg->color = glm::vec3(0); seg->shouldTerminate = true; return; }

    // emissive hit early out
    glm::vec3 Le = SampleEmissive(mat, texs, isect->uv);
    if (Le.x > EPSILON || Le.y > EPSILON || Le.z > EPSILON) {
        seg->color *= Le;
        seg->shouldTerminate = true;
        return;
    }

    thrust::default_random_engine rng = MakeSeededRandomEngine(iter, idx, seg->remainingBounces);

    bool isDielectric = (mat->transmission > 0.0f);
    BSDFSample s = isDielectric
        ? SampleDielectricBSDF(isect, seg, mat, texs, rng)
        : SampleOpaqueBSDF(isect, seg, mat, texs, rng);

    if (s.pdf <= 0.f) { seg->color = glm::vec3(0); seg->shouldTerminate = true; return; }
    ShadeUnifiedAdvance(isect, seg, mat, texs, s);
}

// maps a direction on the unit sphere to equirectangular uwv
DEVICE_INLINE glm::vec2 Sphere2MapUvEquirectangular(glm::vec3 p) {
    return glm::vec2(
        atan2(p.x, -p.z) / (2 * PI) + .5f,
        -p.y * .5f + .5f
    );
}

// shades with an environment map and terminates the path
DEVICE_INLINE void ShadeEnvMapImpl(
    int iter, int idx,
    ShadeableIntersection* s,
    PathSegment* p,
    const cpt::Texture2D envMap)
{
    PathSegment* seg = p + idx;
    glm::vec2 uv = Sphere2MapUvEquirectangular(normalize(seg->ray.direction));
    float4 texel = tex2D<float4>(envMap.texObj, uv.x, uv.y);

    seg->color *= glm::vec3(texel.x, texel.y, texel.z);
    seg->shouldTerminate = true;
}

// writes a conspicuous error color and terminates the path
DEVICE_INLINE void ShadeErrorImpl(
    int iter, int idx,
    ShadeableIntersection* s,
    PathSegment* p)
{
    PathSegment* seg = p + idx;
    glm::vec3 errorColor = glm::vec3(1.0f, 0.0f, 1.0f);
    seg->color = errorColor;
    seg->shouldTerminate = true;
}

DEVICE_INLINE void ShadeNormalImpl(
    int iter, int idx,
    ShadeableIntersection* intersections,
    Material* materials, 
    cpt::Texture2D* textures, 
    PathSegment* pathSegments)
{
    ShadeableIntersection* isect = intersections + idx;
    Material* mat = materials + isect->materialId;
    PathSegment* seg = pathSegments + idx;
    const glm::vec3 n = ApplyNormalMap(mat, textures, isect);
    glm::vec3 c = 0.5f * (n + glm::vec3(1.0f));
    c = glm::clamp(c, glm::vec3(0.0f), glm::vec3(1.0f));
    seg->color = (isect->t > 0.0f)? c : glm::vec3(1.0f);
    seg->shouldTerminate = true;
}

DEVICE_INLINE void ShadeAlbedoImpl(
    int iter, int idx,
    ShadeableIntersection* intersections,
    Material* materials, 
    cpt::Texture2D* textures, 
    PathSegment* pathSegments)
{
    ShadeableIntersection* isect = intersections + idx;
    PathSegment* seg = pathSegments + idx;
    Material* mat = materials + isect->materialId;

    glm::vec3 c = SampleBaseColor(mat, textures, isect->uv);
    seg->color = (isect->t > 0.0f) ? c : glm::vec3(1.0f);
    seg->shouldTerminate = true;
}

DEVICE_INLINE void ShadeRoughnessImpl(
    int iter, int idx,
    ShadeableIntersection* intersections,
    Material* materials,
    cpt::Texture2D* textures,
    PathSegment* pathSegments)
{
    ShadeableIntersection* isect = intersections + idx;
    PathSegment* seg = pathSegments + idx;
    Material* mat = materials + isect->materialId;

    glm::vec3 c = SampleRoughness(mat, textures, isect->uv);
    seg->color = (isect->t > 0.0f) ? c : glm::vec3(1.0f);
    seg->shouldTerminate = true;
}

DEVICE_INLINE void ShadeMetallicImpl(
    int iter, int idx,
    ShadeableIntersection* intersections,
    Material* materials,
    cpt::Texture2D* textures,
    PathSegment* pathSegments)
{
    ShadeableIntersection* isect = intersections + idx;
    PathSegment* seg = pathSegments + idx;
    Material* mat = materials + isect->materialId;

    glm::vec3 c = SampleMetallic(mat, textures, isect->uv);
    seg->color = (isect->t > 0.0f) ? c : glm::vec3(1.0f);
    seg->shouldTerminate = true;
}

__global__ void KernShadeEmissive(
    int iter, int n,
    ShadeableIntersection* s,
    PathSegment* p,
    Material* m);

__global__ void KernShadeDiffuse(
    int iter, int n,
    ShadeableIntersection* s,
    PathSegment* p,
    Material* m);

__global__ void KernShadeSpecular(
    int iter, int n,
    ShadeableIntersection* s,
    PathSegment* p,
    Material* m);

__global__ void KernShadeTransmissive(
    int iter, int n,
    ShadeableIntersection* s,
    PathSegment* p,
    Material* m);

__global__ void KernShadePbr(
    int iter, int n,
    ShadeableIntersection* s,
    PathSegment* p,
    Material* m, 
    cpt::Texture2D* t);

__global__ void KernShadeEnvMap(
    int iter, int n,
    ShadeableIntersection* s,
    PathSegment* p,
    const cpt::Texture2D envMap);

__global__ void KernShadeError(
    int iter, int n,
    ShadeableIntersection* s,
    PathSegment* p);

__global__ void KernShadeNormal(
    int iter, int n, 
    ShadeableIntersection* intersections, 
    Material* materials, 
    cpt::Texture2D* textures, 
    PathSegment* pathSegments); 

__global__ void KernShadeAlbedo(
    int iter, int n,
    ShadeableIntersection* intersections,
    Material* materials,
    cpt::Texture2D* textures,
    PathSegment* pathSegments);

__global__ void KernShadeRoughness(
    int iter, int n,
    ShadeableIntersection* intersections,
    Material* materials,
    cpt::Texture2D* textures,
    PathSegment* pathSegments);

__global__ void KernShadeMetallic(
    int iter, int n,
    ShadeableIntersection* intersections,
    Material* materials,
    cpt::Texture2D* textures,
    PathSegment* pathSegments);

__global__ void KernShadeAllMaterials(
    int iter, int num_paths,
    ShadeableIntersection* shadeableIntersections,
    PathSegment* pathSegments,
    Material* materials,
    cpt::Texture2D* textures, 
    const cpt::Texture2D envMap);

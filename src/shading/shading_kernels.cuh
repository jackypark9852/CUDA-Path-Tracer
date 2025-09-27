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
#define DEVICE_INLINE static __device__

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
    seg->color *= mat.baseColor * mat.emittance;
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

struct SmoothRefractSample {
    glm::vec3 wiW = glm::vec3(0);
    glm::vec3 f = glm::vec3(0);
    float     pdf = 1.f;    // delta convention
    bool      isDelta = true;
};

DEVICE_INLINE SmoothRefractSample SampleSmoothRefractOnly(
    const glm::vec3& nW,   // world normal
    const glm::vec3& woW,  // world outgoing
    float iorB             // material ior (glass)
) {
    // implements only refraction. no fresnel split. on tir, falls back to perfect reflection.
    // throughput scaling: transmission uses eta^2, reflection uses 1.
    SmoothRefractSample out{};

    glm::vec3 woL; worldToLocal(nW, glm::normalize(woW), woL);

    const float etaA = 1.0f;
    const float etaB = iorB;
    const bool entering = (woL.z > 0.f);
    const float etaI = entering ? etaA : etaB;
    const float etaT = entering ? etaB : etaA;
    const float eta = etaI / etaT;

    glm::vec3 wiL;
    bool refrOk = RefractLocal(woL, eta, wiL);

    if (refrOk) {
        // refraction branch (no fresnel): weight = eta^2
        out.wiW = localToWorld(nW, wiL);
        const float cosNI = fmaxf(1e-6f, fabsf(glm::dot(nW, out.wiW)));
        const float weight = eta * eta;
        out.f = glm::vec3(weight / cosNI);
        out.pdf = 1.f;      // delta
        out.isDelta = true;
        return out;
    }
    else {
        // tir fallback: perfect mirror, weight = 1
        glm::vec3 wiLR = glm::reflect(-woL, glm::vec3(0, 0, 1));
        if (wiLR.z <= 0.f) wiLR.z = fabsf(wiLR.z); // safety
        out.wiW = localToWorld(nW, wiLR);
        const float cosNI = fmaxf(1e-6f, fabsf(glm::dot(nW, out.wiW)));
        out.f = glm::vec3(1.f / cosNI);
        out.pdf = 1.f;      // delta
        out.isDelta = true;
        return out;
    }
}

// samples the material bsdf (diffuse + ggx specular) and returns a single sampled lobe
DEVICE_INLINE BSDFSample SampleBSDF(
    ShadeableIntersection* isect,
    PathSegment* seg,
    Material* mat,
    thrust::default_random_engine& rng)
{
    BSDFSample sample{}; sample.pdf = 0.f;
    thrust::uniform_real_distribution<float> u01(0, 1);

    // uses local frame (+z = normal) for microfacet math
    glm::vec3 n = glm::normalize(isect->surfaceNormal);
    glm::vec3 woWorld = glm::normalize(-seg->ray.direction);
    glm::vec3 woLocal; worldToLocal(n, woWorld, woLocal);

    // dielectric f0 from ior; blend toward baseColor if metallic
    glm::vec3 F0_dielectric = F0FromIOR(mat->ior);
    glm::vec3 F0 = glm::mix(F0_dielectric, mat->baseColor, mat->metallic);

    float NdotV = fmaxf(glm::dot(n, woWorld), 0.f);
    glm::vec3 Fv = FresnelSchlick(F0, NdotV);
    float F_avg = (Fv.x + Fv.y + Fv.z) * (1.f / 3.f);
    float alpha = mat->roughness * mat->roughness;

    // transmission params
    const bool transmissive = (mat->metallic < EPSILON && mat->transmission > 0.0f);
    float etaI = 1.0f, etaT = mat->ior;
    if (woLocal.z < 0.f) { etaI = mat->ior; etaT = 1.0f; } // outside or inside

    // lobe weights: diffuse suppressed by fresnel and metallic, specular by fresnel
    float wDiffuse = (1.f - mat->metallic) * (1.f - mat->transmission) * (1.f - F_avg);
    float wRefl = F_avg;    
    float wMS = (transmissive)? 0.0f : MicrofacetMSWeight(alpha, F_avg);
    float wTrans = transmissive ? (1.f - F_avg) : 0.f; // transmission is complementary to 

    // sampling floors
    //float wReflS = fmaxf(0.08f, wRefl);
    float wReflS = 0.0f; 
    float wTransS = wTrans;

    // mixture probabilities
    float wSum = wDiffuse + wReflS + wTransS + wMS + 1e-7f;
    float pDiffuse = wDiffuse / wSum;
    float pRefl = wReflS / wSum;
    float pTrans = wTransS / wSum;
    float pMS = wMS / wSum;


    float xi = u01(rng);

    // diffuse lobe (cosine-weighted in world space)
    if (xi < pDiffuse) {
        glm::vec3 wi = CalculateRandomDirectionInHemisphere(n, rng);
        wi = glm::normalize(wi);

        float NdotL = fmaxf(glm::dot(n, wi), 0.0f);
        float fdFr = DisneyDiffuseFresnel(NdotL, NdotV);

        glm::vec3 fd = (1.f - mat->metallic) * fdFr * LambertBRDF(mat->baseColor);

        sample.incomingDir = wi;                     // world space
        sample.bsdfValue = fd;                     // brdf value
        sample.pdf = LambertPDF(NdotL) * pDiffuse; // include mixture weight
        sample.isDelta = false;
        return sample;
    }
    // specular reflection lobe sampled via vndf
    else if (xi < pDiffuse + pRefl) {
        float alpha = mat->roughness * mat->roughness;
        glm::vec3 wiWorld;
        float pdfLobe = 0.f;

        glm::vec3 fSpec = SampleMicrofacetReflVNDF(
            F0, n, woLocal, alpha, u01(rng), u01(rng), wiWorld, pdfLobe);

        sample.incomingDir = glm::normalize(wiWorld);
        sample.bsdfValue = fSpec;
        sample.pdf = pdfLobe * pRefl;
        sample.isDelta = false;
        return sample;
    }
    else if (xi < pDiffuse + pRefl + pTrans) {
        SmoothRefractSample g = SampleSmoothRefractOnly(
            glm::normalize(isect->surfaceNormal), // world normal
            woWorld,                              // world outgoing
            mat->ior                               // glass ior
        );

        BSDFSample sample{};
        sample.incomingDir = glm::normalize(g.wiW);
        sample.bsdfValue = glm::vec3(1.0f);    // encodes eta^2 or 1 divided by |cosNI|
        sample.pdf = g.pdf;  // 1 for delta
        sample.isDelta = g.isDelta;
        return sample;
    }

    // Multiple scattering compensation
    else {
        glm::vec3 wi = CalculateRandomDirectionInHemisphere(n, rng);
        wi = glm::normalize(wi);
        float NdotL = fmaxf(glm::dot(n, wi), 0.0f);
        glm::vec3 msTint = MicrofacetMSTint(mat->baseColor, mat->metallic); 
        glm::vec3 fms = wMS * MicrofacetMSBrdf(msTint);   // scale brdf by ms weight

        sample.incomingDir = wi;
        sample.bsdfValue = fms;
        sample.pdf = LambertPDF(NdotL) * pMS;
        sample.isDelta = false;
        return sample;
    }
}

DEVICE_INLINE void ShadePbrImpl(
    int iter, int idx,
    ShadeableIntersection* s,
    PathSegment* p,
    Material* m)
{
    ShadeableIntersection* isect = s + idx;
    PathSegment* seg = p + idx;
    Material* mat = m + isect->materialId;
    if (seg->shouldTerminate) return;

    // early out if miss or exhausted
    if (isect->t <= 0.f || seg->remainingBounces <= 0) {
        seg->color = glm::vec3(0.f);
        seg->shouldTerminate = true;
        return;
    }

    thrust::default_random_engine rng =
        MakeSeededRandomEngine(iter, idx, seg->remainingBounces);

    BSDFSample sample = SampleBSDF(isect, seg, mat, rng);

    // dev-note: prefer geometric normal for smooth dielectric interfaces if available
    glm::vec3 nW = glm::normalize(isect->surfaceNormal);
    glm::vec3 wi = glm::normalize(sample.incomingDir);
    float pdf = sample.pdf;
    glm::vec3 f = sample.bsdfValue;

    // basic validity guards
    if (!(pdf > 0.f) || !isfinite(pdf)) { seg->shouldTerminate = true; return; }
    if (!isfinite(wi.x) || !isfinite(wi.y) || !isfinite(wi.z)) { seg->shouldTerminate = true; return; }

    // advance ray origin and direction with oriented normal logic
    glm::vec3 hitP = seg->ray.origin + seg->ray.direction * isect->t;

    // compute entering based on outgoing (toward camera) direction
    glm::vec3 woW = glm::normalize(-seg->ray.direction);
    bool entering = glm::dot(woW, nW) > 0.0f;
    glm::vec3 orientedN = entering ? nW : -nW;

    // transmission if wi and wo are on opposite sides of the surface
    bool isTransmission = (glm::dot(nW, woW) * glm::dot(nW, wi)) < 0.0f;

    // choose offset direction and epsilon
    float epsRefl = EPSILON;                         // keep project default for reflection
    float epsRefr = 3e-4f;
    glm::vec3 offsetN = isTransmission ? (-orientedN) : orientedN;
    float eps = isTransmission ? epsRefr : epsRefl;

    seg->ray.origin = hitP + offsetN * eps;
    seg->ray.direction = wi;

    // accumulate throughput with standard path tracing weight
    float cosNI = fabsf(glm::dot(nW, wi));
    seg->color *= f * fminf(cosNI / pdf, FLT_MAX);

    --seg->remainingBounces;
}

// maps a direction on the unit sphere to equirectangular uv
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
    Material* m);

__global__ void KernShadeEnvMap(
    int iter, int n,
    ShadeableIntersection* s,
    PathSegment* p,
    const cpt::Texture2D envMap);

__global__ void KernShadeError(
    int iter, int n,
    ShadeableIntersection* s,
    PathSegment* p);

__global__ void KernShadeAllMaterials(
    int iter, int num_paths,
    ShadeableIntersection* shadeableIntersections,
    PathSegment* pathSegments,
    Material* materials,
    const cpt::Texture2D envMap);

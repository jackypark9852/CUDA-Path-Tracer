#pragma once

#include <glm/glm.hpp>
#include <cuda_runtime.h>
#include "../utilities.h"
#include "shading_common.cuh"

// device inline helper for cuda kernels
#define DEVICE_INLINE static __device__

// schlick fresnel with clamped cosine
DEVICE_INLINE glm::vec3 FresnelSchlick(const glm::vec3& F0, float cosTheta) {
    float ct = fminf(fmaxf(cosTheta, 0.f), 1.f);
    float m = 1.f - ct;
    float m2 = m * m, m5 = m2 * m2 * m;
    return F0 + (glm::vec3(1.f) - F0) * m5;
}

// heitz vndf sampling for isotropic ggx (alpha = roughness^2 recommended)
// expects wo to be in the local shading frame (n = +z)
DEVICE_INLINE glm::vec3 SampleWhVNDF(const glm::vec3& wo, float alpha, float u1, float u2) {
    // stretch view vector
    glm::vec3 v = glm::normalize(glm::vec3(alpha * wo.x, alpha * wo.y, wo.z));

    // build orthonormal basis around v
    float lensq = v.x * v.x + v.y * v.y;
    glm::vec3 T1 = lensq > 0 ? glm::vec3(-v.y, v.x, 0) / sqrtf(lensq) : glm::vec3(1, 0, 0);
    glm::vec3 T2 = glm::cross(v, T1);

    // sample unit disk
    float r = sqrtf(u1);
    float phi = 2.f * PI * u2;
    float t1 = r * cosf(phi);
    float t2 = r * sinf(phi);

    // stretch-compensated y component (heitz trick)
    float s = 0.5f * (1.f + v.z);
    t2 = (1.f - s) * sqrtf(1.f - t1 * t1) + s * t2;

    // reproject onto hemisphere, then unstretch
    glm::vec3 nh = t1 * T1 + t2 * T2 + sqrtf(fmaxf(0.f, 1.f - t1 * t1 - t2 * t2)) * v;
    glm::vec3 wh = glm::normalize(glm::vec3(alpha * nh.x, alpha * nh.y, fmaxf(0.f, nh.z)));
    return wh;
}

// smith lambda for isotropic ggx; diverges at grazing angles as expected
DEVICE_INLINE float Lambda(glm::vec3 w, float alpha) {
    float absTanTheta = fabsf(TanTheta(w));
    if (!isfinite(absTanTheta)) return INFINITY; // safe guard at grazing
    float aTan = alpha * absTanTheta;
    return 0.5f * (-1.0f + sqrtf(1.0f + aTan * aTan));
}

// normal distribution function: isotropic ggx
DEVICE_INLINE float DGGX(glm::vec3 wh, float alpha) {
    float NdotH = AbsCosTheta(wh);
    float a2 = alpha * alpha;
    float denom = (NdotH * NdotH) * (a2 - 1.f) + 1.f;
    return a2 / (PI * denom * denom);
}

// smith g1 for ggx (separable form)
DEVICE_INLINE float G1SmithGGX(glm::vec3 w, float alpha) {
    float tan2 = Tan2Theta(w);
    if (isinf(tan2)) return 0.f;
    float a2 = alpha * alpha;
    return 2.f / (1.f + sqrtf(1.f + a2 * tan2));
}

// exact smith g using lambda (preferred over product of g1s here)
DEVICE_INLINE float GSmithGGX(glm::vec3 wo, glm::vec3 wi, float alpha) {
    return 1.0f / (1.0f + Lambda(wo, alpha) + Lambda(wi, alpha));
}

// microfacet reflection bsdf evaluation (cook-torrance with ggx)
DEVICE_INLINE glm::vec3 MicrofacetRefl(glm::vec3 F0, glm::vec3 wo, glm::vec3 wi, float alpha) {
    float cosO = AbsCosTheta(wo);
    float cosI = AbsCosTheta(wi);
    if (cosI <= 0.f || cosO <= 0.f) return glm::vec3(0);

    glm::vec3 wh = normalize(wi + wo);
    float Fh = fmaxf(0.f, glm::dot(wi, wh));
    glm::vec3 F = FresnelSchlick(F0, Fh);

    float D = DGGX(wh, alpha);
    float G = GSmithGGX(wo, wi, alpha);

    return (F * D * G) / (4.f * cosI * cosO);
}

// importance sample the ggx microfacet reflection lobe using vndf
// returns bsdf value; outputs sampled wi in world frame and its pdf under this lobe
DEVICE_INLINE glm::vec3 SampleMicrofacetReflVNDF(
    const glm::vec3& F0, const glm::vec3& n, const glm::vec3& woLocal, float alpha,
    float u1, float u2, glm::vec3& wiWorld, float& pdf_wi_lobe)
{
    // handle delta-like mirror for tiny roughness
    const float ALPHA_EPS = 1e-5f;
    if (alpha <= ALPHA_EPS) {
        glm::vec3 wiLocal = reflect(-woLocal, glm::vec3(0, 0, 1));
        if (wiLocal.z <= 0.f) return glm::vec3(0);

        wiWorld = localToWorld(n, wiLocal);
        pdf_wi_lobe = 1.f; // delta distribution convention
        float Fh = fmaxf(0.f, glm::dot(wiLocal, glm::vec3(0, 0, 1)));
        glm::vec3 F = FresnelSchlick(F0, Fh);
        const float c = fmaxf(1e-7f, wiLocal.z);
        return F / c;
    }

    // sample half-vector via vndf and reflect
    glm::vec3 wh = SampleWhVNDF(woLocal, alpha, u1, u2);
    glm::vec3 wiLocal = reflect(-woLocal, wh);
    if (wiLocal.z <= 0.f) return glm::vec3(0);

    // pdf: p(wi) = D(wh) * G1(wo) / (4 * |wo * wh|)
    float D = DGGX(wh, alpha);
    float G1o = G1SmithGGX(woLocal, alpha);
    float CosVo = fmaxf(1e-6f, AbsCosTheta(woLocal));
    pdf_wi_lobe = (D * G1o) / (4.f * CosVo);

    if (!(pdf_wi_lobe > 0.f) || !isfinite(pdf_wi_lobe)) return glm::vec3(0);

    wiWorld = localToWorld(n, wiLocal);
    return MicrofacetRefl(F0, woLocal, wiLocal, alpha);
}

// dielectric f0 from ior (used by disney and others)
DEVICE_INLINE glm::vec3 F0FromIOR(float ior) {
    float f = (ior - 1.f) / (ior + 1.f);
    float f0 = f * f;
    return glm::vec3(f0);
}

// disney-ish diffuse fresnel factor (no retro-reflection term)
DEVICE_INLINE float DisneyDiffuseFresnel(float NdotL, float NdotV) {
    float FL = powf(fmaxf(0.f, 1.f - NdotL), 5.f);
    float FV = powf(fmaxf(0.f, 1.f - NdotV), 5.f);
    return (1.f - 0.5f * FL) * (1.f - 0.5f * FV);
}

// simple shape that grows with roughness; 
// this is a fit based on heuristic; better implementation would use LUT for MSWeight described in Heitz's paper 
DEVICE_INLINE float MicrofacetMSWeight(float alpha, float Favg) {
    // base growth from roughness
    float kAlpha = 1.f - 1.f / ((1.f + alpha) * (1.f + alpha)); // 1 - (1+alpha)^-2
    // less ms when average fresnel is already high (lots of single-scatter return)
    float kF = 1.f - Favg; 
    return fmaxf(0.f, fminf(1.f, kAlpha * kF));
}

DEVICE_INLINE glm::vec3 MicrofacetMSBrdf(const glm::vec3& tint) {
    // diffuse-like redistribution of lost energy
    return tint * INV_PI;
}

// tint for microfacet multiple-scattering compensation
// dielectrics: use basecolor scaled by (1 - metallic)
// metals:      use full basecolor (metal color)
DEVICE_INLINE glm::vec3 MicrofacetMSTint(const glm::vec3& baseColor, float metallic) {
    float scale = (metallic < 0.5f) ? (1.f - metallic) : 1.f;
    return scale * baseColor;
}

// lambert brdf and pdf helpers
DEVICE_INLINE glm::vec3 LambertBRDF(const glm::vec3& albedo) { return albedo * INV_PI; }
DEVICE_INLINE float     LambertPDF(float NdotL) { return fmaxf(NdotL, 0.f) * INV_PI; }

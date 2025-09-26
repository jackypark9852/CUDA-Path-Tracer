#pragma once

#include <glm/glm.hpp>
#include <cuda_runtime.h>
#include "../utilities.h"
#include "shading_common.cuh"

#define DEVICE_INLINE static __device__

DEVICE_INLINE glm::vec3 fresnelSchlick(const glm::vec3& F0, float cosTheta) {
    float m = fmaxf(0.f, 1.f - cosTheta);
    float m2 = m * m, m5 = m2 * m2 * m;
    return F0 + (glm::vec3(1.f) - F0) * m5;
}

DEVICE_INLINE glm::vec3 sample_wh_vndf(const glm::vec3& wo, float alpha, float u1, float u2) {
    // stretch view
    glm::vec3 v = glm::normalize(glm::vec3(alpha * wo.x, alpha * wo.y, wo.z));

    // build orthonormal basis (t,b,n) around v
    float lensq = v.x * v.x + v.y * v.y;
    glm::vec3 T1 = lensq > 0 ? glm::vec3(-v.y, v.x, 0) / sqrtf(lensq) : glm::vec3(1, 0, 0);
    glm::vec3 T2 = glm::cross(v, T1);

    // sample disk
    float r = sqrtf(u1);
    float phi = 2.f * PI * u2;
    float t1 = r * cosf(phi);
    float t2 = r * sinf(phi);
    float s = 0.5f * (1.f + v.z);
    t2 = (1.f - s) * sqrtf(1.f - t1 * t1) + s * t2;

    // reproject onto hemisphere
    glm::vec3 nh = t1 * T1 + t2 * T2 + sqrtf(fmaxf(0.f, 1.f - t1 * t1 - t2 * t2)) * v;

    // unstretch
    glm::vec3 wh = glm::normalize(glm::vec3(alpha * nh.x, alpha * nh.y, fmaxf(0.f, nh.z)));
    return wh;
}

DEVICE_INLINE float Lambda(glm::vec3 w, float roughness) {
    float absTanTheta = abs(TanTheta(w));
    if (isinf(absTanTheta)) return 0.;

    // Compute alpha for direction w
    float alpha =
        sqrt(Cos2Phi(w) * roughness * roughness + Sin2Phi(w) * roughness * roughness);
    float alpha2Tan2Theta = (roughness * absTanTheta) * (roughness * absTanTheta);
    return (-1 + sqrt(1.f + alpha2Tan2Theta)) / 2;
}

// D: isotropic GGX
DEVICE_INLINE float D_GGX(glm::vec3 wh, float alpha) {
    float NdotH = AbsCosTheta(wh);
    float a2 = alpha * alpha;
    float denom = (NdotH * NdotH) * (a2 - 1.f) + 1.f;
    return a2 / (PI * denom * denom);
}

// G: Smith GGX (separable)
DEVICE_INLINE float G1_SmithGGX(glm::vec3 w, float alpha) {
    float tan2 = Tan2Theta(w);
    if (isinf(tan2)) return 0.f;
    float a2 = alpha * alpha;
    return 2.f / (1.f + sqrtf(1.f + a2 * tan2));
}
DEVICE_INLINE float G_SmithGGX(glm::vec3 wo, glm::vec3 wi, float alpha) {
    return G1_SmithGGX(wo, alpha) * G1_SmithGGX(wi, alpha);
}

DEVICE_INLINE glm::vec3 f_microfacet_refl(glm::vec3 F0, glm::vec3 wo, glm::vec3 wi, float alpha) {
    float cosO = AbsCosTheta(wo);
    float cosI = AbsCosTheta(wi);
    if (cosI <= 0.f || cosO <= 0.f) return glm::vec3(0);

    glm::vec3 wh = normalize(wi + wo);
    float Fh = fmaxf(0.f, glm::dot(wi, wh));
    glm::vec3 F = fresnelSchlick(F0, Fh);

    float D = D_GGX(wh, alpha);
    float G = G_SmithGGX(wo, wi, alpha);

    return (F * D * G) / (4.f * cosI * cosO);
}

DEVICE_INLINE glm::vec3 Sample_f_microfacet_refl_vndf(
    const glm::vec3& F0, const glm::vec3& n, const glm::vec3& woLocal, float alpha,
    float u1, float u2, glm::vec3& wiWorld, float& pdf_wi_lobe)
{
    pdf_wi_lobe = 0.f;

    if (woLocal.z <= 0.f) return glm::vec3(0);

    // --- inside Sample_f_microfacet_refl_vndf ---
    glm::vec3 wh = sample_wh_vndf(woLocal, alpha, u1, u2);
    glm::vec3 wiLocal = reflect(-woLocal, wh);
    if (wiLocal.z <= 0.f) return glm::vec3(0);

    float NdotH = fmaxf(1e-7f, AbsCosTheta(wh));
    float WoDotH = fmaxf(1e-7f, glm::dot(woLocal, wh));

    float D = D_GGX(wh, alpha);
    float G1o = G1_SmithGGX(woLocal, alpha);

    pdf_wi_lobe = (D * G1o * NdotH) / (4.f * WoDotH);

    if (pdf_wi_lobe <= 1e-7f) {
        pdf_wi_lobe = 0.0f; 
    }

    if (!(pdf_wi_lobe > 0.f) || !isfinite(pdf_wi_lobe)) return glm::vec3(0);

    wiWorld = localToWorld(n, wiLocal);
    return f_microfacet_refl(F0, woLocal, wiLocal, alpha);
}


// compute dielectric f0 from ior (used by disney)
DEVICE_INLINE glm::vec3 f0_from_ior(float ior) {
    // f0 = ((eta - 1)/(eta + 1))^2
    float f = (ior - 1.f) / (ior + 1.f);
    float f0 = f * f;
    return glm::vec3(f0);
}

// disney-ish diffuse fresnel factor (no retro term here for now)
DEVICE_INLINE float disney_diffuse_fresnel(float NdotL, float NdotV) {
    // fl = (1 - ndotl)^5, fv = (1 - ndotv)^5
    float FL = powf(fmaxf(0.f, 1.f - NdotL), 5.f);
    float FV = powf(fmaxf(0.f, 1.f - NdotV), 5.f);
    // (1 - 0.5 fl)(1 - 0.5 fv)
    return (1.f - 0.5f * FL) * (1.f - 0.5f * FV);
}


DEVICE_INLINE glm::vec3 lambertBRDF(const glm::vec3& albedo) {return albedo * INV_PI;}
DEVICE_INLINE float     lambertPDF(float NdotL) {return fmaxf(NdotL, 0.f) * INV_PI;}


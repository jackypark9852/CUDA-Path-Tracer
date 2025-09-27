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

// fresnel for dielectrics
DEVICE_INLINE float FresnelDielectric(float cosThetaI, float etaI, float etaT) {
    cosThetaI = fminf(fmaxf(cosThetaI, -1.f), 1.f);
    bool entering = cosThetaI > 0.f;
    if (!entering) { float tmp = etaI; etaI = etaT; etaT = tmp; cosThetaI = fabsf(cosThetaI); }
    float sinThetaI = sqrtf(fmaxf(0.f, 1.f - cosThetaI * cosThetaI));
    float sinThetaT = etaI / etaT * sinThetaI;
    if (sinThetaT >= 1.f) return 1.f;
    float cosThetaT = sqrtf(fmaxf(0.f, 1.f - sinThetaT * sinThetaT));
    float rParl = ((etaT * cosThetaI) - (etaI * cosThetaT)) / ((etaT * cosThetaI) + (etaI * cosThetaT));
    float rPerp = ((etaI * cosThetaI) - (etaT * cosThetaT)) / ((etaI * cosThetaI) + (etaT * cosThetaT));
    return 0.5f * (rParl * rParl + rPerp * rPerp);
}


DEVICE_INLINE bool RefractLocal(const glm::vec3& woL, float eta, glm::vec3& wiLOut) {
    // snell refraction in local (+z is normal). returns false on tir.
    float cosI = woL.z;
    float sin2I = fmaxf(0.f, 1.f - cosI * cosI);
    float sin2T = eta * eta * sin2I;
    if (sin2T >= 1.f) return false; // tir
    float cosT = sqrtf(fmaxf(0.f, 1.f - sin2T));

    // transmitted dir flips hemisphere across the interface
    float z = (woL.z > 0.f) ? -cosT : cosT;
    wiLOut = glm::normalize(glm::vec3(-eta * woL.x, -eta * woL.y, z));
    return true;
}


// ggx microfacet btdf (isotropic), local frame (+z = normal)
// wi and wo must be on opposite sides of the interface (wi.z * wo.z < 0)
DEVICE_INLINE glm::vec3 MicrofacetBTDF(
    const glm::vec3& wo, const glm::vec3& wi,
    float alpha, float etaI, float etaT)
{
    if (wo.z == 0.f || wi.z == 0.f) return glm::vec3(0);
    if (wo.z * wi.z >= 0.f) return glm::vec3(0); // must be opposite hemispheres

    // half-vector for transmission: points to the same side as the normal
    glm::vec3 m = glm::normalize(wo + wi * (etaT / etaI));
    if (m.z < 0.f) m = -m;

    float D = DGGX(m, alpha);
    float G = GSmithGGX(wo, wi, alpha);

    float cosWoM = fabsf(glm::dot(wo, m));
    float cosWiM = fabsf(glm::dot(wi, m));

    float F = FresnelDielectric(glm::dot(wo, m), etaI, etaT); // reflectance for the microfacet
    float denom = (etaI * cosWoM + etaT * cosWiM);
    if (denom <= 0.f) return glm::vec3(0);

    // transmission factor (walter 2007): (1 - F) * D * G * etaT^2 * |wi*m| * |wo*m|
    // / (|wi*n| * |wo*n| * (etaI * wo*m + etaT * wi*m)^2)
    float scale = (1.f - F) * D * G * (etaT * etaT) * (cosWiM * cosWoM)
        / (fabsf(wi.z) * fabsf(wo.z) * denom * denom);
   
    return glm::vec3(scale);
}

// sample the ggx transmission lobe using vndf half-vector sampling + snell refraction
// returns bsdf value and pdf for the transmission lobe; wiWorld is in world space
//DEVICE_INLINE glm::vec3 SampleMicrofacetBTDFVNDF(
//    const glm::vec3& n,            // world normal (+z in local frame)
//    const glm::vec3& woLocal,      // outgoing in local frame
//    float alpha, float etaI, float etaT,
//    float u1, float u2,
//    glm::vec3& wiWorld, float& pdfLobe)
//{
//    const float ALPHA_EPS = 1e-5f;
//
//    // alpha == 0 : perfect smooth interface => delta event (Fresnel split)
//    if (alpha <= ALPHA_EPS) {
//        // macro normal for the microfacet
//        glm::vec3 m = glm::vec3(0.f, 0.f, 1.f);
//        if (woLocal.z < 0.f) m = -m;
//
//        const float F = FresnelDielectric(glm::dot(woLocal, m), etaI, etaT);
//
//        // use u1 to choose reflection vs refraction by Fresnel
//        if (u1 < F) {
//            // reflection branch
//            glm::vec3 wiLocal = reflect(-woLocal, m);
//            if (wiLocal.z == 0.f) { pdfLobe = 1.f; return glm::vec3(1.f); }
//            wiWorld = localToWorld(n, wiLocal);
//            pdfLobe = 1.f;       // delta convention
//            return glm::vec3(1.f);
//        }
//        else {
//            // refraction branch
//            const float eta = (woLocal.z > 0.f) ? (etaI / etaT) : (etaT / etaI);
//            glm::vec3 wiLocal;
//            if (!RefractMicrofacet(woLocal, m, eta, wiLocal)) {
//                // in theory with Fresnel split this should not happen, but guard anyway:
//                glm::vec3 wiRef = reflect(-woLocal, m);
//                wiWorld = localToWorld(n, wiRef);
//                pdfLobe = 1.f;   // delta
//                return glm::vec3(1.f);
//            }
//            wiWorld = localToWorld(n, wiLocal);
//            pdfLobe = 1.f;       // delta
//            return glm::vec3(1.f);
//        }
//    }
//
//    // rough branch: standard VNDF + Walter mapping
//    pdfLobe = 0.f;
//
//    glm::vec3 m = SampleWhVNDF(woLocal, alpha, u1, u2);
//    if (m.z <= 0.f) m = -m; // keep m on macro-normal side
//
//    // refraction across m
//    glm::vec3 wiLocal;
//    const float eta = (woLocal.z > 0.f) ? (etaI / etaT) : (etaT / etaI);
//    if (!RefractMicrofacet(woLocal, m, eta, wiLocal)) return glm::vec3(0);
//
//    // must be opposite hemispheres
//    if (wiLocal.z * woLocal.z >= 0.f) return glm::vec3(0);
//
//    // pdf mapped from half-vector domain
//    const float D = DGGX(m, alpha);
//    const float G1 = G1SmithGGX(woLocal, alpha);
//    const float cosWoM = fabsf(glm::dot(woLocal, m));
//    const float cosWiM = fabsf(glm::dot(wiLocal, m));
//    const float denom = fmaxf(1e-7f, etaI * cosWoM + etaT * cosWiM);
//
//    // p(wi) = D(m) * G1(wo) * |cosWoM| * |cosWiM| * etaT^2 /
//    //         ( |wo.n| * |wi.n| * (etaI*cosWoM + etaT*cosWiM)^2 )
//    pdfLobe = (D * G1 * cosWoM * cosWiM * (etaT * etaT)) /
//        (fmaxf(1e-7f, fabsf(woLocal.z) * fabsf(wiLocal.z)) * denom * denom);
//
//    // bsdf value (Walter 2007)
//    glm::vec3 f = MicrofacetBTDF(woLocal, wiLocal, alpha, etaI, etaT);
//
//    wiWorld = localToWorld(n, wiLocal);
//    return f;
//}


// lambert brdf and pdf helpers
DEVICE_INLINE glm::vec3 LambertBRDF(const glm::vec3& albedo) { return albedo * INV_PI; }
DEVICE_INLINE float     LambertPDF(float NdotL) { return fmaxf(NdotL, 0.f) * INV_PI; }

#pragma once

#include <glm/glm.hpp>
#include <cuda_runtime.h>
#include "../utilities.h"
#include "shading_common.cuh"

// device inline helper for cuda kernels
#define DEVICE_INLINE static __device__ __forceinline__

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

// exact fresnel for real ior, unpolarized. handles brewster + tir.
DEVICE_INLINE float FresnelDielectricExact(float cosThetaI, float etaI, float etaT, bool& tir)
{
    cosThetaI = fminf(fmaxf(cosThetaI, -1.0f), 1.0f);
    float ei = etaI, et = etaT;
    bool entering = cosThetaI > 0.0f;
    if (!entering) { float tmp = ei; ei = et; et = tmp; cosThetaI = fabsf(cosThetaI); }

    float sinThetaI = sqrtf(fmaxf(0.0f, 1.0f - cosThetaI * cosThetaI));
    float sinThetaT = ei / et * sinThetaI;

    if (sinThetaT >= 1.0f) { tir = true; return 1.0f; } // total internal reflection

    float cosThetaT = sqrtf(fmaxf(0.0f, 1.0f - sinThetaT * sinThetaT));

    float Rs = (ei * cosThetaI - et * cosThetaT) / (ei * cosThetaI + et * cosThetaT);
    float Rp = (ei * cosThetaT - et * cosThetaI) / (ei * cosThetaT + et * cosThetaI);
    tir = false;
    return 0.5f * (Rs * Rs + Rp * Rp);
}

// ggx microfacet btdf (isotropic), local frame (+z = normal)
// wi and wo must be on opposite sides of the interface (wi.z * wo.z < 0)
DEVICE_INLINE glm::vec3 MicrofacetBTDF(
    const glm::vec3& wo, const glm::vec3& wi,
    float alpha, float etaI, float etaT)
{
    if (wo.z == 0.f || wi.z == 0.f) return glm::vec3(0);
    if (wo.z * wi.z >= 0.f) return glm::vec3(0);

    glm::vec3 m = glm::normalize(wo + wi * (etaT / etaI));
    if (m.z < 0.f) m = -m;

    float D = DGGX(m, alpha);
    float G = GSmithGGX(wo, wi, alpha);

    float cosWoM = fabsf(glm::dot(wo, m));
    float cosWiM = fabsf(glm::dot(wi, m));

    bool tir = false; 
    float F = FresnelDielectricExact(glm::dot(wo, m), etaI, etaT, tir);
    float denom = (etaI * cosWoM + etaT * cosWiM);
    if (denom <= 0.f) return glm::vec3(0);

    float scale = (1.f - F) * D * G * (etaT * etaT) * (cosWiM * cosWoM)
        / (fabsf(wi.z) * fabsf(wo.z) * denom * denom);
   
    return glm::vec3(scale);
}

// refract across the microfacet normal m (local frame, +z is surface normal)
DEVICE_INLINE bool RefractThroughFacet(const glm::vec3& wo, const glm::vec3& m,
    float etaI, float etaT, glm::vec3& wi)
{
    float cosWoM = glm::dot(wo, m);
    float eta = (wo.z > 0.f) ? (etaI / etaT) : (etaT / etaI);

    float sin2I = fmaxf(0.f, 1.f - cosWoM * cosWoM);
    float sin2T = eta * eta * sin2I;
    if (sin2T >= 1.f) return false; // total internal reflection at the microfacet

    float cosT = sqrtf(fmaxf(0.f, 1.f - sin2T));
    wi = glm::normalize(eta * -wo + (eta * cosWoM - cosT) * m);
    return true;
}

// pdf for ggx microfacet transmission with vndf sampling of m
DEVICE_INLINE float PdfMicrofacetBTDF(const glm::vec3& wo, const glm::vec3& wi,
    const glm::vec3& m, float alpha,
    float etaI, float etaT)
{
    const float D = DGGX(m, alpha);
    const float G1wo = G1SmithGGX(wo, alpha);
    const float cosWo = fmaxf(1e-6f, AbsCosTheta(wo));
    const float cosWoM = fabsf(glm::dot(wo, m));
    const float cosWiM = fabsf(glm::dot(wi, m));
    const float denom = (etaI * cosWoM + etaT * cosWiM);
    if (cosWo <= 0.f || denom <= 0.f) return 0.f;

    const float pm = (D * G1wo * cosWoM) / cosWo;
    const float dmdw = (etaT * etaT * cosWiM) / (denom * denom);
    return pm * dmdw;
}

// sample ggx transmission (world-space in/out), evaluate btdf and its lobe pdf
DEVICE_INLINE bool SampleMicrofacetTransmission_GGX(
    const glm::vec3& nW,           // world shading normal
    const glm::vec3& woW,          // world outgoing (toward camera)
    float alpha,
    float etaI, float etaT,
    float u1, float u2,
    glm::vec3& wiW, glm::vec3& f, float& pdfLobe)
{
    // delta fallback for perfectly smooth case
    const float ALPHA_EPS = 1e-3f;
    if (alpha <= ALPHA_EPS) {
        glm::vec3 wo = glm::normalize(woW);
        float cosNo = glm::dot(nW, wo);
        bool entering = (cosNo > 0.0f);
        float ei = entering ? etaI : etaT;
        float et = entering ? etaT : etaI;
        float eta = ei / et;

        glm::vec3 N = entering ? nW : -nW;

        // snell
        float k = 1.0f - eta * eta * (1.0f - cosNo * cosNo);
        if (k > 0.0f) {
            glm::vec3 wi = glm::normalize(eta * (-wo) + (eta * cosNo - sqrtf(k)) * N);
            wiW = wi;

            float cosNI = fmaxf(1e-6f, fabsf(glm::dot(nW, wi)));
            float weight = eta * eta;
            f = glm::vec3(weight / cosNI);
            pdfLobe = 1.0f;
            return true;
        }
        else {
            glm::vec3 wi = glm::reflect(-wo, N);
            wiW = glm::normalize(wi);
            float cosNI = fmaxf(1e-6f, fabsf(glm::dot(nW, wiW)));
            f = glm::vec3(1.0f / cosNI);
            pdfLobe = 1.0f;
            return true;
        }
    }

    // to local
    glm::vec3 wo;
    worldToLocal(nW, glm::normalize(woW), wo);
    if (wo.z == 0.f) { pdfLobe = 0.f; f = glm::vec3(0); return false; }

    // sample facet normal with heitz vndf
    glm::vec3 m = SampleWhVNDF(wo, alpha, u1, u2);
    if (m.z < 0.f) m = -m;

    // refract through the facet
    glm::vec3 wi;
    if (!RefractThroughFacet(wo, m, etaI, etaT, wi)) { pdfLobe = 0.f; f = glm::vec3(0); return false; }
    if (wi.z * wo.z >= 0.f) { pdfLobe = 0.f; f = glm::vec3(0); return false; } // must change hemisphere

    // evaluate btdf
    glm::vec3 ft = MicrofacetBTDF(wo, wi, alpha, etaI, etaT);
    float pdfT = PdfMicrofacetBTDF(wo, wi, m, alpha, etaI, etaT);
    if (!(pdfT > 0.f)) { pdfLobe = 0.f; f = glm::vec3(0); return false; }

    // back to world
    glm::vec3 wiWorld = localToWorld(nW, wi);
    wiW = wiWorld;
    f = ft;
    pdfLobe = pdfT;
    return true;
}

// sample base color: uses texture if present, else constant
DEVICE_INLINE glm::vec3 SampleBaseColor(const Material* mat, const cpt::Texture2D* textures, const glm::vec2& uv) {
    if (HasBaseColorTex(mat)) {
        cudaTextureObject_t texObj = textures[mat->baseColorTex].texObj;
        float4 texel = tex2D<float4>(texObj, uv.x, uv.y);
        return MakeVec3(texel);
    }
    return mat->baseColor;
}

// sample roughness: uses texture if present, else constant
DEVICE_INLINE glm::vec3 SampleRoughness(const Material* mat, const cpt::Texture2D* textures, const glm::vec2& uv) {
    if (HasMetallicRoughnessTex(mat)) {
        glm::vec3 roughness(1.f);
        cudaTextureObject_t texObj = textures[mat->metallicRoughnessTex].texObj;
        float4 texel = tex2D<float4>(texObj, uv.x, uv.y);
        roughness *= texel.y;
        return roughness;
    }
    return glm::vec3(mat->roughness);
}

// sample metallic: uses texture if present, else constant
DEVICE_INLINE glm::vec3 SampleMetallic(const Material* mat, const cpt::Texture2D* textures, const glm::vec2& uv) {
    if (HasMetallicRoughnessTex(mat)) {
        glm::vec3 metallic(1.f);
        cudaTextureObject_t texObj = textures[mat->metallicRoughnessTex].texObj;
        float4 texel = tex2D<float4>(texObj, uv.x, uv.y);
        metallic *= texel.z;
        return metallic;
    }
    return glm::vec3(mat->metallic);
}

DEVICE_INLINE glm::vec3 SampleNormalTS(cudaTextureObject_t tex, glm::vec2 uv)
{
    float4 t = tex2D<float4>(tex, uv.x, uv.y);
    float nx = 2.0f * t.x - 1.0f;
    float ny = 2.0f * t.y - 1.0f;
    float nz = sqrtf(fmaxf(0.0f, 1.0f - nx * nx - ny * ny));
    return glm::vec3(nx, ny, nz);
}

DEVICE_INLINE
glm::vec3 ApplyNormalMap(const Material* mat,
    const cpt::Texture2D* textures,
    const ShadeableIntersection* isect)
{
    if (!HasNormalTex(mat))
        return glm::normalize(isect->surfaceNormal);

    const cudaTextureObject_t tex = textures[mat->normalTex].texObj;
    const glm::vec3 n_ts = SampleNormalTS(tex, isect->uv);

    const glm::vec3 T = isect->tangentWs;
    const glm::vec3 B = isect->bitangentWs;
    const glm::vec3 N = glm::normalize(isect->surfaceNormal);

    glm::vec3 n_ws = glm::normalize(n_ts.x * T + n_ts.y * B + n_ts.z * N);

    if (glm::dot(n_ws, N) < 0.0f) n_ws = -n_ws;
    return n_ws;
}

// lambert brdf and pdf helpers
DEVICE_INLINE glm::vec3 LambertBRDF(const glm::vec3& albedo) { return albedo * INV_PI; }
DEVICE_INLINE float     LambertPDF(float NdotL) { return fmaxf(NdotL, 0.f) * INV_PI; }

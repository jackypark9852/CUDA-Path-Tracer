#pragma once

#include <cuda_runtime.h>
#include <string>
#include <vector>

#include "glm/glm.hpp"
#include "utilities.h"
 
#define BACKGROUND_COLOR (glm::vec3(0.0f))

enum class MaterialType
{
    INVALID = 0, 
    EMISSIVE,
    DIFFUSE,
    SPECULAR,
    TRANSMISSIVE,
    PBR,
    ENVMAP,
    COUNT 
};

struct Material
{
    MaterialType type;
    
    glm::vec3 baseColor;
    float ior;
    glm::vec3 emissiveFactor; 
    float emissiveStrength; 
    float metallic;
    float roughness;
    float transmission; 

    // Texture indices (optional; -1 if none)
    int baseColorTex;
    int metallicRoughnessTex;
    int normalTex;
    float normalScale; 
    int emissiveTex;

    // For perfectly specular case
    struct
    {
        float exponent;
        glm::vec3 color;
    } specular; 
};

inline Material MakeDefaultMaterial() {
    Material m{};
    m.type = MaterialType::PBR;
    m.baseColor = glm::vec3(0.063f, 0.024f, 0.624f);
    m.metallic = 0.0f;
    m.roughness = 1.0f;
    m.emissiveFactor = glm::vec3(0.f);
    m.emissiveTex = -1;
    m.emissiveStrength = 0.f;
    m.ior = 1.5f;
    m.transmission = 0.f;
    m.baseColorTex = -1;
    m.metallicRoughnessTex = -1;
    m.normalTex = -1;
    m.normalScale = 1.f;
    m.specular.exponent = 0.f;
    m.specular.color = glm::vec3(1.f);
    return m;
}

enum GeomType
{
    SPHERE,
    CUBE
};

struct Ray
{
    glm::vec3 origin;
    glm::vec3 direction;
};

struct Geom
{
    enum GeomType type;
    int materialid; 
    MaterialType materialType; 
    glm::vec3 translation;
    glm::vec3 rotation;
    glm::vec3 scale;
    glm::mat4 transform;
    glm::mat4 inverseTransform;
    glm::mat4 invTranspose;
};

struct Camera
{
    glm::ivec2 resolution;
    glm::vec3 position;
    glm::vec3 lookAt;

    float lensRadius;
    float focusDist;

    glm::vec3 view;
    glm::vec3 up;
    glm::vec3 right;
    glm::vec2 fov;
    glm::vec2 pixelLength;
    glm::vec3 horizontal;
    glm::vec3 vertical;
    glm::vec3 lowerLeftCorner;

    void UpdateDerived(float fovy)
    {
        view = glm::normalize(lookAt - position);
        right = glm::normalize(glm::cross(view, glm::vec3(0, 1, 0)));
        up = glm::normalize(glm::cross(right, view));

        // image plane
        const float aspect = float(resolution.x) / float(resolution.y);
        const float theta = glm::radians(fovy);
        const float half_h = tanf(0.5f * theta);
        const float half_w = aspect * half_h;

        horizontal = 2.0f * half_w * focusDist * right;
        vertical = 2.0f * half_h * focusDist * up;
        lowerLeftCorner = position
            - 0.5f * horizontal
            - 0.5f * vertical
            + focusDist * view;

        //calculate fov based on resolution
        float yscaled = tan(fovy * (PI / 180.0f));
        float xscaled = (yscaled * resolution.x) / resolution.y;
        float fovx = (atan(xscaled) * 180) / PI;

        fov = glm::vec2(fovx, fovy);
        pixelLength = glm::vec2(2 * xscaled / (float)resolution.x,
            2 * yscaled / (float)resolution.y);
    }
};

struct RenderState
{
    Camera camera;

    unsigned int beautyIters; 
    unsigned int aovIters; 

    int traceDepth;
    std::vector<glm::vec3> beauty;
    std::vector<glm::vec3> normal; 
    std::vector<glm::vec3> albedo;
    std::vector<glm::vec3> roughness;
    std::vector<glm::vec3> metallic; 
    std::string imageName;
};

struct PathSegment
{
    Ray ray;
    glm::vec3 color;
    int pixelIndex;
    int remainingBounces;
    bool shouldTerminate; 
};

// Use with a corresponding PathSegment to do:
// 1) color contribution computation
// 2) BSDF evaluation: generate a new ray
struct ShadeableIntersection
{
    MaterialType materialType;
    int materialId;
    float t;
    glm::vec3 surfaceNormal;
    glm::vec3 tangentWs; 
    glm::vec3 bitangentWs;
    glm::vec2 uv; 
};

struct HitState {
    bool        hit;
    float       tMin;
    glm::vec3   nWs;
    glm::vec2   uv;
    int         hitGeomIdx;
    int         hitMeshIdx;
    int         hitPrimIdx;
};

struct BSDFSample {
    glm::vec3 incomingDir;
    glm::vec3 bsdfValue;   // f(wo, wi)
    float     pdf;
    bool      isDelta;
};

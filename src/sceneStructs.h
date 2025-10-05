#pragma once

#include <cuda_runtime.h>
#include "glm/glm.hpp"
#include <string>
#include <vector>

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
    glm::vec3 view;
    glm::vec3 up;
    glm::vec3 right;
    glm::vec2 fov;
    glm::vec2 pixelLength;
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

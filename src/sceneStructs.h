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
    METALLIC, 
    DIELECTRIC, 
    ENVMAP,
    COUNT 
};

struct Material
{
    MaterialType type;
    
    glm::vec3 baseColor;
    float ior;
    float emittance;
    float metallic;
    float roughness;

    // Transmission
    float     transmission; // 0..1
    float     thickness; // meters
    glm::vec3 attenuationColor; // Beer–Lambert
    float     attenuationDistance; // meters

    // Texture indices (optional; -1 if none)
    int baseColorTex;
    int metallicRoughnessTex;
    int normalTex;
    int emissiveTex;

    // For perfectly specular case
    struct
    {
        float exponent;
        glm::vec3 color;
    } specular; 
};

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
    unsigned int iterations;
    int traceDepth;
    std::vector<glm::vec3> image;
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
  float t;
  glm::vec3 surfaceNormal;
  MaterialType materialType; 
  int materialId;
};

#pragma once

// this module loads geometry-only gltf/glb into your host structs
// for now, supports triangles only (mode == 4)
// positions (required), normals (optional), indices (optional)
// produces meshes (with primitives) and instances (node transforms)
// no materials/textures yet; materialIndex is populated if desired later

#include <string>
#include <vector>
#include <glm/glm.hpp>

struct HostGLTFPrimitive {
    std::vector<glm::vec3> positions;
    std::vector<glm::vec3> normals;
    std::vector<uint32_t>  indices;         // 0 if non-indexed
    int materialIndex = -1;
    glm::vec3 aabbMin = glm::vec3(FLT_MAX);
    glm::vec3 aabbMax = glm::vec3(-FLT_MAX);
};

struct HostGLTFMesh {
    std::string name;
    std::vector<HostGLTFPrimitive> primitives;
};

struct HostGLTFInstance {
    int meshIndex = -1;         // index into the array of meshes
    glm::mat4 world = glm::mat4(1.0f);
    int nodeIndex = -1;         // just for debug
};

struct HostGLTFScene {
    std::vector<HostGLTFMesh>     meshes;
    std::vector<HostGLTFInstance> instances;
};

// high-level api
bool LoadGltfFile(const std::string& path, HostGLTFScene& outScene, std::string* err = nullptr);

// utility: apply an extra root transform to all instances
void ApplyRootTransform(HostGLTFScene& scene, const glm::mat4& root);

// utility: compose trs (in radians for rotation)
glm::mat4 ComposeTrs(const glm::vec3& t, const glm::vec3& rRadians, const glm::vec3& s);

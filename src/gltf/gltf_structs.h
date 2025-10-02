#pragma once

#include <glm/common.hpp>
#include <glm/matrix.hpp>
#include <string>
#include <vector>

struct HostGltfPrimitive {
    std::vector<glm::vec3> positions;
    std::vector<glm::vec3> normals;
    std::vector<glm::vec2> uvs;
    std::vector<uint32_t>  indices;         // 0 if non-indexed
    int materialIndex = -1;
    glm::vec3 aabbMin = glm::vec3(FLT_MAX);
    glm::vec3 aabbMax = glm::vec3(-FLT_MAX);
};

struct HostGltfMesh {
    std::string name;
    std::vector<HostGltfPrimitive> primitives;
};

struct HostGltfInstance {
    int meshIndex = -1;         // index into the array of meshes
    glm::mat4 world = glm::mat4(1.0f);
    int nodeIndex = -1;         // just for debug
};

struct HostGltfScene {
    std::vector<HostGltfMesh>     meshes;
    std::vector<HostGltfInstance> instances;
};

// per-primitive pod view (device-friendly)
struct DevicePrimitive {
    const glm::vec3* positions;
    const glm::vec3* normals;           // can be nullptr
    const glm::vec2* uvs;               // can be nullptr
    const uint32_t* indices;            // can be nullptr (for non-indexed)
    int numVertices;                    // size of positions (and normals if present)
    int numIndices;                     // 3*n for triangles, 0 if non-indexed
    int materialIndex;                  // one material per primitive

    glm::vec3 aabbMin, aabbMax;
};

// primitives are grouped by mesh
struct DeviceMesh {
    const DevicePrimitive* primitives; // device ptr to contiguous array
    int numPrimitives;
};

// instance references a mesh
struct DeviceInstance {
    int meshIndex;                    // index into device array of DeviceMesh
    glm::mat4 world;                  // node world transform
};

struct DeviceGltfScene {
    DeviceInstance* instances = nullptr;
    int             numInstances = 0;

    DeviceMesh* meshes = nullptr;
    int             numMeshes = 0;

    // ownership lists for cleanup
    std::vector<void*> ownedVertexBuffers;     // per-primitive: positions/normals/indices
    std::vector<void*> ownedPrimArrays;        // per-mesh: array<DevicePrimitive>
};

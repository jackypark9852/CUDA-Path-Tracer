#pragma once

// this module loads geometry-only gltf/glb into your host structs
// for now, supports triangles only (mode == 4)
// positions (required), normals (optional), indices (optional)
// produces meshes (with primitives) and instances (node transforms)
// no materials/textures yet; materialIndex is populated if desired later

#include <string>
#include <vector>
#include <glm/glm.hpp>

#include "gltf/gltf_structs.h"
#include "sceneStructs.h"
#include "texture.h"

bool LoadGltfFile(
    const std::string&              gltfPath,          // relative path to .gltf
    HostGltfScene&                  outScene,          // host-side data with mesh and instances array
    std::vector<Material>&          outMaterials,      // appends new materials to this array
    std::vector<std::string>&       outMaterialNames,     // for debugging 
    std::vector<cpt::Texture2D>&    outTextures,       // appends new textures to this array
    std::string*                    err = nullptr      // for error-handling
);

// Sends host-side data to device 
DeviceGltfScene UploadGltfData(
    const std::vector<HostGltfInstance>&    hostInstances,
    const std::vector<HostGltfMesh>&        hostMeshes
);

void FreeDeviceGltfScene(DeviceGltfScene gltfScene); 

// utility: apply an extra root transform to all instances
void ApplyRootTransform(HostGltfScene& scene, const glm::mat4& root);

// utility: compose trs (in radians for rotation)
glm::mat4 ComposeTrs(const glm::vec3& t, const glm::vec3& rRadians, const glm::vec3& s);

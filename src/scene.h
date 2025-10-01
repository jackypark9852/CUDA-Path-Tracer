#pragma once

#include "gltf_loader.h"
#include "sceneStructs.h"
#include "texture.h"\
#include <vector>

class Scene
{
private:
    void loadFromJSON(const std::string& jsonName);
public:
    Scene(std::string filename);

    std::vector<Geom> geoms;
    std::vector<Material> materials;
    std::vector<std::string> materialNames;
    std::vector<cpt::Texture2D> textures; // excluding environemnt map
    std::vector<HostGltfMesh> meshes;
    std::vector<HostGltfInstance> instances;
    cpt::Texture2D envMap; 
    RenderState state;
};

#pragma once

#include "sceneStructs.h"
#include "texture.h"
#include <vector>

class Scene
{
private:
    void loadFromJSON(const std::string& jsonName);
public:
    Scene(std::string filename);

    std::vector<Geom> geoms;
    std::vector<Material> materials;
    std::vector<cpt::Texture2D> textures; 
    RenderState state;
};

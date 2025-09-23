#include "scene.h"

#include "utilities.h"

#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtx/string_cast.hpp>
#include "json.hpp"

#include <fstream>
#include <filesystem>
#include <iostream>
#include <string>
#include <unordered_map>
#include "texture.h"

using namespace std;
using json = nlohmann::json;
namespace fs = std::filesystem;

static fs::path resolvePathRelativeTo(const fs::path& baseFile, const std::string& p) {
    fs::path candidate = fs::path(p);
    if (!candidate.is_absolute()) {
        candidate = baseFile.parent_path() / candidate; 
    }
    candidate = candidate.lexically_normal();
    return candidate;
}

Scene::Scene(string filename)
{
    cout << "Reading scene from " << filename << " ..." << endl;
    cout << " " << endl;
    auto ext = filename.substr(filename.find_last_of('.'));
    if (ext == ".json")
    {
        loadFromJSON(filename);
        return;
    }
    else
    {
        cout << "Couldn't read from " << filename << endl;
        exit(-1);
    }
}

void Scene::loadFromJSON(const std::string& jsonName)
{
    std::ifstream f(jsonName);
    json data = json::parse(f);
    const auto& materialsData = data["Materials"];
    std::unordered_map<std::string, uint32_t> MatNameToID;
    for (const auto& item : materialsData.items())
    {
        const auto& name = item.key();
        const auto& p = item.value();
        Material newMaterial{};
        if (p["TYPE"] == "Diffuse")
        {
            newMaterial.type = MaterialType::DIFFUSE;
            const auto& col = p["RGB"];
            newMaterial.baseColor = glm::vec3(col[0], col[1], col[2]);
        } 
        else if (p["TYPE"] == "Emitting")
        {
            newMaterial.type = MaterialType::EMISSIVE;
            const auto& col = p["RGB"];
            newMaterial.baseColor = glm::vec3(col[0], col[1], col[2]);
            newMaterial.emittance = p["EMITTANCE"];
        }
        else if (p["TYPE"] == "Specular")
        {
            newMaterial.type = MaterialType::SPECULAR; 
            const auto& col = p["RGB"];
            newMaterial.baseColor = glm::vec3(col[0], col[1], col[2]);
        }
        else if (p["TYPE"] == "Transmissive") 
        {
            newMaterial.type = MaterialType::TRANSMISSIVE; 
            const auto& col = p["RGB"];
            newMaterial.baseColor = glm::vec3(col[0], col[1], col[2]);
            newMaterial.ior = p["IOR"];
        }
        else if (p["TYPE"] == "Pbr")
        {
            newMaterial.type = MaterialType::PBR; 
            const auto& col = p["RGB"]; 
            newMaterial.baseColor = glm::vec3(col[0], col[1], col[2]);
            newMaterial.ior = p["IOR"]; 
            newMaterial.emittance = p["EMITTANCE"]; 
            newMaterial.metallic = p["METALLIC"];
            newMaterial.roughness = p["ROUGHNESS"];
            newMaterial.transmission = p["TRANSMISSIVE"];
        }
        MatNameToID[name] = materials.size();
        materials.emplace_back(newMaterial);
    }
    const auto& objectsData = data["Objects"];
    for (const auto& p : objectsData)
    {
        const auto& type = p["TYPE"];
        Geom newGeom;
        if (type == "cube")
        {
            newGeom.type = CUBE;
        }
        else
        {
            newGeom.type = SPHERE;
        }
        newGeom.materialid = MatNameToID[p["MATERIAL"]];
        newGeom.materialType = materials.at(newGeom.materialid).type;
        const auto& trans = p["TRANS"];
        const auto& rotat = p["ROTAT"];
        const auto& scale = p["SCALE"];
        newGeom.translation = glm::vec3(trans[0], trans[1], trans[2]);
        newGeom.rotation = glm::vec3(rotat[0], rotat[1], rotat[2]);
        newGeom.scale = glm::vec3(scale[0], scale[1], scale[2]);
        newGeom.transform = utilityCore::buildTransformationMatrix(
            newGeom.translation, newGeom.rotation, newGeom.scale);
        newGeom.inverseTransform = glm::inverse(newGeom.transform);
        newGeom.invTranspose = glm::inverseTranspose(newGeom.transform);

        geoms.push_back(newGeom);
    }

    // load environment map 
    const auto& envMapData = data["EnvironmentMap"]; 
    std::string envRel = envMapData["Path"].get<std::string>();

    fs::path envPath = resolvePathRelativeTo(jsonName, envRel);
    if (!fs::exists(envPath)) {
        throw std::runtime_error("EnvironmentMap not found at: " + envPath.string());
    }

    cpt::TextureDesc hdrEnvDesc;
    hdrEnvDesc.pixelFormat = cpt::PixelFormat::RGBA32F;
    hdrEnvDesc.colorSpace = cpt::ColorSpace::Linear;
    hdrEnvDesc.sampler.addressU = cudaAddressModeWrap;
    hdrEnvDesc.sampler.addressV = cudaAddressModeClamp;
    hdrEnvDesc.sampler.filter = cudaFilterModeLinear;
    hdrEnvDesc.sampler.normalizedCoords = true;
    hdrEnvDesc.sampler.readMode = cudaReadModeElementType;
    cpt::createTextureFromFile(envMap, envPath, hdrEnvDesc); 

    // load camera settings
    const auto& cameraData = data["Camera"];
    Camera& camera = state.camera;
    RenderState& state = this->state;
    camera.resolution.x = cameraData["RES"][0];
    camera.resolution.y = cameraData["RES"][1];
    float fovy = cameraData["FOVY"];
    state.iterations = cameraData["ITERATIONS"];
    state.traceDepth = cameraData["DEPTH"];
    state.imageName = cameraData["FILE"];
    const auto& pos = cameraData["EYE"];
    const auto& lookat = cameraData["LOOKAT"];
    const auto& up = cameraData["UP"];
    camera.position = glm::vec3(pos[0], pos[1], pos[2]);
    camera.lookAt = glm::vec3(lookat[0], lookat[1], lookat[2]);
    camera.up = glm::vec3(up[0], up[1], up[2]);

    //calculate fov based on resolution
    float yscaled = tan(fovy * (PI / 180));
    float xscaled = (yscaled * camera.resolution.x) / camera.resolution.y;
    float fovx = (atan(xscaled) * 180) / PI;
    camera.fov = glm::vec2(fovx, fovy);

    camera.right = glm::normalize(glm::cross(camera.view, camera.up));
    camera.pixelLength = glm::vec2(2 * xscaled / (float)camera.resolution.x,
        2 * yscaled / (float)camera.resolution.y);

    camera.view = glm::normalize(camera.lookAt - camera.position);

    //set up render camera stuff
    int arraylen = camera.resolution.x * camera.resolution.y;
    state.image.resize(arraylen);
    std::fill(state.image.begin(), state.image.end(), glm::vec3());
}

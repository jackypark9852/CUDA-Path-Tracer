#include "scene.h"

#include <fstream>
#include <filesystem>
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtx/string_cast.hpp>
#include <iostream>
#include <string>
#include <unordered_map>

#include "json.hpp"
#include "texture.h"
#include "utilities.h"

using namespace std;
using json = nlohmann::json;
namespace fs = std::filesystem;

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
    std::unordered_map<std::string, uint32_t> MatNameToID;

    std::string defaultMaterialName = "Default Material"; 
    materials.push_back(MakeDefaultMaterial());
    materialNames.push_back(defaultMaterialName); 

    if (data.contains("Materials")) {
        const auto& materialsData = data["Materials"];
        for (const auto& item : materialsData.items())
        {
            const auto& name = item.key();
            const auto& p = item.value();
            Material newMaterial = MakeDefaultMaterial(); 
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
                newMaterial.emissiveStrength = p["EMITTANCE"];
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
                newMaterial.emissiveStrength = p["EMITTANCE"];
                newMaterial.metallic = p["METALLIC"];
                newMaterial.roughness = p["ROUGHNESS"];
                newMaterial.transmission = p["TRANSMISSIVE"];
            }
            materialNames.push_back(name); 
            MatNameToID[name] = materials.size();
            materials.emplace_back(newMaterial);
        }
    }
    if (data.contains("Objects")) {
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
            newGeom.transform = UtilityCore::buildTransformationMatrix(
                newGeom.translation, newGeom.rotation, newGeom.scale);
            newGeom.inverseTransform = glm::inverse(newGeom.transform);
            newGeom.invTranspose = glm::inverseTranspose(newGeom.transform);

            geoms.push_back(newGeom);
        }
    }

    if (data.contains("Imports")) {
        for (const auto& imp : data["Imports"]) {
            std::string rel = imp["PATH"].get<std::string>();
            fs::path path = UtilityCore::ResolvePathRelativeTo(jsonName, rel);

            HostGltfScene gltfScene;
            std::string err;
            if (!LoadGltfFile(path.string(), gltfScene, materials, materialNames,textures, &err)) {
                std::cerr << "gltf load failed: " << err << std::endl;
            }

            glm::mat4 root = glm::mat4(1.f);
            const auto& t = imp["TRANS"];
            const auto& r = imp["ROTAT"];
            const auto& s = imp["SCALE"];
            glm::vec3 translation = glm::vec3(t[0], t[1], t[2]);
            glm::vec3 rotation = glm::vec3(r[0], r[1], r[2]);
            glm::vec3 scale = glm::vec3(s[0], s[1], s[2]);
            glm::mat4 transform = UtilityCore::buildTransformationMatrix(translation, rotation, scale);
            ApplyRootTransform(gltfScene, transform);
                
            int meshesSize = meshes.size(); 
            meshes.insert(meshes.end(), gltfScene.meshes.begin(), gltfScene.meshes.end()); 
            for (HostGltfInstance& instance : gltfScene.instances) {
                instance.meshIndex += meshesSize;
                instances.push_back(instance); 
            }
        }
    }
 
    // load environment map 
    if (data.contains("EnvironmentMap")) {
        const auto& envMapData = data["EnvironmentMap"];
        std::string envRel = envMapData["Path"].get<std::string>();

        fs::path envPath = UtilityCore::ResolvePathRelativeTo(jsonName, envRel);
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
    }

    // load camera settings
    RenderState& state = this->state;
    Camera& camera = state.camera;
        
    const auto& cameraData = data["Camera"];
    const auto& pos = cameraData["EYE"];
    const auto& lookat = cameraData["LOOKAT"];
    float fovy = cameraData["FOVY"];

    camera.position = glm::vec3(pos[0], pos[1], pos[2]);
    camera.lookAt = glm::vec3(lookat[0], lookat[1], lookat[2]);
    camera.resolution.x = cameraData["RES"][0];
    camera.resolution.y = cameraData["RES"][1];
    camera.focusDist = (cameraData.contains("FOCUS_DIST")) ? 
        cameraData["FOCUS_DIST"] : 0.6 * glm::length(camera.lookAt - camera.position);
    camera.lensRadius = (cameraData.contains("LENS_RADIUS"))? 
        cameraData["LENS_RADIUS"] : 0.005f * camera.focusDist;
    
    camera.UpdateDerived(fovy); 
    
    //set up render camera stuff 
    state.beautyIters = (cameraData.contains("ITERATIONS")) ? cameraData["ITERATIONS"] : DEFAULT_ITERS;
    state.aovIters = (cameraData.contains("AOV_ITERATIONS")) ? cameraData["AOV_ITERATIONS"] : DEFAULT_ITERS;
    state.traceDepth = cameraData["DEPTH"];
    state.imageName = cameraData["FILE"];

    int arraylen = camera.resolution.x * camera.resolution.y;
    state.beauty.resize(arraylen);
    state.normal.resize(arraylen); 
    state.albedo.resize(arraylen); 
    state.roughness.resize(arraylen); 
    state.metallic.resize(arraylen); 
    std::fill(state.beauty.begin(), state.beauty.end(), glm::vec3());
}

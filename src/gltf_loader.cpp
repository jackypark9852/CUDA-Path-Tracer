#include <cfloat>
#include <cmath>
#include <cuda_runtime.h>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/quaternion.hpp>

#define TINYGLTF_IMPLEMENTATION
#define TINYGLTF_NO_STB_IMAGE_WRITE
#include "tiny_gltf.h"

#include "gltf_loader.h"
#include "utilities.h"


// internal helpers
namespace {

    // read float vec3 accessor into dst (handles stride/offset). returns false on type mismatch
    bool ReadAccessorVec3(const tinygltf::Model& model, int accessorIndex, std::vector<glm::vec3>& dst);

    // read indices accessor into uint32 dst (supports u16/u32). returns false on unsupported types
    bool ReadAccessorIndicesU32(const tinygltf::Model& model, int accessorIndex, std::vector<uint32_t>& dst);

    // compute aabb of positions
    void BuildAabb(const std::vector<glm::vec3>& pos, glm::vec3& outMin, glm::vec3& outMax);

    // fetch node local matrix (trs or matrix). rotation in gltf is quaternion, which is converted to mat4 directly.
    glm::mat4 NodeLocalMatrix(const tinygltf::Node& n);

    // dfs to compute world matrices, filling worldPerNode
    void ComputeWorldTransforms(const tinygltf::Model& model, int nodeIndex, const glm::mat4& parent,
        std::vector<glm::mat4>& worldPerNode);

    // create HostGLTFMesh from a gltf mesh
    bool ConvertMesh(
        const tinygltf::Model&  model,
        const tinygltf::Mesh&   src,
        HostGltfMesh&           dst,
        int                     matOffset,
        int                     defaultMatIdx,
        std::string*            err
    );

    enum TexUse : int { UseNone = 0, UseSRGB = 1 << 0, UseLinear = 1 << 1 };

    inline int ChooseUseForTexture(const tinygltf::Model& model, int texIndex);

    inline cpt::ColorSpace PickColorSpaceFromUse(int use);

    inline void FillDefaultSampler(cpt::SamplerDesc& s); 

    inline void ConfigureSamplerForFormat(cpt::TextureDesc& d);

    inline void ParseOneMaterial(const tinygltf::Material& gm,
        int texOffset,
        const tinygltf::Model& model,
        Material& outM);
} // namespace


bool LoadGltfFile(
    const std::string&              gltfPath, 
    HostGltfScene&                  outScene, 
    std::vector<Material>&          outMaterials, 
    std::vector<std::string>&       outMaterialNames,
    std::vector<cpt::Texture2D>&    outTextures,
    std::string*                    err)
{
    tinygltf::Model model;
    tinygltf::TinyGLTF loader;
    std::string warn, error;

    bool ok = false;
    if (gltfPath.size() >= 4 && (gltfPath.substr(gltfPath.size() - 4) == ".glb" || gltfPath.substr(gltfPath.size() - 4) == ".GLB")) {
        ok = loader.LoadBinaryFromFile(&model, &error, &warn, gltfPath);
    }
    else {
        ok = loader.LoadASCIIFromFile(&model, &error, &warn, gltfPath);
    }
    if (!warn.empty() && err) *err += ("gltf warn: " + warn + "\n");
    if (!ok) {
        if (err) *err += ("gltf load failed: " + error + "\n");
        return false;
    }

    const int texOffset = (int)outTextures.size();
    const int matOffset = (int)outMaterials.size();

    // infer texture usage for colorspace
    std::vector<int> texUse(model.textures.size(), UseNone);
    for (size_t ti = 0; ti < model.textures.size(); ++ti) {
        texUse[ti] = ChooseUseForTexture(model, (int)ti);
    }

    // upload textures in order
    outTextures.reserve(outTextures.size() + model.textures.size());

    for (size_t ti = 0; ti < model.textures.size(); ++ti) {
        const auto& gt = model.textures[ti];
        int srcIdx = gt.source;
        if (srcIdx < 0 || srcIdx >= (int)model.images.size()) {
            if (err) *err += "texture missing image source idx " + std::to_string(srcIdx) + "\n";
            // push a dummy texture so indices stay aligned
            outTextures.emplace_back();
            continue;
        }

        const auto& img = model.images[srcIdx];
        if (img.uri.empty()) {
            if (err) *err += "embedded images not supported (bufferView)\n";
            outTextures.emplace_back();
            continue;
        }

        cpt::TextureDesc desc{};
        desc.pixelFormat = cpt::PixelFormat::RGBA8; // jpg/png path
        desc.colorSpace = PickColorSpaceFromUse(texUse[ti]);
        ConfigureSamplerForFormat(desc); 

        std::filesystem::path imgPath = UtilityCore::ResolvePathRelativeTo(gltfPath, img.uri);
        cpt::Texture2D tex{};
        if (!cpt::createTextureFromFile(tex, imgPath, desc, 0)) {
            if (err) *err += "failed to upload texture: " + imgPath.string() + "\n";
            outTextures.emplace_back();
        }
        else {
            outTextures.push_back(std::move(tex));
        }
    }

    // parse materials
    outMaterials.reserve(outMaterials.size() + model.materials.size() + 1);
    for (const auto& gm : model.materials) {
        Material m = MakeDefaultMaterial();
        ParseOneMaterial(gm, texOffset, model, m);
        outMaterials.push_back(m);
        outMaterialNames.push_back(gm.name); 
    }

    // default material for missing indices
    const int defaultMatIdx = 0; 

    // meshes
    outScene.meshes.clear();
    outScene.instances.clear();
    outScene.meshes.reserve(model.meshes.size());

    for (const auto& m : model.meshes) {
        HostGltfMesh dst;
        if (!ConvertMesh(model, m, dst, matOffset, defaultMatIdx, err)) {
            continue;
        }
        outScene.meshes.push_back(std::move(dst));
    }

    // world transforms
    std::vector<glm::mat4> worldPerNode(model.nodes.size(), glm::mat4(1.0f));
    const tinygltf::Scene& gscene = (model.scenes.empty() ? tinygltf::Scene() : model.scenes[model.defaultScene > -1 ? model.defaultScene : 0]);
    for (int root : gscene.nodes) {
        ComputeWorldTransforms(model, root, glm::mat4(1.0f), worldPerNode);
    }

    // instances
    for (size_t ni = 0; ni < model.nodes.size(); ++ni) {
        const auto& n = model.nodes[ni];
        if (n.mesh < 0) continue;
        if (n.mesh >= (int)outScene.meshes.size()) continue;
        HostGltfInstance inst;
        inst.meshIndex = n.mesh;
        inst.world = worldPerNode[ni];
        inst.nodeIndex = (int)ni;
        outScene.instances.push_back(inst);
    }

    return true;
}

DeviceGltfScene UploadGltfData(
    const std::vector<HostGltfInstance>&        hostInstances, 
    const std::vector<HostGltfMesh>&            hostMeshes)
{
    const size_t nInst = hostInstances.size();
    const size_t nMesh = hostMeshes.size();

    DeviceGltfScene ds; 
    ds.numInstances = nInst; 
    ds.numMeshes = nMesh; 

    // allocate top-level arrays (not tracked in outGltfAllocs)
    checkCUDAError("UploadGltfData Start; Seek errors before here");
    cudaMalloc((void**)&ds.instances, sizeof(DeviceInstance) * nInst);
    checkCUDAError("cudaMalloc outDeviceInstances");
    cudaMalloc((void**)&ds.meshes, sizeof(DeviceMesh) * nMesh);
    checkCUDAError("cudaMalloc outDeviceMeshes");

    // upload instances
    for (size_t i = 0; i < nInst; ++i) {
        DeviceInstance di{};
        di.meshIndex = hostInstances[i].meshIndex;
        di.world = hostInstances[i].world;
        cudaMemcpy(ds.instances + i, &di, sizeof(DeviceInstance), cudaMemcpyHostToDevice);
        checkCUDAError("cudaMemcpy device instance");
    }

    // per-mesh primitives
    for (size_t mi = 0; mi < nMesh; ++mi) {
        const auto& hmesh = hostMeshes[mi];
        const int nprim = static_cast<int>(hmesh.primitives.size());

        DevicePrimitive* dprims = nullptr;
        if (nprim > 0) {
            cudaMalloc((void**)&dprims, sizeof(DevicePrimitive) * nprim);
            checkCUDAError("cudaMalloc device primitives array");
            ds.ownedPrimArrays.push_back(dprims);
        }

        DeviceMesh dmesh{};
        dmesh.primitives = dprims;
        dmesh.numPrimitives = nprim;

        cudaMemcpy(ds.meshes + mi, &dmesh, sizeof(DeviceMesh), cudaMemcpyHostToDevice);
        checkCUDAError("cudaMemcpy device mesh header");

        // per-primitive attribute buffers
        for (int pi = 0; pi < nprim; ++pi) {
            const auto& hp = hmesh.primitives[pi];

            DevicePrimitive dp{};
            dp.positions = nullptr;
            dp.normals = nullptr;
            dp.uvs = nullptr;
            dp.indices = nullptr;
            dp.numVertices = static_cast<int>(hp.positions.size());
            dp.numIndices = static_cast<int>(hp.indices.size());
            dp.materialIndex = hp.materialIndex;
            dp.aabbMin = hp.aabbMin;
            dp.aabbMax = hp.aabbMax;

            // positions (required)
            if (dp.numVertices == 0) {
                // positions required; write empty dp and continue to avoid crash
                cudaMemcpy(dprims + pi, &dp, sizeof(DevicePrimitive), cudaMemcpyHostToDevice);
                checkCUDAError("cudaMemcpy empty primitive");
                continue;
            }
            {
                glm::vec3* dpos = nullptr;
                size_t bytes = sizeof(glm::vec3) * hp.positions.size();
                cudaMalloc((void**)&dpos, bytes);
                checkCUDAError("cudaMalloc positions");
                ds.ownedVertexBuffers.push_back(dpos);
                cudaMemcpy(dpos, hp.positions.data(), bytes, cudaMemcpyHostToDevice);
                checkCUDAError("cudaMemcpy positions");
                dp.positions = dpos;
            }

            // normals (optional)
            if (!hp.normals.empty()) {
                glm::vec3* dnor = nullptr;
                size_t bytes = sizeof(glm::vec3) * hp.normals.size();
                cudaMalloc((void**)&dnor, bytes);
                checkCUDAError("cudaMalloc normals");
                ds.ownedVertexBuffers.push_back(dnor);
                cudaMemcpy(dnor, hp.normals.data(), bytes, cudaMemcpyHostToDevice);
                checkCUDAError("cudaMemcpy normals");
                dp.normals = dnor;
            }

            // uvs (optional)
            if (!hp.uvs.empty()) {
                glm::vec2* duv = nullptr;
                size_t bytes = sizeof(glm::vec2) * hp.indices.size();
                cudaMalloc((void**)&duv, bytes);
                checkCUDAError("cudaMalloc indices");
                ds.ownedVertexBuffers.push_back(duv);
                cudaMemcpy(duv, hp.uvs.data(), bytes, cudaMemcpyHostToDevice);
                checkCUDAError("cudaMemcpy indices");
                dp.uvs = duv;
            }

            // indices (optional)
            if (!hp.indices.empty()) {
                uint32_t* didx = nullptr;
                size_t bytes = sizeof(uint32_t) * hp.indices.size();
                cudaMalloc((void**)&didx, bytes);
                checkCUDAError("cudaMalloc indices");
                ds.ownedVertexBuffers.push_back(didx);
                cudaMemcpy(didx, hp.indices.data(), bytes, cudaMemcpyHostToDevice);
                checkCUDAError("cudaMemcpy indices");
                dp.indices = didx;
            }

            // write primitive header
            cudaMemcpy(dprims + pi, &dp, sizeof(DevicePrimitive), cudaMemcpyHostToDevice);
            checkCUDAError("cudaMemcpy primitive header");
        }
    }

    return ds; 
}

void FreeDeviceGltfScene(DeviceGltfScene gltfScene)
{
    for (void* primArray : gltfScene.ownedPrimArrays) {
        cudaFree(primArray); 
        checkCUDAError("FreeGltfScene cudaFree primArray"); 
    }

    for (void* vertBuffer : gltfScene.ownedVertexBuffers) {
        cudaFree(vertBuffer); 
        checkCUDAError("FreeGltfScene cudaFree vertBuffer"); 
    }

    cudaFree(gltfScene.meshes); 
    cudaFree(gltfScene.instances); 
}

void ApplyRootTransform(HostGltfScene& scene, const glm::mat4& root)
{
    for (auto& inst : scene.instances) {
        inst.world = root * inst.world;
    }
}

glm::mat4 ComposeTrs(const glm::vec3& t, const glm::vec3& rRadians, const glm::vec3& s) {
    // note: glm uses column-major; order scale*rot*trans typical is translate * rotZ * rotY * rotX * scale
    glm::mat4 T = glm::translate(glm::mat4(1.f), t);
    glm::mat4 Rx = glm::rotate(glm::mat4(1.f), rRadians.x, glm::vec3(1, 0, 0));
    glm::mat4 Ry = glm::rotate(glm::mat4(1.f), rRadians.y, glm::vec3(0, 1, 0));
    glm::mat4 Rz = glm::rotate(glm::mat4(1.f), rRadians.z, glm::vec3(0, 0, 1));
    glm::mat4 S = glm::scale(glm::mat4(1.f), s);
    return T * Rz * Ry * Rx * S;
}

// === internals ===

namespace {
    bool ReadAccessorVec3(const tinygltf::Model& model, int accessorIndex, std::vector<glm::vec3>& dst)
    {
        if (accessorIndex < 0) return false;
        const auto& acc = model.accessors[accessorIndex];
        if (acc.type != TINYGLTF_TYPE_VEC3 || acc.componentType != TINYGLTF_COMPONENT_TYPE_FLOAT) return false;

        const auto& bv = model.bufferViews[acc.bufferView];
        const auto& buf = model.buffers[bv.buffer];

        const size_t stride = bv.byteStride ? bv.byteStride : sizeof(float) * 3;
        const uint8_t* base = buf.data.data() + bv.byteOffset + acc.byteOffset;

        dst.resize(acc.count);
        for (size_t i = 0; i < acc.count; ++i) {
            const float* f = reinterpret_cast<const float*>(base + i * stride);
            dst[i] = glm::vec3(f[0], f[1], f[2]);
        }
        return true;
    }

    bool ReadAccessorIndicesU32(const tinygltf::Model& model, int accessorIndex, std::vector<uint32_t>& dst)
    {
        if (accessorIndex < 0) { dst.clear(); return true; } // non-indexed ok
        const auto& acc = model.accessors[accessorIndex];
        const auto& bv = model.bufferViews[acc.bufferView];
        const auto& buf = model.buffers[bv.buffer];

        const size_t stride = bv.byteStride ? bv.byteStride : tinygltf::GetComponentSizeInBytes(acc.componentType);
        const uint8_t* base = buf.data.data() + bv.byteOffset + acc.byteOffset;

        dst.resize(acc.count);
        switch (acc.componentType) {
        case TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT: {
            for (size_t i = 0; i < acc.count; ++i) dst[i] = reinterpret_cast<const uint16_t*>(base + i * stride)[0];
            return true;
        }
        case TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT: {
            for (size_t i = 0; i < acc.count; ++i) dst[i] = reinterpret_cast<const uint32_t*>(base + i * stride)[0];
            return true;
        }
        case TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE: {
            for (size_t i = 0; i < acc.count; ++i) dst[i] = reinterpret_cast<const uint8_t*>(base + i * stride)[0];
            return true;
        }
        default: return false;
        }
    }

    static bool ReadAccessorVec2(const tinygltf::Model& model, int accessorIndex, std::vector<glm::vec2>& dst)
    {
        if (accessorIndex < 0) { dst.clear(); return true; }
        const auto& acc = model.accessors[accessorIndex];
        if (acc.type != TINYGLTF_TYPE_VEC2) return false;
        const auto& bv = model.bufferViews[acc.bufferView];
        const auto& buf = model.buffers[bv.buffer];

        const size_t elemSize = tinygltf::GetComponentSizeInBytes(acc.componentType) * 2;
        const size_t stride = bv.byteStride ? bv.byteStride : elemSize;
        const uint8_t* base = buf.data.data() + bv.byteOffset + acc.byteOffset;

        dst.resize(acc.count);

        switch (acc.componentType) {
        case TINYGLTF_COMPONENT_TYPE_FLOAT: {
            for (size_t i = 0; i < acc.count; ++i) {
                const float* f = reinterpret_cast<const float*>(base + i * stride);
                dst[i] = glm::vec2(f[0], f[1]);
            }
            return true;
        }
        case TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT: {
            const float scale = acc.normalized ? (1.0f / 65535.0f) : 1.0f;
            for (size_t i = 0; i < acc.count; ++i) {
                const uint16_t* v = reinterpret_cast<const uint16_t*>(base + i * stride);
                dst[i] = glm::vec2(v[0] * scale, v[1] * scale);
            }
            return true;
        }
        case TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE: {
            const float scale = acc.normalized ? (1.0f / 255.0f) : 1.0f;
            for (size_t i = 0; i < acc.count; ++i) {
                const uint8_t* v = reinterpret_cast<const uint8_t*>(base + i * stride);
                dst[i] = glm::vec2(v[0] * scale, v[1] * scale);
            }
            return true;
        }
        default:
            return false;
        }
    }

    void BuildAabb(const std::vector<glm::vec3>& pos, glm::vec3& outMin, glm::vec3& outMax)
    {
        glm::vec3 mn(FLT_MAX), mx(-FLT_MAX);
        for (auto& p : pos) {
            mn = glm::min(mn, p);
            mx = glm::max(mx, p);
        }
        outMin = mn; outMax = mx;
    }

    glm::mat4 NodeLocalMatrix(const tinygltf::Node& n)
    {
        if (!n.matrix.empty()) {
            glm::mat4 M(1.f);
            // gltf is column-major; data stored row-major array length 16
            for (int c = 0; c < 4; ++c)
                for (int r = 0; r < 4; ++r)
                    M[c][r] = (float)n.matrix[c * 4 + r];
            return M;
        }

        glm::vec3 t(0.f), s(1.f);
        if (!n.translation.empty()) t = glm::vec3((float)n.translation[0], (float)n.translation[1], (float)n.translation[2]);
        if (!n.scale.empty())       s = glm::vec3((float)n.scale[0], (float)n.scale[1], (float)n.scale[2]);
        glm::mat4 T = glm::translate(glm::mat4(1.f), t);

        glm::mat4 R(1.f);
        if (!n.rotation.empty()) {
            glm::quat q((float)n.rotation[3], (float)n.rotation[0], (float)n.rotation[1], (float)n.rotation[2]);
            R = glm::mat4_cast(q);
        }
        glm::mat4 S = glm::scale(glm::mat4(1.f), s);
        return T * R * S;
    }

    void ComputeWorldTransforms(const tinygltf::Model& model, int nodeIndex, const glm::mat4& parent,
        std::vector<glm::mat4>& worldPerNode)
    {
        const auto& n = model.nodes[nodeIndex];
        glm::mat4 local = NodeLocalMatrix(n);
        glm::mat4 world = parent * local;
        worldPerNode[nodeIndex] = world;
        for (int child : n.children) {
            ComputeWorldTransforms(model, child, world, worldPerNode);
        }
    }

    bool ConvertMesh(
        const tinygltf::Model&  model, 
        const tinygltf::Mesh&   src, 
        HostGltfMesh&           dst, 
        int                     matOffset,
        int                     defaultMatIdx,
        std::string*            err)
    {
        dst.name = src.name;
        dst.primitives.clear();
        dst.primitives.reserve(src.primitives.size());

        for (const auto& prim : src.primitives) {
            int mode = prim.mode == -1 ? TINYGLTF_MODE_TRIANGLES : prim.mode;
            if (mode != TINYGLTF_MODE_TRIANGLES) {
                if (err) *err += "skip non-triangles primitive in mesh " + src.name + "\n";
                continue;
            }

            HostGltfPrimitive hp;

            auto itPos = prim.attributes.find("POSITION");
            if (itPos == prim.attributes.end()) { if (err) *err += "missing POSITION\n"; continue; }
            if (!ReadAccessorVec3(model, itPos->second, hp.positions)) { if (err) *err += "bad POSITION accessor\n"; continue; }

            auto itNrm = prim.attributes.find("NORMAL");
            if (itNrm != prim.attributes.end()) {
                ReadAccessorVec3(model, itNrm->second, hp.normals);
            }

            auto itUv0 = prim.attributes.find("TEXCOORD_0");
            if (itUv0 != prim.attributes.end()) {
                if (!ReadAccessorVec2(model, itUv0->second, hp.uvs)) {
                    if (err) *err += "bad TEXCOORD_0 accessor\n";
                }
            }

            if (!ReadAccessorIndicesU32(model, prim.indices, hp.indices)) {
                if (err) *err += "unsupported index type\n";
                continue;
            }

            // generate sequential indices if non-indexed
            if (hp.indices.empty()) {
                const size_t n = hp.positions.size();
                hp.indices.resize(n);
                for (uint32_t i = 0; i < (uint32_t)n; ++i) hp.indices[i] = i;

                // guard: triangle list requires multiple of 3
                if (n % 3 != 0 && err) {
                    *err += "non-indexed vertex count not multiple of 3 in mesh " + src.name + "\n";
                }
            }

            BuildAabb(hp.positions, hp.aabbMin, hp.aabbMax);

            hp.materialIndex = (prim.material >= 0) ? (matOffset + prim.material) : defaultMatIdx;

            dst.primitives.push_back(std::move(hp));
        }

        return true;
    }

    inline int ChooseUseForTexture(const tinygltf::Model& model, int texIndex) {
        int use = UseNone;
        if (texIndex < 0 || texIndex >= (int)model.textures.size()) return use;
        // scan materials for references
        for (const auto& gm : model.materials) {
            const auto& pmr = gm.pbrMetallicRoughness;

            if (pmr.baseColorTexture.index == texIndex) use |= UseSRGB;
            if (gm.emissiveTexture.index == texIndex)   use |= UseSRGB;
            if (pmr.metallicRoughnessTexture.index == texIndex) use |= UseLinear;
            if (gm.normalTexture.index == texIndex) use |= UseLinear;
            if (gm.occlusionTexture.index == texIndex) use |= UseLinear;

            // extension textures (optional, can be added later)
        }
        return use;
    }

    inline cpt::ColorSpace PickColorSpaceFromUse(int use) {
        // if any srgb usage exists, prefer srgb, else linear
        return (use & UseSRGB) ? cpt::ColorSpace::sRGB : cpt::ColorSpace::Linear;
    }

    inline void FillDefaultSampler(cpt::SamplerDesc& s) {
        s.addressU = cudaAddressModeWrap;
        s.addressV = cudaAddressModeWrap;
        s.filter = cudaFilterModeLinear;
        s.normalizedCoords = true;
        s.readMode = cudaReadModeElementType;
    }

    inline void ConfigureSamplerForFormat(cpt::TextureDesc& d) {
        FillDefaultSampler(d.sampler); 
        switch (d.pixelFormat) {
        case cpt::PixelFormat::RGBA8:      // any unorm/uint8 format
            d.sampler.readMode = cudaReadModeNormalizedFloat;
            d.sampler.filter = cudaFilterModeLinear;
            break;
        case cpt::PixelFormat::RGBA32F:    // float formats
            d.sampler.readMode = cudaReadModeElementType;
            d.sampler.filter = cudaFilterModeLinear;
            break;
        default:                           // safe fallback
            d.sampler.readMode = cudaReadModeNormalizedFloat;
            d.sampler.filter = cudaFilterModeLinear;
            break;
        }
    }

    inline void ParseOneMaterial(const tinygltf::Material& gm,
        int texOffset,
        const tinygltf::Model& model,
        Material& outM)
    {
        outM.type = MaterialType::PBR;
        outM.baseColor = glm::vec3(1.f);
        outM.metallic = 1.f;
        outM.roughness = 1.f;
        outM.emissiveColor = glm::vec3(0.f);
        outM.emissiveTex = -1;
        outM.emissiveStrength = 1.f; // emissiveStrength default
        outM.ior = 1.5f;
        outM.transmission = 0.f;
        outM.baseColorTex = -1;
        outM.metallicRoughnessTex = -1;
        outM.normalTex = -1;
        outM.normalScale = 1.f;

        const auto& pmr = gm.pbrMetallicRoughness;

        if (pmr.baseColorFactor.size() >= 3) {
            outM.baseColor = glm::vec3(
                (float)pmr.baseColorFactor[0],
                (float)pmr.baseColorFactor[1],
                (float)pmr.baseColorFactor[2]);
        }
        if (pmr.metallicFactor >= 0.0)  outM.metallic = (float)pmr.metallicFactor;
        if (pmr.roughnessFactor >= 0.0) outM.roughness = (float)pmr.roughnessFactor;

        if (pmr.baseColorTexture.index >= 0)
            outM.baseColorTex = texOffset + pmr.baseColorTexture.index;

        if (pmr.metallicRoughnessTexture.index >= 0)
            outM.metallicRoughnessTex = texOffset + pmr.metallicRoughnessTexture.index;

        if (!gm.emissiveFactor.empty()) {
            outM.emissiveColor = glm::vec3(
                (float)gm.emissiveFactor[0],
                (float)gm.emissiveFactor[1],
                (float)gm.emissiveFactor[2]);
        }
        if (gm.emissiveTexture.index >= 0)
            outM.emissiveTex = texOffset + gm.emissiveTexture.index;

        auto itES = gm.extensions.find("KHR_materials_emissive_strength");
        if (itES != gm.extensions.end()) {
            const auto& ext = itES->second;
            auto val = ext.Get("emissiveStrength");
            if (val.IsNumber()) outM.emissiveStrength = (float)val.Get<double>();
        }

        if (gm.normalTexture.index >= 0) {
            outM.normalTex = texOffset + gm.normalTexture.index;
            outM.normalScale = (float)gm.normalTexture.scale;
        }

        auto itIOR = gm.extensions.find("KHR_materials_ior");
        if (itIOR != gm.extensions.end()) {
            auto val = itIOR->second.Get("ior");
            if (val.IsNumber()) outM.ior = (float)val.Get<double>();
        }

        auto itTR = gm.extensions.find("KHR_materials_transmission");
        if (itTR != gm.extensions.end()) {
            auto val = itTR->second.Get("transmissionFactor");
            if (val.IsNumber()) outM.transmission = (float)val.Get<double>();
        }
    }
} // namespace

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
        const tinygltf::Model&          model, 
        const tinygltf::Mesh&           src, 
        HostGltfMesh&                   dst, 
        std::vector<Material>&          mats, 
        std::vector<cpt::Texture2D>&    texs,
        std::string*                    err
    );



} // namespace

// public api

bool LoadGltfFile(
    const std::string&              path, 
    HostGltfScene&                  outScene, 
    std::vector<Material>&          outMaterials, 
    std::vector<cpt::Texture2D>&    outTextures,
    std::string*                    err)
{
    // parse gltf/glb
    tinygltf::Model model;
    tinygltf::TinyGLTF loader;
    std::string warn, error;

    bool ok = false;
    if (path.size() >= 4 && (path.substr(path.size() - 4) == ".glb" || path.substr(path.size() - 4) == ".GLB")) {
        ok = loader.LoadBinaryFromFile(&model, &error, &warn, path);
    }
    else {
        ok = loader.LoadASCIIFromFile(&model, &error, &warn, path);
    }
    if (!warn.empty() && err) *err += ("gltf warn: " + warn + "\n");
    if (!ok) {
        if (err) *err += ("gltf load failed: " + error + "\n");
        return false;
    }

    // convert meshes
    outScene.meshes.clear();
    outScene.instances.clear();
    outScene.meshes.reserve(model.meshes.size());

    for (const auto& m : model.meshes) {
        HostGltfMesh dst;
        if (!ConvertMesh(model, m, dst, outMaterials, outTextures, err)) {
            // skip invalid meshes but continue
            continue;
        }
        outScene.meshes.push_back(std::move(dst));
    }

    // compute world transforms for all nodes
    std::vector<glm::mat4> worldPerNode(model.nodes.size(), glm::mat4(1.0f));
    const tinygltf::Scene& scene = (model.scenes.empty() ? tinygltf::Scene() : model.scenes[model.defaultScene > -1 ? model.defaultScene : 0]);

    for (int root : scene.nodes) {
        ComputeWorldTransforms(model, root, glm::mat4(1.0f), worldPerNode);
    }

    // collect instances: one per node that references a mesh
    for (size_t ni = 0; ni < model.nodes.size(); ++ni) {
        const auto& n = model.nodes[ni];
        if (n.mesh < 0) continue;
        if (n.mesh >= (int)outScene.meshes.size()) continue; // guard
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
        const tinygltf::Model&          model, 
        const tinygltf::Mesh&           src, 
        HostGltfMesh&                   dst, 
        std::vector<Material>&          mats, 
        std::vector<cpt::Texture2D>&    texs, 
        std::string*                    err)
    {
        dst.name = src.name;
        dst.primitives.clear();
        dst.primitives.reserve(src.primitives.size());

        for (const auto& prim : src.primitives) {
            // triangles only
            int mode = prim.mode == -1 ? TINYGLTF_MODE_TRIANGLES : prim.mode;
            if (mode != TINYGLTF_MODE_TRIANGLES) {
                if (err) *err += "skip non-triangles primitive in mesh " + src.name + "\n";
                continue;
            }

            HostGltfPrimitive hp;

            // positions (required)
            auto itPos = prim.attributes.find("POSITION");
            if (itPos == prim.attributes.end()) { if (err) *err += "missing POSITION\n"; continue; }
            if (!ReadAccessorVec3(model, itPos->second, hp.positions)) { if (err) *err += "bad POSITION accessor\n"; continue; }

            // normals (optional)
            auto itNrm = prim.attributes.find("NORMAL");
            if (itNrm != prim.attributes.end()) {
                ReadAccessorVec3(model, itNrm->second, hp.normals); // ignore failure silently
            }

            // indices (optional)
            if (!ReadAccessorIndicesU32(model, prim.indices, hp.indices)) {
                if (err) *err += "unsupported index type\n";
                continue;
            }

            // if non-indexed, encourage a later step to generate 0..N-1 or leave empty
            BuildAabb(hp.positions, hp.aabbMin, hp.aabbMax);

            // material index (optional; keep for future)
            hp.materialIndex = prim.material;

            dst.primitives.push_back(std::move(hp));
        }

        return true;
    }

} // namespace

// this .cu’s own header first
#include "pathtrace.h"

// c/c++ std
#include <cstdio>
#include <cmath>

// cuda runtime (prefer runtime api over driver api)
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <thrust/execution_policy.h>
#include <thrust/partition.h>
#include <thrust/random.h>

#include "gltf/gltf_structs.h"
#include "gltf_loader.h"
#include "intersections.h"
#include "interactions.h"
#include "settings.h"
#include "sceneStructs.h"
#include "scene.h"
#include "shading/shading_common.cuh"
#include "shading/shading_kernels.cuh"
#include "texture.h"
#include "utilities.h"

//Kernel that writes the image to the OpenGL PBO directly.
__global__ void sendImageToPBO(uchar4* pbo, glm::ivec2 resolution, int iter, glm::vec3* image)
{
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;
    if (x >= resolution.x || y >= resolution.y) return;

    int index = x + y * resolution.x;

    // average then display-map
    glm::vec3 lin  = image[index] / float(iter);
    glm::vec3 disp = to_display_v3(lin);

    uchar4 out;
    out.x = to_u8(disp.x);
    out.y = to_u8(disp.y);
    out.z = to_u8(disp.z);
    out.w = 255; // opaque

    pbo[index] = out;
}

static Scene* hst_scene = NULL;
static GuiDataContainer* guiData = NULL;
static glm::vec3* dev_beauty = NULL;
static glm::vec3* dev_normal = NULL; 
static glm::vec3* dev_albedo = NULL; 
static glm::vec3* dev_roughness = NULL;
static glm::vec3* dev_metallic = NULL;
static Geom* dev_geoms = NULL;
static Material* dev_materials = NULL;
static cpt::Texture2D* dev_textures = NULL; 
static cpt::Texture2D* envMap = NULL; 
static PathSegment* dev_paths = NULL;
static ShadeableIntersection* dev_intersections = NULL;
static int* dev_startIdx = NULL; 
static int* dev_endIdx = NULL; 
static int* hst_startIdx = NULL; 
static int* hst_endIdx = NULL;

// device buffer for gltf mesh data 
DeviceGltfScene gltfScene; 

struct is_active {
    __host__ __device__
        bool operator()(const PathSegment& seg) {
        return !seg.shouldTerminate; 
    }
};


void InitDataContainer(GuiDataContainer* imGuiData)
{
    guiData = imGuiData;
}

void pathtraceInit(Scene* scene)
{
    hst_scene = scene;

    const Camera& cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;

    cudaMalloc(&dev_beauty, pixelcount * sizeof(glm::vec3));
    cudaMemset(dev_beauty, 0, pixelcount * sizeof(glm::vec3));

    cudaMalloc(&dev_normal, pixelcount * sizeof(glm::vec3));
    cudaMemset(dev_normal, 0, pixelcount * sizeof(glm::vec3));

    cudaMalloc(&dev_albedo, pixelcount * sizeof(glm::vec3));
    cudaMemset(dev_albedo, 0, pixelcount * sizeof(glm::vec3));

    cudaMalloc(&dev_roughness, pixelcount * sizeof(glm::vec3));
    cudaMemset(dev_roughness, 0, pixelcount * sizeof(glm::vec3));

    cudaMalloc(&dev_metallic, pixelcount * sizeof(glm::vec3));
    cudaMemset(dev_metallic, 0, pixelcount * sizeof(glm::vec3));

    cudaMalloc(&dev_paths, pixelcount * sizeof(PathSegment));

    cudaMalloc(&dev_geoms, scene->geoms.size() * sizeof(Geom));
    cudaMemcpy(dev_geoms, scene->geoms.data(), scene->geoms.size() * sizeof(Geom), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_materials, scene->materials.size() * sizeof(Material));
    cudaMemcpy(dev_materials, scene->materials.data(), scene->materials.size() * sizeof(Material), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_textures, scene->textures.size() * sizeof(cpt::Texture2D));
    cudaMemcpy(dev_textures, scene->textures.data(), scene->textures.size() * sizeof(cpt::Texture2D), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_intersections, pixelcount * sizeof(ShadeableIntersection));
    cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

    cudaMalloc(&dev_startIdx, static_cast<int>(MaterialType::COUNT) * sizeof(int));
    cudaMalloc(&dev_endIdx, static_cast<int>(MaterialType::COUNT) * sizeof(int));

    hst_startIdx = new int[static_cast<int>(MaterialType::COUNT)];
    hst_endIdx = new int[static_cast<int>(MaterialType::COUNT)];

    envMap = &(scene->envMap); 

    // send gltf data from scene to device
    gltfScene = UploadGltfData(scene->instances, scene->meshes);
}

void pathtraceFree()
{
    cudaFree(dev_beauty);  // no-op if dev_image is null
    cudaFree(dev_normal); 
    cudaFree(dev_albedo); 
    cudaFree(dev_roughness);
    cudaFree(dev_metallic);
    cudaFree(dev_paths);
    cudaFree(dev_geoms);
    cudaFree(dev_materials);
    cudaFree(dev_textures); 
    cudaFree(dev_intersections);

    cudaFree(dev_startIdx); 
    cudaFree(dev_endIdx); 

    FreeDeviceGltfScene(gltfScene); 

    delete[] hst_startIdx; 
    delete[] hst_endIdx; 
}

__global__ void generateRayFromCamera(Camera cam, int iter, int traceDepth, PathSegment* pathSegments)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= cam.resolution.x || y >= cam.resolution.y) return;

    int index = x + y * cam.resolution.x;
    PathSegment& seg = pathSegments[index];
    seg.color = glm::vec3(1.0f);
    seg.pixelIndex = index;
    seg.remainingBounces = traceDepth;
    seg.shouldTerminate = false;

    thrust::default_random_engine rng = MakeSeededRandomEngine(iter, index, traceDepth);
    thrust::uniform_real_distribution<float> u01(0.0f, 1.0f);

    // jitter for stochastic aa
    float jitterX = (u01(rng) - 0.5f);
    float jitterY = (u01(rng) - 0.5f);
    float px = float(x) + jitterX;
    float py = float(y) + jitterY;

    // compute focal point
    glm::vec3 d_pinhole = glm::normalize(
        cam.view
        - cam.right * cam.pixelLength.x * (px - 0.5f * float(cam.resolution.x))
        - cam.up * cam.pixelLength.y * (py - 0.5f * float(cam.resolution.y))
    );

    float denom = glm::dot(d_pinhole, cam.view);
    float t_focus = cam.focusDist / fmaxf(fabsf(denom), 1e-6f);
    glm::vec3 focalPoint = cam.position + d_pinhole * t_focus;

    // simualte random ray starting point on thin lens
    glm::vec3 origin = cam.position;
    if (cam.lensRadius > 0.0f) {
        glm::vec3 d = cam.lensRadius * RandomInUnitDisk(rng);
        glm::vec3 offset = d.x * cam.right + d.y * cam.up;
        origin += offset;
    }

    seg.ray.origin = origin;
    seg.ray.direction = glm::normalize(focalPoint - origin);
}


__device__ inline glm::vec3 XformPoint(const glm::mat4& m, const glm::vec3& p) {
    glm::vec4 r = m * glm::vec4(p, 1.f);
    return glm::vec3(r) / r.w;
}

__device__ inline glm::vec3 XformVector(const glm::mat4& m, const glm::vec3& v) {
    return glm::vec3(m * glm::vec4(v, 0.f));
}

__device__ inline glm::vec3 InterpVec3(const glm::vec3* inVec3, int i0, int i1, int i2, float u, float v) {
    glm::vec3 n0 = inVec3[i0], n1 = inVec3[i1], n2 = inVec3[i2];
    float w = 1.f - u - v;
    return w * n0 + u * n1 + v * n2;
}

__device__ inline glm::vec2 InterpVec2(const glm::vec2* inVec2, int i0, int i1, int i2, float u, float v) {
    glm::vec2 n0 = inVec2[i0], n1 = inVec2[i1], n2 = inVec2[i2];
    float w = 1.f - u - v;
    return w * n0 + u * n1 + v * n2;
}

__device__ inline void InitHitState(HitState& s) {
    s.hit = false;
    s.tMin = FLT_MAX;
    s.nWs = glm::vec3(0.f);
    s.uv = glm::vec2(0.f);
    s.hitGeomIdx = -1;
    s.hitMeshIdx = -1;
    s.hitPrimIdx = -1;
}

__device__ inline void MaybeUpdateBest(float tWorld,
    const glm::vec3& nWs,
    const glm::vec2& uv,
    int meshIdx, int primIdx,
    HitState& s)
{
    if (tWorld < s.tMin) {
        s.hit = true;
        s.tMin = tWorld;
        s.nWs = nWs;
        s.uv = uv;
        s.hitGeomIdx = -1;
        s.hitMeshIdx = meshIdx;
        s.hitPrimIdx = primIdx;
    }
}

__device__ inline void TraverseSceneNaiveGltf(
    const DeviceGltfScene& scene,
    const PathSegment& seg,
    HitState& hs)
{
    for (int i = 0; i < scene.numInstances; ++i) {
        const DeviceInstance inst = scene.instances[i];
        if (inst.meshIndex < 0 || inst.meshIndex >= scene.numMeshes) continue;

        const DeviceMesh dmesh = scene.meshes[inst.meshIndex];
        if (dmesh.numPrimitives <= 0) continue;

        const glm::mat4& M = inst.world;
        const glm::mat4& Mi = inst.invWorld;
        const glm::mat4& Nxf = inst.normalXf; 

        const glm::vec3 ro_os = XformPoint(Mi, seg.ray.origin);
        const glm::vec3 rd_os = glm::normalize(XformVector(Mi, seg.ray.direction));

        for (int pi = 0; pi < dmesh.numPrimitives; ++pi) {
            const DevicePrimitive dp = dmesh.primitives[pi];
            if (dp.numVertices <= 0) continue;

            const int triCount = dp.numIndices / 3;

            for (int ti = 0; ti < triCount; ++ti) {
                int b = ti * 3;
                int i0 = dp.indices[b + 0];
                int  i1 = dp.indices[b + 1];
                int i2 = dp.indices[b + 2];

                const glm::vec3 v0 = dp.positions[i0];
                const glm::vec3 v1 = dp.positions[i1];
                const glm::vec3 v2 = dp.positions[i2];

                float t;
                glm::vec3 bary;
                if (!RayTriangleIntersect(v0, v1, v2, ro_os, rd_os, t, bary)) continue;

                const glm::vec3 p_os = ro_os + t * rd_os;
                const glm::vec3 p_ws = XformPoint(M, p_os);
                const float tWorld = glm::length(p_ws - seg.ray.origin);
                if (tWorld >= hs.tMin) continue;

                const float u = bary.y;
                const float v = bary.z;

                glm::vec3 n_os = dp.normals ?
                    glm::normalize(InterpVec3(dp.normals, i0, i1, i2, u, v)) :
                    glm::normalize(glm::cross(v1 - v0, v2 - v0));
                glm::vec3 n_ws = glm::normalize(XformVector(Nxf, n_os));
                if (glm::dot(n_ws, seg.ray.direction) > 0.f) n_ws = -n_ws;

                const glm::vec2 uv = dp.uvs ? InterpVec2(dp.uvs, i0, i1, i2, u, v) : glm::vec2(0.f);

                MaybeUpdateBest(tWorld, n_ws, uv, inst.meshIndex, pi, hs);
            }
        }
    }
}

inline __host__ __device__ bool isMiss(float t) { return t < 0.0f; }

struct candidate_hit {
    bool hit;
    float t_ws;     // world t for pruning
    float t_os;     // object t for per-instance pruning
    int inst_idx;
    int mesh_idx;
    int prim_idx;
    int i0, i1, i2; // triangle vertex indices
    float u, v;     // barycentric
};

__device__ __forceinline__ void IniteCandidate(candidate_hit& c) {
    c.hit = false; c.t_ws = FLT_MAX; c.t_os = FLT_MAX;
    c.inst_idx = -1; c.mesh_idx = -1; c.prim_idx = -1;
    c.i0 = c.i1 = c.i2 = -1; c.u = c.v = 0.f;
}

// choose near/far children
__device__ __forceinline__ void OrderChildren(float& t0, float& t1,
    uint32_t& i0, uint32_t& i1,
    const BvhNode*& n0, const BvhNode*& n1) {
    float k0 = isMiss(t0) ? FLT_MAX : t0;
    float k1 = isMiss(t1) ? FLT_MAX : t1;
    if ((k1 + EPSILON) < k0) {
        dswap(t0, t1);
        dswap(i0, i1);
        dswap(n0, n1);
    }
}

// compute final shading normal and uv once
__device__ __forceinline__ void compute_final_shading(
    const DeviceGltfScene& scene,
    const DeviceInstance& inst,
    const DevicePrimitive& dp,
    const candidate_hit& c,
    glm::vec3& n_ws_out,
    glm::vec2& uv_out)
{
    const glm::mat4& Nxf = inst.normalXf;
    const glm::vec3 v0 = dp.positions[c.i0];
    const glm::vec3 v1 = dp.positions[c.i1];
    const glm::vec3 v2 = dp.positions[c.i2];
    const float u = c.u, v = c.v;

    glm::vec3 ng_os = glm::normalize(glm::cross(v1 - v0, v2 - v0));
    glm::vec3 ns_os = dp.normals ?
        glm::normalize(InterpVec3(dp.normals, c.i0, c.i1, c.i2, u, v)) :
        ng_os;

    glm::vec3 ng_ws = glm::normalize(XformVector(Nxf, ng_os));
    glm::vec3 ns_ws = glm::normalize(XformVector(Nxf, ns_os));

    if (glm::determinant(glm::mat3(inst.world)) < 0.0f) ng_ws = -ng_ws;
    if (glm::dot(ns_ws, ng_ws) < 0.0f) ns_ws = -ns_ws;

    n_ws_out = ns_ws;
    uv_out = dp.uvs ? InterpVec2(dp.uvs, c.i0, c.i1, c.i2, u, v) : glm::vec2(0.f);
}

// main traversal
__device__ inline void TraverseSceneBvh(
    const DeviceGltfScene& scene,
    const PathSegment& seg,
    HitState& hs)
{
    int stack[64]; // keep as-is; can be moved to shared later

    candidate_hit best;
    IniteCandidate(best);

    for (int ii = 0; ii < scene.numInstances; ++ii) {
        const DeviceInstance inst = scene.instances[ii];
        if (inst.meshIndex < 0 || inst.meshIndex >= scene.numMeshes) continue;

        const DeviceMesh dmesh = scene.meshes[inst.meshIndex];
        if (dmesh.numPrimitives <= 0) continue;

        // os ray
        Ray objRay;
        objRay.origin = XformPoint(inst.invWorld, seg.ray.origin);
        const glm::vec3 d_os_unnorm = XformVector(inst.invWorld, seg.ray.direction);
        const float dir_os_len = glm::length(d_os_unnorm);
        if (dir_os_len <= 0.0f) continue;
        objRay.direction = d_os_unnorm / dir_os_len;

        // start with a conservative os upper bound derived from current global ws best
        float bestT_os = (hs.tMin < FLT_MAX * 0.5f) ? (hs.tMin * dir_os_len) : FLT_MAX;

        for (int pi = 0; pi < dmesh.numPrimitives; ++pi) {
            const DevicePrimitive& dp = dmesh.primitives[pi];
            if (dp.numVertices <= 0 || dp.bvhNodes == nullptr) continue;

            uint32_t sp = 0;
            const BvhNode* node = &dp.bvhNodes[0];

            while (true) {
                if (node->triCount > 0) {
                    const uint32_t start = node->leftFirst;
                    const uint32_t end = start + node->triCount;

                    for (uint32_t triIdx = start; triIdx < end; ++triIdx) {
                        const int b = static_cast<int>(triIdx) * 3;
                        const int i0 = dp.indices[b + 0];
                        const int i1 = dp.indices[b + 1];
                        const int i2 = dp.indices[b + 2];

                        const glm::vec3 v0 = dp.positions[i0];
                        const glm::vec3 v1 = dp.positions[i1];
                        const glm::vec3 v2 = dp.positions[i2];

                        float t_os; glm::vec3 bary;
                        if (!RayTriangleIntersect(v0, v1, v2, objRay.origin, objRay.direction, t_os, bary)) continue;
                        if (t_os >= bestT_os) continue;

                        const glm::vec3 p_os = objRay.origin + t_os * objRay.direction;
                        const glm::vec3 p_ws = XformPoint(inst.world, p_os);
                        const float t_ws = glm::length(p_ws - seg.ray.origin);
                        if (t_ws >= hs.tMin) continue;

                        // record only minimal data; update global ws best for pruning
                        hs.tMin = t_ws;
                        bestT_os = t_os;
                        best.hit = true;
                        best.t_ws = t_ws;
                        best.t_os = t_os;
                        best.inst_idx = ii;
                        best.mesh_idx = inst.meshIndex;
                        best.prim_idx = pi;
                        best.i0 = i0; best.i1 = i1; best.i2 = i2;
                        best.u = bary.y; best.v = bary.z;
                    }

                    if (sp == 0) break;
                    node = &dp.bvhNodes[stack[--sp]];
                }
                else {
                    uint32_t child0Idx = node->leftFirst + 0;
                    uint32_t child1Idx = node->leftFirst + 1;
                    const BvhNode* child0 = &dp.bvhNodes[child0Idx];
                    const BvhNode* child1 = &dp.bvhNodes[child1Idx];

                    float t0 = RayAABBIntersection(child0->aabb, objRay, bestT_os);
                    float t1 = RayAABBIntersection(child1->aabb, objRay, bestT_os);

                    if (isMiss(t0) & isMiss(t1)) {
                        if (sp == 0) break;
                        node = &dp.bvhNodes[stack[--sp]];
                        continue;
                    }

                    OrderChildren(t0, t1, child0Idx, child1Idx, child0, child1);

                    node = child0;
                    if (!isMiss(t1) && t1 < bestT_os) stack[sp++] = child1Idx;
                }
            }
        }
    }

    if (!best.hit) {
        hs.hit = false;
        return;
    }

    // compute final shading once for the winner and write hs
    const DeviceInstance inst = scene.instances[best.inst_idx];
    const DeviceMesh     dmesh = scene.meshes[best.mesh_idx];
    const DevicePrimitive dp = dmesh.primitives[best.prim_idx];

    glm::vec3 n_ws; glm::vec2 uv;
    compute_final_shading(scene, inst, dp, best, n_ws, uv);

    hs.hit = true;
    hs.nWs = n_ws;
    hs.uv = uv;
    hs.hitGeomIdx = -1;
    hs.hitMeshIdx = best.mesh_idx;
    hs.hitPrimIdx = best.prim_idx;
}



__global__ void ComputeIntersections(
    int numPaths,
    PathSegment* pathSegments,
    Geom* geoms, int geomsSize,
    DeviceGltfScene deviceGltfScene,
    ShadeableIntersection* intersections)
{
    int path_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (path_index >= numPaths) return;

    PathSegment& seg = pathSegments[path_index];

    HitState hs; InitHitState(hs);

    {
        float t;
        glm::vec3 tmpI, tmpN;
        bool outside;
        for (int i = 0; i < geomsSize; ++i) {
            Geom& g = geoms[i];
            if (g.type == CUBE)        t = boxIntersectionTest(g, seg.ray, tmpI, tmpN, outside);
            else if (g.type == SPHERE) t = sphereIntersectionTest(g, seg.ray, tmpI, tmpN, outside);
            else                       t = -1.f;

            if (t > 0.f && t < hs.tMin) {
                hs.hit = true;
                hs.tMin = t;
                hs.nWs = tmpN;
                hs.hitGeomIdx = i;
                hs.hitMeshIdx = -1;
                hs.hitPrimIdx = -1;
                hs.uv = glm::vec2(0.f);
            }
        }
    }

    if (0) {
        TraverseSceneNaiveGltf(deviceGltfScene, seg, hs);
    }
    else {
        TraverseSceneBvh(deviceGltfScene, seg, hs);
    }

    if (!hs.hit) {
        intersections[path_index].t = -1.f;
        intersections[path_index].materialType = MaterialType::ENVMAP;
        return;
    }

    if (hs.hitGeomIdx != -1) {
        intersections[path_index].t = hs.tMin;
        intersections[path_index].materialId = geoms[hs.hitGeomIdx].materialid;
        intersections[path_index].materialType = geoms[hs.hitGeomIdx].materialType;
        intersections[path_index].surfaceNormal = hs.nWs;
        branchlessONB(intersections[path_index].surfaceNormal, intersections[path_index].tangentWs, intersections[path_index].bitangentWs);
        intersections[path_index].uv = glm::vec2(0.f);
        return;
    }

    if (hs.hitMeshIdx != -1) {
        intersections[path_index].t = hs.tMin;
        intersections[path_index].materialId =
            deviceGltfScene.meshes[hs.hitMeshIdx].primitives[hs.hitPrimIdx].materialIndex;
        intersections[path_index].materialType = MaterialType::PBR;
        intersections[path_index].surfaceNormal = hs.nWs;
        branchlessONB(intersections[path_index].surfaceNormal, intersections[path_index].tangentWs, intersections[path_index].bitangentWs);
        intersections[path_index].uv = hs.uv;
        return;
    }
}

// comparator for material sorting
struct IsectKeyLess {
    __host__ __device__
        bool operator()(const ShadeableIntersection& a,
            const ShadeableIntersection& b) const
    {
        const bool aMiss = (a.t < -EPSILON);
        const bool bMiss = (b.t < -EPSILON);

        // hits before miss
        if (aMiss != bMiss) return !aMiss;

        // both hits, then sort by material id

        if (!aMiss) {
            if (a.materialType != b.materialType) return a.materialType < b.materialType;
            return a.t < b.t;
        }

        // both misses, just sort based on distance
        return a.t < b.t;
    }
};

__global__ void kernResetIntBuffer(int N, int* intBuffer, int value) {
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (index < N) {
        intBuffer[index] = value;
    }
}

__global__ void kernIdentifyMaterialTypeStartEnd(int numPaths, const ShadeableIntersection* intersections,
    int* matTypeStartIndices, int* matTypeEndIndices) {
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (index >= numPaths) {
        return;
    }

    MaterialType curMatType = intersections[index].materialType;
    MaterialType prevMatType = (index > 0) ? intersections[index - 1].materialType : MaterialType::INVALID;
    MaterialType nextMatType = (index < (numPaths - 1)) ? intersections[index + 1].materialType : MaterialType::INVALID;

    if (curMatType != MaterialType::INVALID && curMatType != prevMatType) {
        matTypeStartIndices[static_cast<int>(curMatType)] = index;
    }

    if (curMatType != MaterialType::INVALID && curMatType != nextMatType) {
        matTypeEndIndices[static_cast<int>(curMatType)] = index;
    }
}

// Add the current iteration's output to the overall image
__global__ void finalGather(int nPaths, glm::vec3* image, PathSegment* iterationPaths)
{
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (index >= nPaths) return; 

    PathSegment iterationPath = iterationPaths[index];
    image[iterationPath.pixelIndex] += iterationPath.color; 
}

static void MaterialSortAndShade(
    int iter,
    int numPaths,
    int blockSize1d,
    ShadeableIntersection* dev_intersections,
    PathSegment* dev_paths,
    Material* dev_materials,
    int* dev_startIdx,
    int* dev_endIdx,
    int* hst_startIdx,
    int* hst_endIdx)
{
    thrust::sort_by_key(
        thrust::device,
        dev_intersections,
        dev_intersections + numPaths,
        dev_paths,
        IsectKeyLess());

    const int materialTypeCount = static_cast<int>(MaterialType::COUNT);
    const dim3 blocksMat((materialTypeCount + blockSize1d - 1) / blockSize1d);
    kernResetIntBuffer KERNEL_ARGS2(blocksMat, blockSize1d)(materialTypeCount, dev_startIdx, -1);
    kernResetIntBuffer KERNEL_ARGS2(blocksMat, blockSize1d)(materialTypeCount, dev_endIdx, -1);

    const dim3 blocksTrace((numPaths + blockSize1d - 1) / blockSize1d);
    kernIdentifyMaterialTypeStartEnd KERNEL_ARGS2(blocksTrace, blockSize1d)(
        numPaths, dev_intersections, dev_startIdx, dev_endIdx);

    cudaMemcpy(hst_startIdx, dev_startIdx, materialTypeCount * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(hst_endIdx, dev_endIdx, materialTypeCount * sizeof(int), cudaMemcpyDeviceToHost);

    for (int mt = 0; mt < materialTypeCount; ++mt) {
        const int start = hst_startIdx[mt];
        const int end = hst_endIdx[mt];

        if (start < 0 || end < start) continue;

        const int count = end - start + 1;
        ShadeableIntersection* isectSlice = dev_intersections + start;
        PathSegment* pathSlice = dev_paths + start;

        const int blocksRange = (count + blockSize1d - 1) / blockSize1d;

        switch (static_cast<MaterialType>(mt)) {
        case MaterialType::EMISSIVE:
            KernShadeEmissive KERNEL_ARGS2(blocksRange, blockSize1d)(iter, count, isectSlice, pathSlice, dev_materials);
            break;
        case MaterialType::DIFFUSE:
            KernShadeDiffuse KERNEL_ARGS2(blocksRange, blockSize1d)(iter, count, isectSlice, pathSlice, dev_materials);
            break;
        case MaterialType::SPECULAR:
            KernShadeSpecular KERNEL_ARGS2(blocksRange, blockSize1d)(iter, count, isectSlice, pathSlice, dev_materials);
            break;
        case MaterialType::TRANSMISSIVE:
            KernShadeTransmissive KERNEL_ARGS2(blocksRange, blockSize1d)(iter, count, isectSlice, pathSlice, dev_materials);
            break;
        case MaterialType::PBR:
            KernShadePbr KERNEL_ARGS2(blocksRange, blockSize1d)(iter, count, isectSlice, pathSlice, dev_materials, dev_textures); 
            break;
        case MaterialType::ENVMAP:
            KernShadeEnvMap KERNEL_ARGS2(blocksRange, blockSize1d)(iter, count, isectSlice, pathSlice, *envMap); 
            break; 
        default:
            KernShadeError KERNEL_ARGS2(blocksRange, blockSize1d)(iter, count, isectSlice, pathSlice); 
            break;
        }
    }
}

// pack averaged glm::vec3 into pitched float4 buffer (RGBA32F)
__global__ void packToFloat4(const glm::vec3* __restrict__ src,
    float4* __restrict__ dst,
    int w, int h, size_t pitchBytes,
    int iter)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;

    const int idx = x + y * w;
    const float invIt = 1.0f / max(iter, 1);

    char* rowBase = reinterpret_cast<char*>(dst) + y * pitchBytes;
    float4* row = reinterpret_cast<float4*>(rowBase);

    glm::vec3 c = src[idx] * invIt;
    row[x] = make_float4(c.x, c.y, c.z, 1.0f);
}

// blit pitched float4 -> PBO [0, 255]
__global__ void blitFloat4ToPBO(uchar4* pbo,
    int w, int h, size_t pitchBytes,
    const float4* __restrict__ src)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;

    const char* rowBase = reinterpret_cast<const char*>(src) + y * pitchBytes;
    const float4* row = reinterpret_cast<const float4*>(rowBase);
    float4 c = row[x];

    int index = x + y * w;
    auto to8 = [](float v)->unsigned char {
        v = fminf(fmaxf(v, 0.0f), 1.0f);
        return static_cast<unsigned char>(v * 255.0f + 0.5f);
    };

    pbo[index] = make_uchar4(to8(c.x), to8(c.y), to8(c.z), 255);
}


void pathtrace(uchar4* pbo, int frame, int iter)
{
    const int traceDepth = hst_scene->state.traceDepth;
    const Camera& cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;

    const dim3 blockSize2d(8, 8);
    const dim3 blocksPerGrid2d(
        (cam.resolution.x + blockSize2d.x - 1) / blockSize2d.x,
        (cam.resolution.y + blockSize2d.y - 1) / blockSize2d.y);

    const int blockSize1d = 128;

    // normal pass
    generateRayFromCamera KERNEL_ARGS2(blocksPerGrid2d, blockSize2d)(cam, iter, traceDepth, dev_paths);
    // checkCUDAError("generate camera ray");

    int depth = 0;
    PathSegment* dev_path_end = dev_paths + pixelcount;
    int numPaths = static_cast<int>(dev_path_end - dev_paths);

    // beauty pass
    generateRayFromCamera KERNEL_ARGS2(blocksPerGrid2d, blockSize2d)(cam, iter, traceDepth, dev_paths);
    // checkCUDAError("generate camera ray");
    bool iterationComplete = false;
    while (!iterationComplete)
    {
        cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));


        dim3 numblocksPathSegmentTracing = (numPaths + blockSize1d - 1) / blockSize1d;
        ComputeIntersections KERNEL_ARGS2(numblocksPathSegmentTracing, blockSize1d)(
            numPaths,
            dev_paths,
            dev_geoms,
            static_cast<int>(hst_scene->geoms.size()),
            gltfScene,
            dev_intersections);
        // checkCUDAError("trace one bounce");
        cudaDeviceSynchronize();

#ifdef NORMAL
        // normal-debug shading
        {
            const int blocksAll = (numPaths + blockSize1d - 1) / blockSize1d;
            KernShadeNormal KERNEL_ARGS2(blocksAll, blockSize1d)(
                iter,
                numPaths,
                dev_intersections,
                dev_materials, 
                dev_textures,
                dev_paths);
            // checkCUDAError("shade normals");
        }
#else
        // regular shading path
        if (g_settings.enableMaterialSorting) {
            MaterialSortAndShade(iter, numPaths, blockSize1d,
                dev_intersections, dev_paths, dev_materials,
                dev_startIdx, dev_endIdx, hst_startIdx, hst_endIdx);
        }
        else {
            const int blocksAll = (numPaths + blockSize1d - 1) / blockSize1d;
            KernShadeAllMaterials KERNEL_ARGS2(blocksAll, blockSize1d)(
                iter,
                numPaths,
                dev_intersections,
                dev_paths,
                dev_materials,
                dev_textures,
                *envMap);
        }
#endif
        if (g_settings.enableStreamCompaction) {
            PathSegment* mid = thrust::partition(thrust::device, dev_paths, dev_paths + numPaths, is_active());
            numPaths = static_cast<int>(mid - dev_paths);
        }

        iterationComplete = (numPaths == 0 || ++depth > traceDepth);
        if (guiData) guiData->TracedDepth = depth;

    }

    dim3 numBlocksPixels = (pixelcount + blockSize1d - 1) / blockSize1d;
    finalGather KERNEL_ARGS2(numBlocksPixels, blockSize1d)(pixelcount, dev_beauty, dev_paths);

    sendImageToPBO KERNEL_ARGS2(blocksPerGrid2d, blockSize2d)(pbo, cam.resolution, iter, dev_beauty);

    cudaMemcpy(hst_scene->state.beauty.data(), dev_beauty,
        pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

    // checkCUDAError("pathtrace");
}

void normalPass(int iterCount)
{
    const int traceDepth = hst_scene->state.traceDepth;
    const Camera& cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;


    const dim3 blockSize2d(8, 8);
    const dim3 blocksPerGrid2d(
        (cam.resolution.x + blockSize2d.x - 1) / blockSize2d.x,
        (cam.resolution.y + blockSize2d.y - 1) / blockSize2d.y);

    const int blockSize1d = 128;

    for(int iter = 0; iter < iterCount; ++iter) {
        // normal pass
        generateRayFromCamera KERNEL_ARGS2(blocksPerGrid2d, blockSize2d)(cam, iter, traceDepth, dev_paths);
        // checkCUDAError("generate camera ray");

        PathSegment* dev_path_end = dev_paths + pixelcount;
        int numPaths = static_cast<int>(dev_path_end - dev_paths);

        cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

        dim3 numblocksPathSegmentTracing = (numPaths + blockSize1d - 1) / blockSize1d;
        ComputeIntersections KERNEL_ARGS2(numblocksPathSegmentTracing, blockSize1d)(
            numPaths,
            dev_paths,
            dev_geoms,
            static_cast<int>(hst_scene->geoms.size()),
            gltfScene,
            dev_intersections);
        // checkCUDAError("trace one bounce");
        cudaDeviceSynchronize();

        const int blocksAll = (numPaths + blockSize1d - 1) / blockSize1d;
        KernShadeNormal KERNEL_ARGS2(blocksAll, blockSize1d)(
            iter,
            numPaths,
            dev_intersections,
            dev_materials, 
            dev_textures,
            dev_paths);
        // checkCUDAError("shade normals");

        dim3 numBlocksPixels = (pixelcount + blockSize1d - 1) / blockSize1d;
        finalGather KERNEL_ARGS2(numBlocksPixels, blockSize1d)(pixelcount, dev_normal, dev_paths);
    }

    

    cudaMemcpy(hst_scene->state.normal.data(), dev_normal,
        pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);
}

void albedoPass(int iterCount)
{
    const int traceDepth = hst_scene->state.traceDepth;
    const Camera& cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;


    const dim3 blockSize2d(8, 8);
    const dim3 blocksPerGrid2d(
        (cam.resolution.x + blockSize2d.x - 1) / blockSize2d.x,
        (cam.resolution.y + blockSize2d.y - 1) / blockSize2d.y);

    const int blockSize1d = 128;

    for (int iter = 0; iter < iterCount; ++iter) {
        // normal pass
        generateRayFromCamera KERNEL_ARGS2(blocksPerGrid2d, blockSize2d)(cam, iter, traceDepth, dev_paths);
        // checkCUDAError("generate camera ray");

        PathSegment* dev_path_end = dev_paths + pixelcount;
        int numPaths = static_cast<int>(dev_path_end - dev_paths);

        cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

        dim3 numblocksPathSegmentTracing = (numPaths + blockSize1d - 1) / blockSize1d;
        ComputeIntersections KERNEL_ARGS2(numblocksPathSegmentTracing, blockSize1d)(
            numPaths,
            dev_paths,
            dev_geoms,
            static_cast<int>(hst_scene->geoms.size()),
            gltfScene,
            dev_intersections);
        // checkCUDAError("trace one bounce");
        cudaDeviceSynchronize();

        const int blocksAll = (numPaths + blockSize1d - 1) / blockSize1d;
        KernShadeAlbedo KERNEL_ARGS2(blocksAll, blockSize1d)(
            iter,
            numPaths,
            dev_intersections,
            dev_materials, 
            dev_textures,
            dev_paths);
        // checkCUDAError("shade normals");

        dim3 numBlocksPixels = (pixelcount + blockSize1d - 1) / blockSize1d;
        finalGather KERNEL_ARGS2(numBlocksPixels, blockSize1d)(pixelcount, dev_albedo, dev_paths);
    }



    cudaMemcpy(hst_scene->state.albedo.data(), dev_albedo,
        pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);
}

void roughnessPass(int iterCount)
{
    const int traceDepth = hst_scene->state.traceDepth;
    const Camera& cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;


    const dim3 blockSize2d(8, 8);
    const dim3 blocksPerGrid2d(
        (cam.resolution.x + blockSize2d.x - 1) / blockSize2d.x,
        (cam.resolution.y + blockSize2d.y - 1) / blockSize2d.y);

    const int blockSize1d = 128;

    for (int iter = 0; iter < iterCount; ++iter) {
        // normal pass
        generateRayFromCamera KERNEL_ARGS2(blocksPerGrid2d, blockSize2d)(cam, iter, traceDepth, dev_paths);
        // checkCUDAError("generate camera ray");

        PathSegment* dev_path_end = dev_paths + pixelcount;
        int numPaths = static_cast<int>(dev_path_end - dev_paths);

        cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

        dim3 numblocksPathSegmentTracing = (numPaths + blockSize1d - 1) / blockSize1d;
        ComputeIntersections KERNEL_ARGS2(numblocksPathSegmentTracing, blockSize1d)(
            numPaths,
            dev_paths,
            dev_geoms,
            static_cast<int>(hst_scene->geoms.size()),
            gltfScene,
            dev_intersections);
        // checkCUDAError("trace one bounce");
        cudaDeviceSynchronize();

        const int blocksAll = (numPaths + blockSize1d - 1) / blockSize1d;
        KernShadeRoughness KERNEL_ARGS2(blocksAll, blockSize1d)(
            iter,
            numPaths,
            dev_intersections,
            dev_materials,
            dev_textures,
            dev_paths);
        // checkCUDAError("shade normals");

        dim3 numBlocksPixels = (pixelcount + blockSize1d - 1) / blockSize1d;
        finalGather KERNEL_ARGS2(numBlocksPixels, blockSize1d)(pixelcount, dev_roughness, dev_paths);
    }

    cudaMemcpy(hst_scene->state.roughness.data(), dev_roughness,
        pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);
}

void metallicPass(int iterCount)
{
    const int traceDepth = hst_scene->state.traceDepth;
    const Camera& cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;


    const dim3 blockSize2d(8, 8);
    const dim3 blocksPerGrid2d(
        (cam.resolution.x + blockSize2d.x - 1) / blockSize2d.x,
        (cam.resolution.y + blockSize2d.y - 1) / blockSize2d.y);

    const int blockSize1d = 128;

    for (int iter = 0; iter < iterCount; ++iter) {
        // normal pass
        generateRayFromCamera KERNEL_ARGS2(blocksPerGrid2d, blockSize2d)(cam, iter, traceDepth, dev_paths);
        // checkCUDAError("generate camera ray");

        PathSegment* dev_path_end = dev_paths + pixelcount;
        int numPaths = static_cast<int>(dev_path_end - dev_paths);

        cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

        dim3 numblocksPathSegmentTracing = (numPaths + blockSize1d - 1) / blockSize1d;
        ComputeIntersections KERNEL_ARGS2(numblocksPathSegmentTracing, blockSize1d)(
            numPaths,
            dev_paths,
            dev_geoms,
            static_cast<int>(hst_scene->geoms.size()),
            gltfScene,
            dev_intersections);
        checkCUDAError("trace one bounce");
        cudaDeviceSynchronize();

        const int blocksAll = (numPaths + blockSize1d - 1) / blockSize1d;
        KernShadeMetallic KERNEL_ARGS2(blocksAll, blockSize1d)(
            iter,
            numPaths,
            dev_intersections,
            dev_materials,
            dev_textures,
            dev_paths);
        // checkCUDAError("shade normals");

        dim3 numBlocksPixels = (pixelcount + blockSize1d - 1) / blockSize1d;
        finalGather KERNEL_ARGS2(numBlocksPixels, blockSize1d)(pixelcount, dev_metallic, dev_paths);
    }

    cudaMemcpy(hst_scene->state.metallic.data(), dev_metallic,
        pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);
}
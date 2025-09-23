#include "pathtrace.h"

#include <cstdio>
#include <cuda.h>
#include <cmath>
#include "device_launch_parameters.h"
#include "utilities.h"
#include "intersections.h"
#include "interactions.h"
#include "sceneStructs.h"
#include "scene.h"
#include "settings.h"
#include "shading/shading_common.cuh" 
#include "shading/shading_kernels.cuh"
#include "texture.h"
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/partition.h>

//Kernel that writes the image to the OpenGL PBO directly.
__global__ void sendImageToPBO(uchar4* pbo, glm::ivec2 resolution, int iter, glm::vec3* image)
{
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < resolution.x && y < resolution.y)
    {
        int index = x + (y * resolution.x);
        glm::vec3 pix = image[index];

        glm::ivec3 color;
        color.x = glm::clamp((int)(pix.x / iter * 255.0), 0, 255);
        color.y = glm::clamp((int)(pix.y / iter * 255.0), 0, 255);
        color.z = glm::clamp((int)(pix.z / iter * 255.0), 0, 255);

        // Each thread writes one pixel location in the texture (textel)
        pbo[index].w = 0;
        pbo[index].x = color.x;
        pbo[index].y = color.y;
        pbo[index].z = color.z;
    }
}

static Scene* hst_scene = NULL;
static GuiDataContainer* guiData = NULL;
static glm::vec3* dev_image = NULL;
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

    cudaMalloc(&dev_image, pixelcount * sizeof(glm::vec3));
    cudaMemset(dev_image, 0, pixelcount * sizeof(glm::vec3));

    cudaMalloc(&dev_paths, pixelcount * sizeof(PathSegment));

    cudaMalloc(&dev_geoms, scene->geoms.size() * sizeof(Geom));
    cudaMemcpy(dev_geoms, scene->geoms.data(), scene->geoms.size() * sizeof(Geom), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_materials, scene->materials.size() * sizeof(Material));
    cudaMemcpy(dev_materials, scene->materials.data(), scene->materials.size() * sizeof(Material), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_intersections, pixelcount * sizeof(ShadeableIntersection));
    cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

    cudaMalloc(&dev_startIdx, static_cast<int>(MaterialType::COUNT) * sizeof(int));
    cudaMalloc(&dev_endIdx, static_cast<int>(MaterialType::COUNT) * sizeof(int));

    hst_startIdx = new int[static_cast<int>(MaterialType::COUNT)];
    hst_endIdx = new int[static_cast<int>(MaterialType::COUNT)];

    envMap = &(scene->envMap); 
}

void pathtraceFree()
{
    cudaFree(dev_image);  // no-op if dev_image is null
    cudaFree(dev_paths);
    cudaFree(dev_geoms);
    cudaFree(dev_materials);
    cudaFree(dev_intersections);

    cudaFree(dev_startIdx); 
    cudaFree(dev_endIdx); 

    delete[] hst_startIdx; 
    delete[] hst_endIdx; 
}

/**
* Generate PathSegments with rays from the camera through the screen into the
* scene, which is the first bounce of rays.
*
* Antialiasing - add rays for sub-pixel sampling
* motion blur - jitter rays "in time"
* lens effect - jitter ray origin positions based on a lens
*/
__global__ void generateRayFromCamera(Camera cam, int iter, int traceDepth, PathSegment* pathSegments)
{
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < cam.resolution.x && y < cam.resolution.y) {
        int index = x + (y * cam.resolution.x);
        PathSegment& segment = pathSegments[index];

        segment.ray.origin = cam.position;
        segment.color = glm::vec3(1.0f, 1.0f, 1.0f);

        // TODO: implement antialiasing by jittering the ray
        segment.ray.direction = glm::normalize(cam.view
            - cam.right * cam.pixelLength.x * ((float)x - (float)cam.resolution.x * 0.5f)
            - cam.up * cam.pixelLength.y * ((float)y - (float)cam.resolution.y * 0.5f)
        );

        segment.pixelIndex = index;
        segment.remainingBounces = traceDepth;
        segment.shouldTerminate = false;   
    }
}

// TODO:
// computeIntersections handles generating ray intersections ONLY.
// Generating new rays is handled in your shader(s).
// Feel free to modify the code below.
__global__ void computeIntersections(
    int depth,
    int numPaths,
    PathSegment* pathSegments,
    Geom* geoms,
    int geomsSize,
    ShadeableIntersection* intersections)
{
    int path_index = blockIdx.x * blockDim.x + threadIdx.x;

    if (path_index < numPaths)
    {
        PathSegment& pathSegment = pathSegments[path_index];

        float t;
        glm::vec3 intersect_point;
        glm::vec3 normal;
        float t_min = FLT_MAX;
        int hit_geom_index = -1;
        bool outside = true;

        glm::vec3 tmp_intersect;
        glm::vec3 tmp_normal;

        // naive parse through global geoms

        for (int i = 0; i < geomsSize; i++)
        {
            Geom& geom = geoms[i];

            if (geom.type == CUBE)
            {
                t = boxIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
            }
            else if (geom.type == SPHERE)
            {
                t = sphereIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
            }
            // TODO: add more intersection tests here... triangle? metaball? CSG?

            // Compute the minimum t from the intersection tests to determine what
            // scene geometry object was hit first.
            if (t > 0.0f && t_min > t)
            {
                t_min = t;
                hit_geom_index = i;
                intersect_point = tmp_intersect;
                normal = tmp_normal;
            }
        }

        if (hit_geom_index == -1)
        {
            intersections[path_index].t = -1.0f; // just some value so that material sorting does 
            intersections[path_index].materialType = MaterialType::ENVMAP; 
        }
        else
        {
            // The ray hits something
            intersections[path_index].t = t_min;
            intersections[path_index].materialId = geoms[hit_geom_index].materialid;
            intersections[path_index].materialType = geoms[hit_geom_index].materialType; 
            intersections[path_index].surfaceNormal = normal;
        }
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
            kernShadeEmissive KERNEL_ARGS2(blocksRange, blockSize1d)(iter, count, isectSlice, pathSlice, dev_materials);
            break;
        case MaterialType::DIFFUSE:
            kernShadeDiffuse KERNEL_ARGS2(blocksRange, blockSize1d)(iter, count, isectSlice, pathSlice, dev_materials);
            break;
        case MaterialType::SPECULAR:
            kernShadeSpecular KERNEL_ARGS2(blocksRange, blockSize1d)(iter, count, isectSlice, pathSlice, dev_materials);
            break;
        case MaterialType::TRANSMISSIVE:
            kernShadeTransmissive KERNEL_ARGS2(blocksRange, blockSize1d)(iter, count, isectSlice, pathSlice, dev_materials);
            break;
        case MaterialType::PBR:
            kernShadePbr KERNEL_ARGS2(blocksRange, blockSize1d)(iter, count, isectSlice, pathSlice, dev_materials); 
            break;
        case MaterialType::ENVMAP:
            kernrShadeEnvMap KERNEL_ARGS2(blocksRange, blockSize1d)(iter, count, isectSlice, pathSlice, *envMap); 
        default:
            break;
        }
    }
}

/**
 * Wrapper for the __global__ call that sets up the kernel calls and does a ton
 * of memory management
 */
void pathtrace(uchar4* pbo, int frame, int iter)
{
    const int traceDepth = hst_scene->state.traceDepth;
    const Camera& cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;
    

    // 2D block for generating ray from camera
    const dim3 blockSize2d(8, 8);
    const dim3 blocksPerGrid2d(
        (cam.resolution.x + blockSize2d.x - 1) / blockSize2d.x,
        (cam.resolution.y + blockSize2d.y - 1) / blockSize2d.y);

    // 1D block for path tracing
    const int blockSize1d = 128;

    generateRayFromCamera KERNEL_ARGS2(blocksPerGrid2d, blockSize2d)(cam, iter, traceDepth, dev_paths);
    checkCUDAError("generate camera ray");

    int depth = 0;
    PathSegment* dev_path_end = dev_paths + pixelcount;
    int numPaths = dev_path_end - dev_paths;

    // --- PathSegment Tracing Stage ---
    // Shoot ray into scene, bounce between objects, push shading chunks

    bool iterationComplete = false;
    while (!iterationComplete)
    {
        // clean shading chunks
        cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

        // tracing
        dim3 numblocksPathSegmentTracing = (numPaths + blockSize1d - 1) / blockSize1d;
        computeIntersections KERNEL_ARGS2(numblocksPathSegmentTracing, blockSize1d) (
            depth,
            numPaths,
            dev_paths,
            dev_geoms,
            hst_scene->geoms.size(),
            dev_intersections
        );
        checkCUDAError("trace one bounce");
        cudaDeviceSynchronize();
        
        // material sorting 
        if (g_settings.enableMaterialSorting) {
            MaterialSortAndShade(iter, numPaths, blockSize1d,
                dev_intersections, dev_paths, dev_materials,
                dev_startIdx, dev_endIdx, hst_startIdx, hst_endIdx);
        }
        else { 
            // use all in one solution for shading 
            const int blocksAll = (numPaths + blockSize1d - 1) / blockSize1d;
            kernShadeAllMaterials KERNEL_ARGS2(blocksAll, blockSize1d)(
                iter,
                numPaths,
                dev_intersections,
                dev_paths,
                dev_materials, 
                *envMap
                );
        }
        
        if (g_settings.enableStreamCompaction) {
            PathSegment* mid = thrust::partition(thrust::device, dev_paths, dev_paths + numPaths, is_active());
            numPaths = static_cast<int>(mid - dev_paths);
        }
        
        iterationComplete = (numPaths == 0 || ++depth > traceDepth); 
        guiData ? guiData->TracedDepth = depth : 0;
    }

    // Assemble this iteration and apply it to the image
    dim3 numBlocksPixels = (pixelcount + blockSize1d - 1) / blockSize1d;
    finalGather KERNEL_ARGS2(numBlocksPixels, blockSize1d)(pixelcount, dev_image, dev_paths);

    ///////////////////////////////////////////////////////////////////////////

    // Send results to OpenGL buffer for rendering
    sendImageToPBO KERNEL_ARGS2(blocksPerGrid2d, blockSize2d)(pbo, cam.resolution, iter, dev_image);

    // Retrieve image from GPU
    cudaMemcpy(hst_scene->state.image.data(), dev_image,
        pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

    checkCUDAError("pathtrace");
}

#include "denoise_optix.h"
#include <optix_function_table_definition.h>
#include <optix_stubs.h>
#include <stdexcept>
#include <cstring>

static inline void checkOptix(OptixResult r, const char* msg = "OptiX call failed") {
    if (r != OPTIX_SUCCESS) throw std::runtime_error(msg);
}
static inline void checkCuda(cudaError_t e, const char* msg = nullptr) {
    if (e != cudaSuccess) throw std::runtime_error(msg ? msg : cudaGetErrorString(e));
}

static inline OptixImage2D createImage2D(
    CUdeviceptr ptr, unsigned int w, unsigned int h, size_t pitchBytes,
    OptixPixelFormat fmt = OPTIX_PIXEL_FORMAT_FLOAT4)
{
    OptixImage2D img{};
    img.data = ptr;
    img.width = w;
    img.height = h;
    img.rowStrideInBytes = static_cast<unsigned int>(pitchBytes);
    img.pixelStrideInBytes = (fmt == OPTIX_PIXEL_FORMAT_FLOAT4) ? sizeof(float) * 4 : sizeof(float) * 3;
    img.format = fmt;
    return img;
}

bool InitOptixDenoiser(OptixDenoiserContext& odc, int w, int h, cudaStream_t stream,
    OptixDenoiserModelKind model)
{
    if (odc.initialized) return true;

    checkOptix(optixInit(), "optixInit failed");

    CUcontext cuCtx = 0; // use current CUDA context
    OptixDeviceContextOptions ctxOpts{};
    ctxOpts.validationMode = OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_OFF;
    ctxOpts.logCallbackFunction = nullptr;
    checkOptix(optixDeviceContextCreate(cuCtx, &ctxOpts, &odc.ctx), "optixDeviceContextCreate failed");

    // create denoiser options
    OptixDenoiserOptions dopt{};
    // set these guide values to 1u when they are always provided
    dopt.guideAlbedo = 0u;  // can be null
    dopt.guideNormal = 0u;  // can be null
    dopt.denoiseAlpha = OPTIX_DENOISER_ALPHA_MODE_DENOISE;

    odc.stream = stream;
    odc.width = static_cast<unsigned int>(w);
    odc.height = static_cast<unsigned int>(h);

    // qurty memory sizes and allocated scratch
    OptixDenoiserSizes sz{};
    checkOptix(optixDenoiserComputeMemoryResources(odc.denoiser, odc.width, odc.height, &sz),
        "optixDenoiserComputeMemoryResources failed");

    odc.stateSizeBytes = sz.stateSizeInBytes;
    odc.scratchSizeBytes = sz.withoutOverlapScratchSizeInBytes;

    checkCuda(cudaMalloc(reinterpret_cast<void**>(&odc.d_state), odc.stateSizeBytes));
    checkCuda(cudaMalloc(reinterpret_cast<void**>(&odc.d_scratch), odc.scratchSizeBytes));

    checkOptix(optixDenoiserSetup(
        odc.denoiser, odc.stream, odc.width, odc.height,
        odc.d_state, odc.stateSizeBytes,
        odc.d_scratch, odc.scratchSizeBytes),
        "optixDenoiserSetup failed");

    odc.initialized = true;
    return true;
}

bool optixDenoise(OptixDenoiserContext& odc,
    CUdeviceptr inColor, size_t inPitchBytes,
    CUdeviceptr inAlbedo, size_t inAlbedoPitchBytes,
    CUdeviceptr inNormal, size_t inNormalPitchBytes,
    CUdeviceptr outColor, size_t outPitchBytes)
{
    if (!odc.initialized) return false;

    // io
    const OptixImage2D colorIn = createImage2D(inColor, odc.width, odc.height, inPitchBytes, OPTIX_PIXEL_FORMAT_FLOAT4);
    const OptixImage2D albedoIn = (inAlbedo ? createImage2D(inAlbedo, odc.width, odc.height, inAlbedoPitchBytes, OPTIX_PIXEL_FORMAT_FLOAT4) : OptixImage2D{});
    const OptixImage2D normalIn = (inNormal ? createImage2D(inNormal, odc.width, odc.height, inNormalPitchBytes, OPTIX_PIXEL_FORMAT_FLOAT4) : OptixImage2D{});
    const OptixImage2D colorOut = createImage2D(outColor, odc.width, odc.height, outPitchBytes, OPTIX_PIXEL_FORMAT_FLOAT4);

    OptixDenoiserGuideLayer guide{};
    if (inAlbedo) guide.albedo = albedoIn;
    if (inNormal) guide.normal = normalIn;

    OptixDenoiserLayer layer{};
    layer.input = colorIn;
    layer.output = colorOut;

    OptixDenoiserSizes sz{};
    checkOptix(optixDenoiserComputeMemoryResources(odc.denoiser, odc.width, odc.height, &sz),
        "optixDenoiserComputeMemoryResources (invoke) failed");

    OptixDenoiserParams params{};
    params.blendFactor = 0.0f;        // 0 = full denoise, 1 = original

    // invoke
    checkOptix(optixDenoiserInvoke(
        odc.denoiser, odc.stream, &params,
        odc.d_state, odc.stateSizeBytes,
        &guide, &layer, 1,
        0, 0,
        odc.d_scratch, odc.scratchSizeBytes),
        "optixDenoiserInvoke failed");

    return true;
}

void optixDenoiserShutdown(OptixDenoiserContext& odc)
{
    if (!odc.initialized) return;

    if (odc.d_state)   checkCuda(cudaFree(reinterpret_cast<void*>(odc.d_state)));
    if (odc.d_scratch) checkCuda(cudaFree(reinterpret_cast<void*>(odc.d_scratch)));
    if (odc.denoiser)  checkOptix(optixDenoiserDestroy(odc.denoiser), "optixDenoiserDestroy failed");
    if (odc.ctx)       checkOptix(optixDeviceContextDestroy(odc.ctx), "optixDeviceContextDestroy failed");

    std::memset(&odc, 0, sizeof(odc));
}


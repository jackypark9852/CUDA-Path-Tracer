#include "denoise_optix.h"
#include <cstring>
#include <cuda_runtime.h>
#include <optix_function_table_definition.h>
#include <stdexcept>

static inline void checkOptix(OptixResult r, const char* msg = "OptiX call failed") {
    if (r != OPTIX_SUCCESS) throw std::runtime_error(msg);
}
static inline void checkCuda(cudaError_t e, const char* msg = nullptr) {
    if (e != cudaSuccess) throw std::runtime_error(msg ? msg : cudaGetErrorString(e));
}

static inline void packVec3ToFloat4(const std::vector<glm::vec3>& in, std::vector<float4>& out) {
    out.resize(in.size());
    for (size_t i = 0; i < in.size(); ++i)
        out[i] = make_float4(in[i].x, in[i].y, in[i].z, 1.0f);
}
static inline void unpackFloat4ToVec3(const std::vector<float4>& in, std::vector<glm::vec3>& out) {
    out.resize(in.size());
    for (size_t i = 0; i < in.size(); ++i)
        out[i] = glm::vec3(in[i].x, in[i].y, in[i].z);
}

void OptixDenoiseVectors(
    int width, int height,
    const std::vector<glm::vec3>& color,                            // required
    std::vector<glm::vec3>& denoised,                               // required
    const std::vector<glm::vec3>* albedo,                           // optional
    const std::vector<glm::vec3>* normal,                           // optional
    OptixDenoiserModelKind model,                                   // LDR by default
    float blend                                                     // 0=full denoise, 1=original
) {
    if (width <= 0 || height <= 0) throw std::runtime_error("Invalid image size");
    const size_t N = static_cast<size_t>(width) * static_cast<size_t>(height);
    if (color.size() != N) throw std::runtime_error("color size != width*height");
    if (albedo && albedo->size() != N) throw std::runtime_error("albedo size mismatch");
    if (normal && normal->size() != N) throw std::runtime_error("normal size mismatch");

    // init optix and device context
    checkOptix(optixInit(), "optixInit failed");
    CUcontext cuCtx = 0;
    OptixDeviceContextOptions ctxOpts{};
    ctxOpts.validationMode = OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_OFF;
    ctxOpts.logCallbackFunction = nullptr;
    OptixDeviceContext ctx = nullptr;
    checkOptix(optixDeviceContextCreate(cuCtx, &ctxOpts, &ctx), "optixDeviceContextCreate failed");

    // denoiser options
    OptixDenoiserOptions dopt{};
    dopt.guideAlbedo = albedo ? 1u : 0u;
    dopt.guideNormal = normal ? 1u : 0u;
    dopt.denoiseAlpha = OPTIX_DENOISER_ALPHA_MODE_COPY;
    OptixDenoiser den = nullptr;
    checkOptix(optixDenoiserCreate(ctx, model, &dopt, &den), "optixDenoiserCreate failed");

    // set sizes
    OptixDenoiserSizes sizes{};
    checkOptix(optixDenoiserComputeMemoryResources(den, width, height, &sizes),
        "optixDenoiserComputeMemoryResources failed");

    void* d_state = nullptr;
    void* d_scratch = nullptr;
    checkCuda(cudaMalloc(&d_state, sizes.stateSizeInBytes));
    checkCuda(cudaMalloc(&d_scratch, sizes.withoutOverlapScratchSizeInBytes));

    // pitched io buffers
    float4* d_inColor = nullptr, * d_outColor = nullptr, * d_inAlbedo = nullptr, * d_inNormal = nullptr;
    size_t pitchIn = 0, pitchOut = 0, pitchAlb = 0, pitchNrm = 0;

    checkCuda(cudaMallocPitch((void**)&d_inColor, &pitchIn, width * sizeof(float4), height));
    checkCuda(cudaMallocPitch((void**)&d_outColor, &pitchOut, width * sizeof(float4), height));
    if (albedo) checkCuda(cudaMallocPitch((void**)&d_inAlbedo, &pitchAlb, width * sizeof(float4), height));
    if (normal) checkCuda(cudaMallocPitch((void**)&d_inNormal, &pitchNrm, width * sizeof(float4), height));

    // upload
    std::vector<float4> hCol4, hAlb4, hNrm4;
    packVec3ToFloat4(color, hCol4);
    checkCuda(cudaMemcpy2D(d_inColor, pitchIn,
        hCol4.data(), width * sizeof(float4),
        width * sizeof(float4), height,
        cudaMemcpyHostToDevice));
    if (albedo) {
        packVec3ToFloat4(*albedo, hAlb4);
        checkCuda(cudaMemcpy2D(d_inAlbedo, pitchAlb,
            hAlb4.data(), width * sizeof(float4),
            width * sizeof(float4), height,
            cudaMemcpyHostToDevice));
    }
    if (normal) {
        packVec3ToFloat4(*normal, hNrm4);
        checkCuda(cudaMemcpy2D(d_inNormal, pitchNrm,
            hNrm4.data(), width * sizeof(float4),
            width * sizeof(float4), height,
            cudaMemcpyHostToDevice));
    }

    // setpup once
    checkOptix(optixDenoiserSetup(
        den, /*stream*/0, width, height,
        reinterpret_cast<CUdeviceptr>(d_state), sizes.stateSizeInBytes,
        reinterpret_cast<CUdeviceptr>(d_scratch), sizes.withoutOverlapScratchSizeInBytes),
        "optixDenoiserSetup failed");

    // wrap as OptixImage2D
    auto makeImg = [](CUdeviceptr ptr, unsigned w, unsigned h, size_t pitch) {
        OptixImage2D img{};
        img.data = ptr;
        img.width = w;
        img.height = h;
        img.rowStrideInBytes = static_cast<unsigned>(pitch);
        img.pixelStrideInBytes = sizeof(float4);
        img.format = OPTIX_PIXEL_FORMAT_FLOAT4;
        return img;
    };
    const OptixImage2D imgIn = makeImg(reinterpret_cast<CUdeviceptr>(d_inColor), width, height, pitchIn);
    const OptixImage2D imgOut = makeImg(reinterpret_cast<CUdeviceptr>(d_outColor), width, height, pitchOut);
    const OptixImage2D imgAlb = albedo ? makeImg(reinterpret_cast<CUdeviceptr>(d_inAlbedo), width, height, pitchAlb) : OptixImage2D{};
    const OptixImage2D imgNrm = normal ? makeImg(reinterpret_cast<CUdeviceptr>(d_inNormal), width, height, pitchNrm) : OptixImage2D{};

    OptixDenoiserGuideLayer guide{};
    if (albedo) guide.albedo = imgAlb;
    if (normal) guide.normal = imgNrm;

    OptixDenoiserLayer layer{};
    layer.input = imgIn;
    layer.output = imgOut;

    OptixDenoiserParams params{};
    params.hdrIntensity = 0;     // LDR path, no HDR intensity
    params.blendFactor = blend; // 0..1

    // invoke
    checkOptix(optixDenoiserInvoke(
        den, /*stream*/0, &params,
        reinterpret_cast<CUdeviceptr>(d_state), sizes.stateSizeInBytes,
        &guide, &layer, 1,
        /*offsetX*/0, /*offsetY*/0,
        reinterpret_cast<CUdeviceptr>(d_scratch), sizes.withoutOverlapScratchSizeInBytes),
        "optixDenoiserInvoke failed");

    // download
    std::vector<float4> hOut4(N);
    checkCuda(cudaMemcpy2D(hOut4.data(), width * sizeof(float4),
        d_outColor, pitchOut,
        width * sizeof(float4), height,
        cudaMemcpyDeviceToHost));

    // clean up
    if (d_inNormal)  cudaFree(d_inNormal);
    if (d_inAlbedo)  cudaFree(d_inAlbedo);
    cudaFree(d_outColor);
    cudaFree(d_inColor);
    cudaFree(d_scratch);
    cudaFree(d_state);
    checkOptix(optixDenoiserDestroy(den), "optixDenoiserDestroy failed");
    checkOptix(optixDeviceContextDestroy(ctx), "optixDeviceContextDestroy failed");

    // convert back to glm::vec3
    unpackFloat4ToVec3(hOut4, denoised);
}

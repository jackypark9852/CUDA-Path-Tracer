#pragma once

#include <optix.h>
#include <optix_stubs.h>
#include <cuda_runtime.h>

struct OptixDenoiserContext {
    OptixDeviceContext ctx = nullptr;
    OptixDenoiser denoiser = nullptr;
    CUdeviceptr d_state = 0;
    CUdeviceptr d_scratch = 0;
    size_t stateSizeBytes = 0;
    size_t scratchSizeBytes = 0;
    unsigned int width = 0, height = 0;
    cudaStream_t stream = 0;
    bool initialized = false;
};

bool  InitOptixDenoiser(OptixDenoiserContext& odc, int width, int height, cudaStream_t stream = 0,
    OptixDenoiserModelKind model = OPTIX_DENOISER_MODEL_KIND_HDR);
bool  optixDenoise(OptixDenoiserContext& odc,
    CUdeviceptr inColor, size_t inPitchBytes,
    CUdeviceptr inAlbedo, size_t inAlbedoPitchBytes, // pass 0 if unused
    CUdeviceptr inNormal, size_t inNormalPitchBytes,  // pass 0 if unused
    CUdeviceptr outColor, size_t outPitchBytes);
void  optixDenoiserShutdown(OptixDenoiserContext& odc);


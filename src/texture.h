#pragma once
#include <filesystem>
#include <vector>
#include <cuda_runtime.h>

namespace cpt {
    enum class PixelFormat { R8, RG8, RGBA8, R16F, RG16F, RGBA16F, R32F, RG32F, RGBA32F };

    enum class ColorSpace { Linear, sRGB };

    // sampler state for cudaTextureObject
    struct Sampler {
        cudaTextureAddressMode addressU = cudaAddressModeClamp;
        cudaTextureAddressMode addressV = cudaAddressModeClamp;
        cudaTextureFilterMode  filter = cudaFilterModeLinear;
        bool                   normalizedCoords = true;
        cudaTextureReadMode    readMode = cudaReadModeElementType;
    };
    
    // describes what type of CUDA texture to create
    struct TextureDesc {
        PixelFormat pixelFormat = PixelFormat::RGBA32F;
        ColorSpace  colorSpace = ColorSpace::Linear;
        Sampler     sampler = {};
    };


    // device facing texture descriptor
    struct Texture2D {
        int width = 0;
        int height = 0;

        // backing storage for texture/surface (don't deref on device)
        cudaArray_t array = nullptr;

        // texture handle for device-side sampling
        cudaTextureObject_t texObj = 0;

        // metadata you may want to carry around
        TextureDesc desc{};

        __host__ __device__ explicit operator bool() const { return texObj != 0; }
    };

    // --- creation / teardown helpers (host-side) --------------------------------
    bool createTextureFromFile(Texture2D& out,
        const std::filesystem::path& filePath,
        const TextureDesc& desc,
        cudaStream_t stream = 0);

    // create from already-prepared pixels with a given row pitch in bytes.
    bool createTextureFromPixels(Texture2D& out,
        int w, int h,
        const void* pixels, size_t rowPitchBytes,
        const TextureDesc& desc,
        cudaStream_t stream = 0);

    // clean-up CUDA resources
    void destroyTexture(Texture2D& t);

    // --- loader & format utilities ----------------------------------------------
    bool loadFile(const std::filesystem::path& path,
        PixelFormat targetFormat,
        ColorSpace  srcColorSpace,
        std::vector<unsigned char>& outBytes,
        int& w, int& h, size_t& rowPitch);

    // CUDA channel descriptor and element size for a PixelFormat.
    cudaChannelFormatDesc channelDesc(PixelFormat fmt);
    size_t                bytesPerPixel(PixelFormat fmt);

    // more helpers
    bool  isFloat16(PixelFormat f);
    bool  isFloat32(PixelFormat f);
    int   dstChannels(PixelFormat f);
    float srgbToLinear(float cs);

}

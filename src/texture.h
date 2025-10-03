#pragma once
#include <filesystem>
#include <vector>
#include <cuda_runtime.h>

namespace cpt {
    enum class PixelFormat { R8, RG8, RGBA8, R16F, RG16F, RGBA16F, R32F, RG32F, RGBA32F };

    enum class ColorSpace { Linear, sRGB };

    // sampler state for cudaTextureObject
    struct SamplerDesc {
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
        SamplerDesc     sampler = {};
    };


    // device facing texture descriptor
    struct Texture2D {
        int width = 0;
        int height = 0;

        // backing storage for texture/surface; never deref this on device
        cudaArray_t array = nullptr;

        // texture handle for device-side sampling
        cudaTextureObject_t texObj = 0;

        // metadata you may want to carry around
        TextureDesc desc{};

        __host__ __device__ explicit operator bool() const { return texObj != 0; }
    };

    // helpers
    bool createTextureFromFile(Texture2D& out,
        const std::filesystem::path& filePath,
        const TextureDesc& desc,
        cudaStream_t stream = 0);

    bool createTextureFromPixels(Texture2D& out,
        int w, int h,
        const void* pixels, size_t rowPitchBytes,
        const TextureDesc& desc,
        cudaStream_t stream = 0);

    // clean-up resources
    void destroyTexture(Texture2D& t);

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

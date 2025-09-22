#pragma once
#include <string>
#include <vector>
#include <filesystem>
#include <cuda_runtime.h>

namespace cpt {
    class Texture2D {
    public:
        enum class PixelFormat { R8, RG8, RGBA8, R16F, RG16F, RGBA16F, R32F, RG32F, RGBA32F };

        enum class ColorSpace { Linear, sRGB };

        struct Sampler {
            cudaTextureAddressMode addressU = cudaAddressModeClamp;
            cudaTextureAddressMode addressV = cudaAddressModeClamp;
            cudaTextureFilterMode  filter = cudaFilterModeLinear;
            bool                   normalizedCoords = true;
            cudaTextureReadMode    readMode = cudaReadModeElementType;
        };

        struct TextureDesc {
            PixelFormat pixelFormat = PixelFormat::RGBA32F;
            ColorSpace  colorSpace = ColorSpace::Linear;
            Sampler     sampler = {};
        };

        struct TextureView {
            cudaTextureObject_t handle = 0;
            int width = 0, height = 0;
            __host__ __device__ explicit operator bool() const { return handle != 0; }
        };

        Texture2D() = delete;
        explicit Texture2D(const std::filesystem::path& filePath,
            const TextureDesc& desc = {},
            cudaStream_t stream = 0);

        Texture2D(int width, int height,
            const void* pixels, size_t rowPitchBytes,
            const TextureDesc& desc = {},
            cudaStream_t stream = 0);

        // move-only
        Texture2D(Texture2D&& other) noexcept { moveFrom(std::move(other)); }
        Texture2D& operator=(Texture2D&& other) noexcept {
            if (this != &other) { destroy(); moveFrom(std::move(other)); }
            return *this;
        }
        Texture2D(const Texture2D&) = delete;
        Texture2D& operator=(const Texture2D&) = delete;

        ~Texture2D() { destroy(); }

        // acessors
        TextureView view() const { return { texObj_, width_, height_ }; }
        int  width()  const { return width_; }
        int  height() const { return height_; }
        bool valid()  const { return texObj_ != 0; }

    private:
        int width_ = 0, height_ = 0;
        cudaArray_t array_ = nullptr;
        cudaTextureObject_t texObj_ = 0;
        TextureDesc desc;

        void destroy() noexcept;
        void moveFrom(Texture2D&& other) noexcept;

        // creation helpers
        void createArray(PixelFormat fmt, int w, int h);
        void createTextureObject(const Sampler& s, PixelFormat fmt, ColorSpace cs);
        static cudaChannelFormatDesc channelDesc(PixelFormat fmt);
        static size_t bytesPerPixel(PixelFormat fmt);

        // file loading
        static bool loadFile(const std::filesystem::path& p,
            PixelFormat targetFormat, ColorSpace cs,
            std::vector<unsigned char>& outBytes,
            int& w, int& h, size_t& rowPitch);
    };
}

#include "texture.h"

#include <stdexcept>
#include <vector>
#include <cstring>
#include <stb_image.h>

namespace cpt {

    static inline bool isFloat16(cpt::Texture2D::PixelFormat f) {
        using PF = cpt::Texture2D::PixelFormat;
        return f == PF::R16F || f == PF::RG16F || f == PF::RGBA16F;
    }

    static inline bool isFloat32(cpt::Texture2D::PixelFormat f) {
        using PF = cpt::Texture2D::PixelFormat;
        return f == PF::R32F || f == PF::RG32F || f == PF::RGBA32F;
    }

    static inline int dstChannels(cpt::Texture2D::PixelFormat f) {
        using PF = cpt::Texture2D::PixelFormat;
        switch (f) {
        case PF::R8: case PF::R16F: case PF::R32F: return 1;
        case PF::RG8: case PF::RG16F: case PF::RG32F: return 2;
        case PF::RGBA8: case PF::RGBA16F: case PF::RGBA32F: return 4;
        default: return 0;
        }
    }

    static inline float srgbToLinear(float cs) {
        return (cs <= 0.04045f) ? (cs / 12.92f) : powf((cs + 0.055f) / 1.055f, 2.4f);
    }

    Texture2D::Texture2D(const std::filesystem::path& filePath,
        const TextureDesc& d,
        cudaStream_t stream)
        : desc(d) {
        std::vector<unsigned char> bytes;
        int w = 0, h = 0;
        size_t rowPitch = 0;
        if (!loadFile(filePath, desc.pixelFormat, desc.colorSpace, bytes, w, h, rowPitch)) {
            throw std::runtime_error("Texture2D: failed to load file: " + filePath.string());
        }
        width_ = w;
        height_ = h;
        createArray(desc.pixelFormat, w, h);
        const size_t rowBytes = static_cast<size_t>(w) * bytesPerPixel(desc.pixelFormat);
        cudaMemcpy2DToArrayAsync(array_, 0, 0,
            bytes.data(), rowPitch,
            rowBytes, h,
            cudaMemcpyHostToDevice, stream);
        createTextureObject(desc.sampler, desc.pixelFormat, desc.colorSpace);
    }

    Texture2D::Texture2D(int w, int h,
        const void* pixels, size_t rowPitchBytes,
        const TextureDesc& d,
        cudaStream_t stream)
        : desc(d) {
        width_ = w;
        height_ = h;
        createArray(desc.pixelFormat, w, h);
        const size_t rowBytes = static_cast<size_t>(w) * bytesPerPixel(desc.pixelFormat);
        cudaMemcpy2DToArrayAsync(array_, 0, 0,
            pixels, rowPitchBytes,
            rowBytes, h,
            cudaMemcpyHostToDevice, stream);
        createTextureObject(desc.sampler, desc.pixelFormat, desc.colorSpace);
    }

    void Texture2D::destroy() noexcept {
        if (texObj_) { cudaDestroyTextureObject(texObj_); texObj_ = 0; }
        if (array_) { cudaFreeArray(array_); array_ = nullptr; }
        width_ = 0; height_ = 0;
    }

    void Texture2D::moveFrom(Texture2D&& other) noexcept {
        width_ = other.width_;
        height_ = other.height_;
        array_ = other.array_;   other.array_ = nullptr;
        texObj_ = other.texObj_;  other.texObj_ = 0;
        desc = other.desc;
    }

    void Texture2D::createArray(PixelFormat fmt, int w, int h) {
        auto ch = channelDesc(fmt);
        cudaMallocArray(&array_, &ch, w, h, cudaArrayDefault);
    }

    void Texture2D::createTextureObject(const Sampler& s, PixelFormat, ColorSpace) {
        cudaResourceDesc res{};
        res.resType = cudaResourceTypeArray;
        res.res.array.array = array_;

        cudaTextureDesc tex{};
        tex.addressMode[0] = s.addressU;
        tex.addressMode[1] = s.addressV;
        tex.addressMode[2] = s.addressV;
        tex.filterMode = s.filter;
        tex.normalizedCoords = s.normalizedCoords ? 1 : 0;
        tex.readMode = s.readMode;

        cudaCreateTextureObject(&texObj_, &res, &tex, nullptr);
    }

    cudaChannelFormatDesc Texture2D::channelDesc(PixelFormat fmt) {
        switch (fmt) {
        case PixelFormat::R8:      return cudaCreateChannelDesc(8, 0, 0, 0, cudaChannelFormatKindUnsigned);
        case PixelFormat::RG8:     return cudaCreateChannelDesc(8, 8, 0, 0, cudaChannelFormatKindUnsigned);
        case PixelFormat::RGBA8:   return cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindUnsigned);
        case PixelFormat::R16F:    return cudaCreateChannelDesc(16, 0, 0, 0, cudaChannelFormatKindFloat);
        case PixelFormat::RG16F:   return cudaCreateChannelDesc(16, 16, 0, 0, cudaChannelFormatKindFloat);
        case PixelFormat::RGBA16F: return cudaCreateChannelDesc(16, 16, 16, 16, cudaChannelFormatKindFloat);
        case PixelFormat::R32F:    return cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
        case PixelFormat::RG32F:   return cudaCreateChannelDesc(32, 32, 0, 0, cudaChannelFormatKindFloat);
        case PixelFormat::RGBA32F: return cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat);
        default:                   return cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat);
        }
    }

    size_t Texture2D::bytesPerPixel(PixelFormat fmt) {
        switch (fmt) {
        case PixelFormat::R8:      return 1;
        case PixelFormat::RG8:     return 2;
        case PixelFormat::RGBA8:   return 4;
        case PixelFormat::R16F:    return 2;
        case PixelFormat::RG16F:   return 4;
        case PixelFormat::RGBA16F: return 8;
        case PixelFormat::R32F:    return 4;
        case PixelFormat::RG32F:   return 8;
        case PixelFormat::RGBA32F: return 16;
        default:                   return 16;
        }
    }

    bool Texture2D::loadFile(const std::filesystem::path& path,
        PixelFormat targetFormat,
        ColorSpace cs,
        std::vector<unsigned char>& outBytes,
        int& w,
        int& h,
        size_t& rowPitch)
    {
        // 16F target not handled (stb loads 16-bit as U16, not half-float)
        if (isFloat16(targetFormat)) {
            return false;
        }

        const int dstC = dstChannels(targetFormat);
        if (dstC == 0) return false;

        // repack source components into CUDA friendly format 
        int srcC = 0;
        int x = 0, y = 0;

        // float (HDR) path: load as float32 with stbi_loadf, then repack to dst channel count.
        // use this when the CUDA array expects 32-bit float texels (R32F/RG32F/RGBA32F).
        if (isFloat32(targetFormat)) {
            float* src = stbi_loadf(path.string().c_str(), &x, &y, &srcC, 0);
            if (!src) return false;

            const size_t stride = static_cast<size_t>(x) * dstC * sizeof(float);
            outBytes.resize(static_cast<size_t>(y) * stride);
            auto* dst = reinterpret_cast<float*>(outBytes.data());

            // repack / convert channels
            for (int j = 0; j < y; ++j) {
                for (int i = 0; i < x; ++i) {
                    const float* s = src + (static_cast<size_t>(j) * x + i) * srcC;
                    float* d = dst + (static_cast<size_t>(j) * x + i) * dstC;

                    // pack into greyscale
                    if (dstC == 1) {
                        float g = (srcC >= 3) ? (0.2126f * s[0] + 0.7152f * s[1] + 0.0722f * s[2])
                            : s[0];
                        // sRGB -> linear if requested
                        d[0] = (cs == ColorSpace::sRGB) ? srgbToLinear(g) : g;
                    }
                    // pack into 2 channels
                    else if (dstC == 2) {
                        // typically non-color data (e.g., roughness/metallic) -> avoid gamma
                        d[0] = (srcC >= 1) ? s[0] : 0.0f;
                        d[1] = (srcC >= 2) ? s[1] : d[0];
                    }
                    // pack into RGBA
                    else { // dstC == 4
                        float r = (srcC >= 1) ? s[0] : 0.0f;
                        float g = (srcC >= 2) ? s[1] : r;
                        float b = (srcC >= 3) ? s[2] : r;
                        float a = (srcC >= 4) ? s[3] : 1.0f;

                        if (cs == ColorSpace::sRGB) {
                            r = srgbToLinear(r);
                            g = srgbToLinear(g);
                            b = srgbToLinear(b);
                        }
                        d[0] = r; d[1] = g; d[2] = b; d[3] = a;
                    }
                }
            }

            stbi_image_free(src);
            w = x; h = y;
            rowPitch = static_cast<size_t>(x) * dstC * sizeof(float);
            return true;
        }
        // byte/UNORM path: load as 8-bit with stbi_load and repack to dst channel count.
        // use this when the CUDA array expects 8-bit normalized texels (R8/RG8/RGBA8).
        // values remain in [0,255]; sampling can return normalized floats if cudaReadModeNormalizedFloat is used.
        else {
            unsigned char* src = stbi_load(path.string().c_str(), &x, &y, &srcC, 0);
            if (!src) return false;

            const size_t stride = static_cast<size_t>(x) * dstC * sizeof(unsigned char);
            outBytes.resize(static_cast<size_t>(y) * stride);
            auto* dst = outBytes.data();

            auto toByte = [](float v) -> unsigned char {
                v = v < 0.f ? 0.f : (v > 1.f ? 1.f : v);
                return static_cast<unsigned char>(v * 255.f + 0.5f);
            };

            for (int j = 0; j < y; ++j) {
                for (int i = 0; i < x; ++i) {
                    const unsigned char* s = src + (static_cast<size_t>(j) * x + i) * srcC;
                    unsigned char* d = dst + (static_cast<size_t>(j) * x + i) * dstC;

                    if (dstC == 1) {
                        // grayscale (treat as color if sRGB, convert)
                        if (srcC >= 3) {
                            float r = s[0] / 255.f, g = s[1] / 255.f, b = s[2] / 255.f;
                            float gray = 0.2126f * r + 0.7152f * g + 0.0722f * b;
                            d[0] = (cs == ColorSpace::sRGB) ? toByte(srgbToLinear(gray)) : toByte(gray);
                        }
                        else {
                            float v = s[0] / 255.f;
                            d[0] = (cs == ColorSpace::sRGB) ? toByte(srgbToLinear(v)) : s[0];
                        }
                    }
                    else if (dstC == 2) {
                        // typically non-color; don't gamma convert to avoid corrupting data
                        if (srcC >= 2) { d[0] = s[0]; d[1] = s[1]; }
                        else /* srcC==1 */ { d[0] = s[0]; d[1] = s[0]; }
                    }
                    else { // dstC == 4
                        if (cs == ColorSpace::sRGB) {
                            // convert RGB from sRGB->linear, alpha untouched
                            float r = (srcC >= 1 ? s[0] : 0) / 255.f;
                            float g = (srcC >= 2 ? s[1] : s[0]) / 255.f;
                            float b = (srcC >= 3 ? s[2] : s[0]) / 255.f;
                            unsigned char a = (srcC >= 4 ? s[3] : 255);
                            d[0] = toByte(srgbToLinear(r));
                            d[1] = toByte(srgbToLinear(g));
                            d[2] = toByte(srgbToLinear(b));
                            d[3] = a;
                        }
                        else {
                            // already linear; just repack
                            if (srcC == 4) { d[0] = s[0]; d[1] = s[1]; d[2] = s[2]; d[3] = s[3]; }
                            else if (srcC == 3) { d[0] = s[0]; d[1] = s[1]; d[2] = s[2]; d[3] = 255; }
                            else if (srcC == 2) { d[0] = s[0]; d[1] = s[1]; d[2] = 0;   d[3] = 255; }
                            else /* srcC==1 */ { d[0] = s[0]; d[1] = s[0]; d[2] = s[0]; d[3] = 255; }
                        }
                    }
                }
            }

            stbi_image_free(src);
            w = x; h = y;
            rowPitch = static_cast<size_t>(x) * dstC * sizeof(unsigned char);
            return true;
        }
    }
}

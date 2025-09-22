#include "texture.h"
#include "utilities.h"
#include <stb_image.h>     // no IMPLEMENTATION macro here; make sure you compile it once in your project
#include <cmath>
#include <cstring>
#include <stdexcept>

namespace cpt {
    static bool   createArray(Texture2D& t, PixelFormat fmt, int w, int h);
    static bool   createTextureObject(Texture2D& t, const Sampler& s);
    static void   clear(Texture2D& t);

    bool isFloat16(PixelFormat f) {
        using PF = PixelFormat;
        return f == PF::R16F || f == PF::RG16F || f == PF::RGBA16F;
    }

    bool isFloat32(PixelFormat f) {
        using PF = PixelFormat;
        return f == PF::R32F || f == PF::RG32F || f == PF::RGBA32F;
    }

    int dstChannels(PixelFormat f) {
        using PF = PixelFormat;
        switch (f) {
        case PF::R8: case PF::R16F: case PF::R32F: return 1;
        case PF::RG8: case PF::RG16F: case PF::RG32F: return 2;
        case PF::RGBA8: case PF::RGBA16F: case PF::RGBA32F: return 4;
        default: return 0;
        }
    }

    float srgbToLinear(float cs) {
        return (cs <= 0.04045f) ? (cs / 12.92f) : std::pow((cs + 0.055f) / 1.055f, 2.4f);
    }

    cudaChannelFormatDesc channelDesc(PixelFormat fmt) {
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

    size_t bytesPerPixel(PixelFormat fmt) {
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

    bool loadFile(const std::filesystem::path& path,
        PixelFormat targetFormat,
        ColorSpace  srcColorSpace,
        std::vector<unsigned char>& outBytes,
        int& w, int& h, size_t& rowPitch)
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
                        float g = (srcC >= 3) ? (0.2126f * s[0] + 0.7152f * s[1] + 0.0722f * s[2]) : s[0];
                        // sRGB -> linear if requested
                        d[0] = (srcColorSpace == ColorSpace::sRGB) ? srgbToLinear(g) : g;
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

                        if (srcColorSpace == ColorSpace::sRGB) {
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
                            d[0] = (srcColorSpace == ColorSpace::sRGB) ? toByte(srgbToLinear(gray)) : toByte(gray);
                        }
                        else {
                            float v = s[0] / 255.f;
                            d[0] = (srcColorSpace == ColorSpace::sRGB) ? toByte(srgbToLinear(v)) : s[0];
                        }
                    }
                    else if (dstC == 2) {
                        // typically non-color; don't gamma convert to avoid corrupting data
                        if (srcC >= 2) { d[0] = s[0]; d[1] = s[1]; }
                        else /* srcC==1 */ { d[0] = s[0]; d[1] = s[0]; }
                    }
                    else { // dstC == 4
                        if (srcColorSpace == ColorSpace::sRGB) {
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

    bool createTextureFromFile(Texture2D& out,
        const std::filesystem::path& filePath,
        const TextureDesc& d,
        cudaStream_t stream)
    {
        std::vector<unsigned char> bytes;
        int w = 0, h = 0;
        size_t rowPitch = 0;

        if (!loadFile(filePath, d.pixelFormat, d.colorSpace, bytes, w, h, rowPitch)) {
            return false;
        }

        out.desc = d;
        out.width = w;
        out.height = h;

        if (!createArray(out, d.pixelFormat, w, h)) {
            clear(out);
            return false;
        }

        const size_t rowBytes = static_cast<size_t>(w) * bytesPerPixel(d.pixelFormat);
        if (cudaMemcpy2DToArrayAsync(out.array, 0, 0,
            bytes.data(), rowPitch,
            rowBytes, h,
            cudaMemcpyHostToDevice, stream) != cudaSuccess) {
            clear(out);
            checkCUDAError("cudaMemcpy2DToArrayAsync");
            return false;
        }

        if (!createTextureObject(out, d.sampler)) {
            clear(out);
            checkCUDAError("createTextureObject");
            return false;
        }

        return true;
    }

    bool createTextureFromPixels(Texture2D& out,
        int w, int h,
        const void* pixels, size_t rowPitchBytes,
        const TextureDesc& d,
        cudaStream_t stream)
    {
        Texture2D tmp{};
        tmp.desc = d;
        tmp.width = w;
        tmp.height = h;

        if (!createArray(tmp, d.pixelFormat, w, h)) {
            clear(tmp);
            return false;
        }

        const size_t rowBytes = static_cast<size_t>(w) * bytesPerPixel(d.pixelFormat);
        if (cudaMemcpy2DToArrayAsync(tmp.array, 0, 0,
            pixels, rowPitchBytes,
            rowBytes, h,
            cudaMemcpyHostToDevice, stream) != cudaSuccess) {
            clear(tmp);
            return false;
        }

        if (!createTextureObject(tmp, d.sampler)) {
            clear(tmp);
            return false;
        }

        destroyTexture(out);
        out = tmp;
        return true;
    }

    void destroyTexture(Texture2D& t) {
        if (t.texObj) {
            cudaDestroyTextureObject(t.texObj);
            checkCUDAError("cudaDestroyTextureObject");
            t.texObj = 0;
        }
        if (t.array) {
            cudaFreeArray(t.array);
            checkCUDAError("cudaFreeArray");
            t.array = nullptr;
        }
        t.width = t.height = 0;
        t.desc = {};
    }

    static bool createArray(Texture2D& t, PixelFormat fmt, int w, int h) {
        auto ch = channelDesc(fmt);
        bool ret = cudaMallocArray(&t.array, &ch, w, h, cudaArrayDefault) == cudaSuccess; 
        checkCUDAError("cudaMallocArray"); 
        return ret;
    }

    static bool createTextureObject(Texture2D& t, const Sampler& s) {
        cudaResourceDesc res{};
        res.resType = cudaResourceTypeArray;
        res.res.array.array = t.array;

        cudaTextureDesc tex{};
        tex.addressMode[0] = s.addressU;
        tex.addressMode[1] = s.addressV;
        tex.addressMode[2] = s.addressV; // unused for 2D
        tex.filterMode = s.filter;
        tex.normalizedCoords = s.normalizedCoords ? 1 : 0;
        tex.readMode = s.readMode;

        return cudaCreateTextureObject(&t.texObj, &res, &tex, nullptr) == cudaSuccess;
    }

    static void clear(Texture2D& t) {
        // helper to release partially-initialized resources
        if (t.texObj) {t.texObj = 0; }
        if (t.array) {t.array = nullptr; }
    }

}
